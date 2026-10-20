"""Spike detection and spike-level features (Allen's ``EphysSweepFeatureExtractor``).

Units follow Allen: ``t`` in s, ``v`` in mV, ``i`` in pA, dV/dt in V/s. A sweep
of ``T`` samples gives per-spike buffers of static length K (``max_spikes``);
detected spikes are packed at the front, missing entries are -1 (indices) or
NaN (values). Batches of equal-length sweeps work with ``jax.vmap``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from ._ops import (
    compact,
    masked_mean,
    next_true,
    prev_true,
    segment_argmax,
    segment_argmin,
    segment_first,
    segment_last,
    segments,
    shift_left,
    shift_right,
    take,
    window_argmax,
)
from .dvdt import calculate_dvdt

# Codes of the ``isi_type`` spike feature (Allen: "direct", "detour", NaN).
ISI_MISSING, ISI_DIRECT, ISI_DETOUR = -1, 0, 1
# First capacity tried by ``detect_spikes(..., max_spikes=None)``; grown 4x as needed.
AUTO_MAX_SPIKES = 256


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Spikes:
    """Detected spikes of one sweep, together with the sweep they come from.

    ``t``, ``v``, ``i`` (optional) and ``dvdt`` are the ``[T]`` traces (s, mV,
    pA, V/s; ``dvdt`` is Bessel-filtered), ``start``/``end`` the analysis window
    (s). The other fields are ``[K]`` buffers: sample indices of each spike's
    threshold, peak, trough, upstroke, downstroke, fast trough, afterdepolarization
    and slow trough (-1 where missing), its ``isi_type`` code and whether it is
    ``clipped`` by the end of the window. ``overflow`` is set if there were more
    than K putative spikes (upward crossings of ``dv_cutoff``); the buffers are
    then incomplete. Spikes of a batch (``jax.vmap``) have a leading batch axis.
    """

    t: Array
    v: Array
    i: Array | None
    dvdt: Array
    start: Array
    end: Array
    threshold: Array
    peak: Array
    trough: Array
    upstroke: Array
    downstroke: Array
    fast_trough: Array
    adp: Array
    slow_trough: Array
    isi_type: Array
    clipped: Array
    overflow: Array

    @property
    def valid(self) -> Array:
        """``[K]`` mask of the detected spikes."""
        return self.threshold >= 0

    @property
    def n_spikes(self) -> Array:
        """Number of detected spikes."""
        return jnp.sum(self.valid, axis=-1)


def find_time_index(t: Array, t_0: Array) -> Array:
    """First index with ``t >= t_0`` (Allen's ``find_time_index``, clamped to T - 1)."""
    return jnp.minimum(jnp.searchsorted(t, t_0, side="left"), t.shape[0] - 1)


def _next_or(idx: Array, default: Array) -> Array:
    """``idx[k + 1]`` where it is a valid index, ``default`` otherwise."""
    nxt = shift_left(idx, -1)
    return jnp.where(nxt >= 0, nxt, default)


def _putative_spikes(dvdt_window, start_idx, end_idx, dv_cutoff, max_spikes):
    """Upward dV/dt crossings of ``dv_cutoff`` (``detect_putative_spikes``)."""
    x = jnp.arange(dvdt_window.shape[0])
    above = dvdt_window >= dv_cutoff
    crossing = ~above & shift_left(above, False)
    crossing = crossing & (x >= start_idx) & (x <= end_idx - 2)
    candidates = jnp.nonzero(crossing, size=max_spikes, fill_value=-1)[0]
    # Only keep a candidate if dV/dt dropped below zero since the previous one.
    in_window = (x >= start_idx) & (x < end_idx)
    next_negative = next_true(in_window & (dvdt_window < 0))
    previous = shift_right(candidates, -1)
    keep = (candidates >= 0) & ((previous < 0) | (next_negative[previous] < candidates))
    return compact(keep, candidates)[0], jnp.sum(crossing) > max_spikes


def _filter_putative_spikes(v, dvdt, spikes, peaks, min_height, min_peak):
    """Merge spikes without a dV/dt dip in between, then apply height criteria
    (``filter_putative_spikes``)."""
    k = jnp.arange(spikes.shape[0])
    valid = spikes >= 0
    next_spike = shift_left(spikes, -1)
    has_dip = (next_spike < 0) | (next_true(dvdt < 0)[peaks] < next_spike)
    spikes = compact(valid & ((k == 0) | shift_right(has_dip, True)), spikes)[0]
    peaks = compact(valid & has_dip, peaks)[0]
    peak_v = take(v, peaks, -jnp.inf)
    height = peak_v - take(v, spikes, jnp.inf)
    keep = (spikes >= 0) & (peak_v >= min_peak) & (height >= min_height)
    return compact(keep, spikes, peaks)


def _putative_upstrokes(
    v,
    dvdt,
    dvdt_window,
    start_idx,
    end_idx,
    dv_cutoff,
    min_height,
    min_peak,
    max_spikes,
):
    """Putative spikes, their peaks and upstrokes (first stages of detection)."""
    size = v.shape[0]
    spikes, overflow = _putative_spikes(
        dvdt_window, start_idx, end_idx, dv_cutoff, max_spikes
    )
    peak_window = segments(spikes, _next_or(spikes, end_idx), size)
    peaks = segment_argmax(v, peak_window, max_spikes)
    spikes, peaks = _filter_putative_spikes(
        v, dvdt, spikes, peaks, min_height, min_peak
    )
    ups = segment_argmax(dvdt, segments(spikes, peaks, size), max_spikes)
    return spikes, peaks, ups, overflow


def _upstroke_target(dvdt, upstrokes, thresh_frac):
    return thresh_frac * masked_mean(take(dvdt, upstrokes), upstrokes >= 0)


def _thresholds_from_upstrokes(dvdt, upstrokes, lower, target):
    """Last sample in ``(lower, upstroke]`` with ``dvdt <= target``, else ``lower``
    (``refine_threshold_indexes``)."""
    below = prev_true(dvdt <= target)[jnp.maximum(upstrokes, 0)]
    return jnp.where(upstrokes < 0, -1, jnp.where(below > lower, below, lower))


def _check_thresholds_and_peaks(
    t, v, dvdt, thr, peaks, ups, end_idx, max_interval, thresh_frac
):
    """Allen's ``check_thresholds_and_peaks``: merge overlapping spikes, repair
    spikes whose peak comes too late after threshold, mark a clipped last spike."""
    size, num = v.shape[0], thr.shape[0]
    valid = thr >= 0
    next_thr = shift_left(thr, -1)
    overlap = (next_thr >= 0) & (next_thr <= peaks + 1)
    thr = compact(valid & ~shift_right(overlap, False), thr)[0]
    peaks, ups = compact(valid & ~overlap, peaks, ups)
    valid = thr >= 0

    # Peaks too long after threshold: first try to find a later threshold ...
    too_long = valid & (take(t, peaks) - take(t, thr) >= max_interval)
    target = _upstroke_target(dvdt, ups, thresh_frac)
    lower = find_time_index(t, take(t, peaks) - max_interval)
    new_thr = _thresholds_from_upstrokes(dvdt, ups, lower, target)
    fix_thr = too_long & (new_thr > lower)
    # ... otherwise look for an earlier peak within 2 * max_interval of threshold
    # (rare; these windows may overlap, so they are searched one by one).
    refind = too_long & ~fix_thr
    (order,) = compact(refind, jnp.arange(num))
    hi = jnp.searchsorted(t, take(t, thr) + 2 * max_interval, side="left")

    def refind_peak(j, peaks):
        k = order[j]
        return peaks.at[k].set(window_argmax(v, thr[k], hi[k]))

    new_peaks = lax.fori_loop(0, jnp.sum(refind), refind_peak, peaks)
    next_thr_t = take(t, shift_left(thr, -1), jnp.inf)
    new_peak_t = take(t, new_peaks)
    peak_ok = (new_peak_t - take(t, thr) < max_interval) & (new_peak_t < next_thr_t)
    thr = jnp.where(fix_thr, new_thr, thr)
    peaks = jnp.where(refind & peak_ok, new_peaks, peaks)
    thr, peaks, ups = compact(valid & ~(refind & ~peak_ok), thr, peaks, ups)

    # The last spike is clipped if v does not return to threshold + 1 mV before end.
    n = jnp.sum(thr >= 0)
    last = jnp.maximum(n - 1, 0)
    x = jnp.arange(size)
    returned = jnp.any((x >= peaks[last]) & (x <= end_idx) & (v <= v[thr[last]] + 1.0))
    clipped = (jnp.arange(num) == n - 1) & ~returned
    return thr, peaks, clipped


def _trough_details(t, v, dvdt, dvdt_heavy, thr, peaks, clipped, end_idx):
    """Allen's ``analyze_trough_details`` with its defaults: fast trough, ADP,
    slow trough and ISI type; spikes without a fast trough become clipped."""
    size, num = v.shape[0], thr.shape[0]
    ok = (thr >= 0) & ~clipped
    next_ok = jnp.where(shift_left(ok, False), shift_left(thr, -1), end_idx)
    # Fast trough: where dV/dt has recovered to 1% of the (most negative) downstroke.
    down = segment_argmin(dvdt, segments(jnp.where(ok, peaks, -1), next_ok, size), num)
    target = 0.01 * take(dvdt, down)
    seg = segments(down, next_ok, size)
    term = segment_first(dvdt >= jnp.append(target, jnp.nan)[seg], seg, num)
    fast = jnp.where(ok, term, -1)
    has_fast = fast >= 0

    # Afterdepolarization: dV/dt (1 kHz filter) rises above 0.5 V/s within 2 ms of
    # the fast trough, then reverses, at most 10 mV above and 5 ms after it.
    cross = next_true(dvdt_heavy >= 0.5)[jnp.maximum(fast, 0)]
    near = has_fast & (cross < next_ok) & (take(t, cross) - take(t, fast) < 0.002)
    zero = next_true(dvdt_heavy <= 0)[jnp.minimum(cross, size - 1)]
    adp = jnp.where(near & (zero < next_ok), zero, -1)
    adp_min = segment_argmin(v, segments(adp, next_ok, size), num)
    is_adp = (
        (adp >= 0)
        & (take(v, adp) - take(v, adp_min) >= 0.5)
        & (take(v, adp) - take(v, fast) <= 10.0)
        & (take(t, adp) - take(t, fast) <= 0.005)
    )
    # Otherwise a "detour" reset if v keeps dropping by >= 0.5 mV after the fast trough.
    fast_min = segment_argmin(v, segments(fast, next_ok, size), num)
    detour = ~is_adp & has_fast & (take(v, fast) - take(v, fast_min) >= 0.5)

    isi_type = jnp.where(
        is_adp | detour, ISI_DETOUR, jnp.where(has_fast, ISI_DIRECT, ISI_MISSING)
    )
    # Allen drops the ADP and slow trough of the last spike as unreliable.
    not_last = jnp.arange(num) < jnp.sum(thr >= 0) - 1
    adp = jnp.where(is_adp & not_last, adp, -1)
    slow = jnp.where(is_adp, adp_min, jnp.where(detour, fast_min, -1))
    slow = jnp.where(not_last, slow, -1)
    return fast, adp, slow, isi_type, clipped | (ok & ~has_fast)


@partial(jax.jit, static_argnames=("filter", "max_spikes"))
def _detect_spikes(
    t,
    v,
    i,
    start,
    end,
    *,
    filter,
    dv_cutoff,
    max_interval,
    min_height,
    min_peak,
    thresh_frac,
    max_spikes,
) -> Spikes:
    # Compute in the default float precision (float64 if ``jax_enable_x64`` is set).
    t, v = jnp.asarray(t, dtype=float), jnp.asarray(v, dtype=float)
    i = None if i is None else jnp.asarray(i, dtype=float)
    size = t.shape[0]
    start = t[0] if start is None else jnp.asarray(start, dtype=t.dtype)
    end = t[-1] if end is None else jnp.asarray(end, dtype=t.dtype)
    start_idx, end_idx = find_time_index(t, start), find_time_index(t, end)

    # Allen detects putative spikes on dV/dt filtered over the analysis window only.
    dvdt = calculate_dvdt(v, t, filter)
    dvdt_window = calculate_dvdt(v, t, filter, window=(start_idx, end_idx))
    spikes, peaks, ups, overflow = _putative_upstrokes(
        v,
        dvdt,
        dvdt_window,
        start_idx,
        end_idx,
        dv_cutoff,
        min_height,
        min_peak,
        max_spikes,
    )
    target = _upstroke_target(dvdt, ups, thresh_frac)
    thr = _thresholds_from_upstrokes(dvdt, ups, shift_right(ups, 0), target)
    thr, peaks, clipped = _check_thresholds_and_peaks(
        t, v, dvdt, thr, peaks, ups, end_idx, max_interval, thresh_frac
    )

    ups = segment_argmax(dvdt, segments(thr, peaks, size), max_spikes)
    trough_window = segments(
        jnp.where(clipped, -1, peaks),
        jnp.where(clipped, -1, _next_or(thr, end_idx)),
        size,
    )
    troughs = segment_argmin(v, trough_window, max_spikes)
    downs = segment_argmin(dvdt, segments(peaks, troughs, size), max_spikes)
    dvdt_heavy = calculate_dvdt(v, t, 1.0)
    fast, adp, slow, isi_type, clipped = _trough_details(
        t, v, dvdt, dvdt_heavy, thr, peaks, clipped, end_idx
    )
    return Spikes(
        t=t,
        v=v,
        i=i,
        dvdt=dvdt,
        start=start,
        end=end,
        threshold=thr,
        peak=peaks,
        trough=troughs,
        upstroke=ups,
        downstroke=downs,
        fast_trough=fast,
        adp=adp,
        slow_trough=slow,
        isi_type=isi_type,
        clipped=clipped & (thr >= 0),
        overflow=overflow,
    )


def _check_cutoff(t, start, end, filter: float) -> None:
    """Raise like Allen's ``calculate_dvdt`` if ``filter`` is not below the Nyquist
    frequency of the trace or of the analysis window (if evenly sampled, with the
    tolerance of ``calculate_dvdt``)."""
    t = np.asarray(t)
    eps = np.finfo(t.dtype).eps if np.issubdtype(t.dtype, np.floating) else 0.0
    t = t.astype(float)
    lo = 0 if start is None else min(np.searchsorted(t, start), t.size - 1)
    hi = t.size - 1 if end is None else min(np.searchsorted(t, end), t.size - 1)
    for a, b in ((0, t.size - 1), (lo, hi)):
        if b <= a:
            continue
        step = (t[b] - t[a]) / (b - a)
        if 0 <= filter * 1e3 * 2.0 * step < 1:  # cutoff / Nyquist frequency
            continue
        tol = 1e-8 + 1e-5 * abs(step) + 2 * eps * np.max(np.abs(t))
        if np.all(np.abs(np.diff(t[a : b + 1]) - step) <= tol):
            raise ValueError(
                f"filter={filter} kHz is not below the Nyquist frequency of the "
                f"sampling rate ({0.5e-3 / step:.1f} kHz); lower it or pass filter=None"
            )


def detect_spikes(
    t: Array,
    v: Array,
    i: Array | None = None,
    start: float | Array | None = None,
    end: float | Array | None = None,
    *,
    filter: float | None = 10.0,
    dv_cutoff: float | Array = 20.0,
    max_interval: float | Array = 0.005,
    min_height: float | Array = 2.0,
    min_peak: float | Array = -30.0,
    thresh_frac: float | Array = 0.05,
    max_spikes: int | None = 256,
) -> Spikes:
    """Detect spikes like Allen's ``EphysSweepFeatureExtractor.process_spikes``.

    Args:
        t: Time (s), ``[T]``, increasing.
        v: Membrane potential (mV), ``[T]``.
        i: Injected current (pA), ``[T]``, optional.
        start, end: Analysis window (s); default: the whole sweep.
        filter: Bessel low-pass cutoff (kHz) for dV/dt, ``None`` to disable. Must
            be below the Nyquist frequency; if ``t`` is traced (``jit``/``vmap``),
            this cannot be checked and a higher cutoff gives NaN dV/dt (no spikes).
        dv_cutoff: dV/dt (V/s) a putative spike has to cross.
        max_interval: Maximum threshold-to-peak time (s).
        min_height: Minimum threshold-to-peak height (mV).
        min_peak: Minimum peak voltage (mV).
        thresh_frac: Fraction of the mean upstroke that defines the threshold.
        max_spikes: Static capacity K of the per-spike buffers. It must hold every
            upward crossing of ``dv_cutoff`` in the window (the putative spikes,
            before they are merged and filtered). ``None`` starts at 256 and grows
            K 4x until they fit; this needs concrete (non-traced) inputs.

    Returns:
        ``Spikes`` with ``[K]`` buffers. If K is too small, a ``ValueError`` is
        raised outside of ``jit``/``vmap``; inside, ``Spikes.overflow`` is set and
        the results are incomplete.

    The parameters and their defaults are those of Allen's extractor. Compiled
    once per input shape, dtype, ``filter`` and ``max_spikes``; other arguments
    can change without recompiling. With float64 (``jax_enable_x64``) the results
    match Allen exactly; in float32 they deviate slightly (a few spike features
    move by a sample, rarely more; sweep features by about 2e-4, ``tau`` and
    ``sag`` up to 1.5e-3).
    """
    shapes = [jnp.shape(x) for x in (t, v, i) if x is not None]
    if len(shapes[0]) != 1 or any(shape != shapes[0] for shape in shapes):
        raise ValueError(
            f"t, v and i must be 1D arrays of equal length (use jax.vmap for batches), "
            f"got shapes {shapes}"
        )
    if filter and not any(isinstance(x, jax.core.Tracer) for x in (t, start, end)):
        _check_cutoff(t, start, end, filter)
    t, v = jnp.asarray(t), jnp.asarray(v)
    i = None if i is None else jnp.asarray(i)
    detect = partial(
        _detect_spikes,
        t,
        v,
        i,
        start,
        end,
        filter=filter,
        dv_cutoff=dv_cutoff,
        max_interval=max_interval,
        min_height=min_height,
        min_peak=min_peak,
        thresh_frac=thresh_frac,
    )
    if max_spikes is not None:
        sp = detect(max_spikes=max_spikes)
        if not isinstance(sp.overflow, jax.core.Tracer) and sp.overflow:
            raise ValueError(
                f"more than max_spikes={max_spikes} putative spikes; "
                "increase max_spikes or use max_spikes=None"
            )
        return sp
    capacity = AUTO_MAX_SPIKES
    while True:
        sp = detect(max_spikes=capacity)
        if isinstance(sp.overflow, jax.core.Tracer):
            raise ValueError(
                "max_spikes=None needs concrete inputs; pass an int inside jit, vmap or grad"
            )
        if not sp.overflow:
            return sp
        capacity *= 4


def _widths(sp: Spikes) -> Array:
    """Width at half height (``find_widths``, using the fast trough), in s."""
    size, num = sp.v.shape[0], sp.threshold.shape[0]
    use = (sp.fast_trough >= 0) & ~sp.clipped
    v_thr, v_peak = take(sp.v, sp.threshold), take(sp.v, sp.peak)
    v_trough = take(sp.v, sp.fast_trough)
    level = v_trough + (v_peak - v_trough) / 2.0
    level = jnp.where(level < v_thr, v_thr + (v_peak - v_thr) / 2.0, level)
    level = jnp.where(use, level, jnp.nan)
    rise = segments(jnp.where(use, sp.threshold + 1, -1), sp.peak + 1, size)
    left = segment_last(sp.v <= jnp.append(level, jnp.nan)[rise], rise, num)
    fall = segments(jnp.where(use, sp.peak, -1), sp.fast_trough, size)
    right = segment_first(sp.v <= jnp.append(level, jnp.nan)[fall], fall, num)
    ok = (left >= 0) & (right >= 0)
    return jnp.where(ok, take(sp.t, right) - take(sp.t, left), jnp.nan)


def _point_features(sp: Spikes, name: str, idx: Array) -> dict[str, Array]:
    out = {
        f"{name}_index": idx,
        f"{name}_t": take(sp.t, idx),
        f"{name}_v": take(sp.v, idx),
    }
    if sp.i is not None:
        out[f"{name}_i"] = take(sp.i, idx)
    return out


@jax.jit
def spike_features(sp: Spikes) -> dict[str, Array]:
    """The columns of Allen's spike table, as ``[K]`` arrays.

    ``*_index`` are sample indices (-1 where undefined), ``*_t`` times (s),
    ``*_v`` voltages (mV), ``*_i`` currents (pA, only if ``sp.i`` is given),
    ``upstroke``/``downstroke`` dV/dt (V/s), ``width`` the width at half height
    (s); undefined values are NaN. ``clipped`` is boolean, ``isi_type`` uses
    ``ISI_DIRECT``, ``ISI_DETOUR`` and ``ISI_MISSING``.
    """
    out = {"threshold_index": sp.threshold, "clipped": sp.clipped}
    for name, idx in (
        ("threshold", sp.threshold),
        ("peak", sp.peak),
        ("trough", sp.trough),
    ):
        out |= _point_features(sp, name, idx)
    for name, idx in (("upstroke", sp.upstroke), ("downstroke", sp.downstroke)):
        out[f"{name}_index"] = idx
        out[name] = take(sp.dvdt, idx)
        out[f"{name}_t"] = take(sp.t, idx)
        out[f"{name}_v"] = take(sp.v, idx)
    out["isi_type"] = sp.isi_type
    for name in ("fast_trough", "adp", "slow_trough"):
        out |= _point_features(sp, name, getattr(sp, name))
    out["width"] = _widths(sp)
    out["upstroke_downstroke_ratio"] = out["upstroke"] / -out["downstroke"]
    return out
