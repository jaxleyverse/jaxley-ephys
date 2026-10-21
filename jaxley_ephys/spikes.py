"""Spike detection and spike-level features (Allen's ``EphysSweepFeatureExtractor``).

Units follow Allen: ``t`` in s, ``v`` in mV, ``i`` in pA, dV/dt in V/s. A sweep
of ``T`` samples gives per-spike buffers of static length K (``max_spikes``);
detected spikes are packed at the front, missing entries are -1 (indices) or
NaN (values). Batches of equal-length sweeps work with ``jax.vmap``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from ._ops import (
    compact,
    fill_nan,
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
from .dvdt import _BESSEL4_POLES, _PADLEN, calculate_dvdt

# Codes of the ``isi_type`` spike feature (Allen: "direct", "detour", NaN).
ISI_MISSING, ISI_DIRECT, ISI_DETOUR = -1, 0, 1
# First capacity tried by ``detect_spikes(..., max_spikes=None)``; grown 4x as needed.
AUTO_MAX_SPIKES = 256


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Spikes:
    """Detected spikes of one sweep, together with the sweep they come from.

    ``t``, ``v``, ``i`` (optional) and ``dvdt`` are the ``[T]`` traces (s, mV,
    pA, V/s; ``dvdt`` is Bessel-filtered), NaN after ``t_cutoff`` (not analysed),
    ``start``/``end`` the analysis window (s). The other fields are ``[K]``
    buffers: sample indices of each spike's
    threshold, peak, trough, upstroke, downstroke, fast trough, afterdepolarization
    and slow trough (-1 where missing), its ``isi_type`` code and whether it is
    ``clipped`` by the end of the window. ``overflow`` is set if there were more
    than K putative spikes (upward crossings of ``dv_cutoff``); the buffers are
    then incomplete. ``thresh_frac`` is the detection parameter that defines the
    thresholds, needed for their times between samples; with ``interpolate``,
    features use these times instead of Allen's sample times. ``min_peak`` and
    ``count_width`` define the soft spike count behind the gradients of counts
    (see ``detect_spikes``). Spikes of a batch (``jax.vmap``) have a leading batch
    axis.
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
    thresh_frac: Array
    min_peak: Array
    count_width: Array
    interpolate: bool = field(default=False, metadata={"static": True})

    @property
    def valid(self) -> Array:
        """``[K]`` mask of the detected spikes."""
        return self.threshold >= 0

    @property
    def n_spikes(self) -> Array:
        """Number of detected spikes."""
        return jnp.sum(self.valid, axis=-1)


def find_time_index(t: Array, t_0: Array) -> Array:
    """First index with ``t >= t_0`` (Allen's ``find_time_index``), clamped to the
    last sample of the sweep (trailing NaN times are not part of it)."""
    last = jnp.searchsorted(t, jnp.inf) - 1  # NaN times sort last
    return jnp.minimum(jnp.searchsorted(t, t_0, side="left"), last)


def _next_or(idx: Array, default: Array) -> Array:
    """``idx[k + 1]`` where it is a valid index, ``default`` otherwise."""
    nxt = shift_left(idx, -1)
    return jnp.where(nxt >= 0, nxt, default)


def _putative_spikes(
    dvdt_window: Array,
    start_idx: Array,
    end_idx: Array,
    dv_cutoff: Array,
    max_spikes: int,
) -> tuple[Array, Array]:
    """Upward dV/dt crossings of ``dv_cutoff`` (``detect_putative_spikes``) and
    whether there were more than ``max_spikes``."""
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


def _filter_putative_spikes(
    v: Array,
    dvdt: Array,
    spikes: Array,
    peaks: Array,
    min_height: Array,
    min_peak: Array,
) -> tuple[Array, Array]:
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
    v: Array,
    dvdt: Array,
    dvdt_window: Array,
    start_idx: Array,
    end_idx: Array,
    dv_cutoff: Array,
    min_height: Array,
    min_peak: Array,
    max_spikes: int,
) -> tuple[Array, Array, Array, Array]:
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


def _upstroke_target(dvdt: Array, upstrokes: Array, thresh_frac: Array) -> Array:
    """dV/dt that defines the threshold: ``thresh_frac`` times the mean upstroke."""
    return thresh_frac * masked_mean(take(dvdt, upstrokes), upstrokes >= 0)


def _thresholds_from_upstrokes(
    dvdt: Array, upstrokes: Array, lower: Array, target: Array
) -> Array:
    """Last sample in ``(lower, upstroke]`` with ``dvdt <= target``, else ``lower``
    (``refine_threshold_indexes``)."""
    below = prev_true(dvdt <= target)[jnp.maximum(upstrokes, 0)]
    return jnp.where(upstrokes < 0, -1, jnp.where(below > lower, below, lower))


def _check_thresholds_and_peaks(
    t: Array,
    v: Array,
    dvdt: Array,
    thr: Array,
    peaks: Array,
    ups: Array,
    end_idx: Array,
    max_interval: Array,
    thresh_frac: Array,
) -> tuple[Array, Array, Array]:
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


def _trough_details(
    t: Array,
    v: Array,
    dvdt: Array,
    dvdt_heavy: Array,
    thr: Array,
    peaks: Array,
    clipped: Array,
    end_idx: Array,
) -> tuple[Array, Array, Array, Array, Array]:
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


@partial(jax.jit, static_argnames=("filter", "max_spikes", "interpolate"))
def _detect_spikes(
    t: Array,
    v: Array,
    i: Array | None,
    start: Array | None,
    end: Array | None,
    *,
    filter: float | None,
    dv_cutoff: Array,
    max_interval: Array,
    min_height: Array,
    min_peak: Array,
    thresh_frac: Array,
    t_cutoff: Array,
    max_spikes: int,
    interpolate: bool,
    count_width: Array,
) -> Spikes:
    """The compiled part of ``detect_spikes`` (same arguments)."""
    # Compute in the default float precision (float64 if ``jax_enable_x64`` is set).
    t, v = jnp.asarray(t, dtype=float), jnp.asarray(v, dtype=float)
    i = None if i is None else jnp.asarray(i, dtype=float)
    # Samples after t_cutoff are not part of the sweep: like trailing NaN times
    # (e.g. padding), they become NaN and do not affect any result.
    outside = ~(t <= t_cutoff)
    t, v = jnp.where(outside, jnp.nan, t), jnp.where(outside, jnp.nan, v)
    i = None if i is None else jnp.where(outside, jnp.nan, i)
    size = t.shape[0]
    start = t[0] if start is None else jnp.asarray(start, dtype=t.dtype)
    end = (
        t[jnp.searchsorted(t, jnp.inf) - 1]
        if end is None
        else jnp.asarray(end, t.dtype)
    )
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
    # The 1 kHz dV/dt only matters after spikes (afterdepolarizations): skip it
    # for sweeps without spikes (under vmap, both branches run).
    dvdt_heavy = lax.cond(
        jnp.any(thr >= 0),
        lambda: calculate_dvdt(v, t, 1.0),
        lambda: jnp.full_like(dvdt, jnp.nan),
    )
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
        thresh_frac=jnp.asarray(thresh_frac, t.dtype),
        min_peak=jnp.asarray(min_peak, t.dtype),
        count_width=jnp.asarray(count_width, t.dtype),
        interpolate=interpolate,
    )


# Samples after the analysis window only reach the results through the end
# transient of the backward filter passes, which decays below this fraction.
_TRANSIENT_DECAY = 1e-20
# Analysed lengths are rounded up to a multiple of this, so that sweeps with
# similar windows share one compiled shape.
_LENGTH_STEP = 2**14


def _host_time(t: Array) -> tuple[np.ndarray, float]:
    """The times of the sweep as a float64 NumPy array (without trailing NaN
    times), and the rounding of their original precision."""
    t = np.asarray(t)
    eps = np.finfo(t.dtype).eps if np.issubdtype(t.dtype, np.floating) else 0.0
    t = t.astype(float)
    t = t[: np.searchsorted(t, np.inf)]  # NaN times sort last
    return t, 2 * eps * np.max(np.abs(t), initial=0.0)


def _sampling_interval(t: np.ndarray, rounding: float) -> float | None:
    """The sampling interval if ``t`` is evenly sampled (with the tolerance of
    ``calculate_dvdt``), else None."""
    if t.size < 2:
        return None
    step = (t[-1] - t[0]) / (t.size - 1)
    tol = 1e-8 + 1e-5 * abs(step) + rounding
    return step if np.all(np.abs(np.diff(t) - step) <= tol) else None


def _check_cutoff(
    t: np.ndarray,
    rounding: float,
    start: float | None,
    end: float | None,
    filter: float,
) -> None:
    """Raise like Allen's ``calculate_dvdt`` if ``filter`` is not below the Nyquist
    frequency of the trace or of the analysis window (where evenly sampled)."""
    lo = 0 if start is None else min(np.searchsorted(t, start), t.size - 1)
    hi = t.size - 1 if end is None else min(np.searchsorted(t, end), t.size - 1)
    for segment in (t, t[lo : hi + 1]):
        if segment.size < 2:
            continue
        step = (segment[-1] - segment[0]) / (segment.size - 1)
        if 0 <= filter * 1e3 * 2.0 * step < 1:  # cutoff / Nyquist frequency
            continue
        if _sampling_interval(segment, rounding) is not None:
            raise ValueError(
                f"filter={filter} kHz is not below the Nyquist frequency of the "
                f"sampling rate ({0.5e-3 / step:.1f} kHz); lower it or pass filter=None"
            )


def _default_cutoff(
    t: np.ndarray,
    rounding: float,
    end: float | None,
    filter: float | None,
    max_interval: float,
) -> float | None:
    """The earliest ``t_cutoff`` that cannot change any result for an analysis
    window ending at ``end``: the reach of the peak search (``2 * max_interval``)
    and of the end transients of the filters (the 1 kHz one is the slowest, unless
    ``filter`` is lower) after ``end``. None if unknown (no ``end``, uneven sampling).
    """
    step = _sampling_interval(t, rounding)
    cutoff = min(filter, 1.0) if filter else 1.0  # kHz
    wn = cutoff * 1e3 * 2.0 * step if step else 1.0  # cutoff / Nyquist frequency
    if end is None or not 0 < wn < 1:
        return None
    warped = 4.0 * np.tan(np.pi * wn / 2.0)  # as in bessel4_sos, slowest pole:
    radius = max(abs((4.0 + p * warped) / (4.0 - p * warped)) for p in _BESSEL4_POLES)
    transient = np.log(_TRANSIENT_DECAY) / np.log(radius) + _PADLEN + 2  # samples
    return float(end) + 2 * max_interval + transient * step


def _analysed_length(t: np.ndarray, t_cutoff: float) -> int:
    """Number of leading samples to analyse: those up to ``t_cutoff``, rounded up
    (the extra ones are masked) so that similar cutoffs share a compiled shape."""
    n = max(int(np.searchsorted(t, t_cutoff, side="right")), 1)
    return int(min(t.size, np.ceil(n / _LENGTH_STEP) * _LENGTH_STEP))


def _analysed(sp: Spikes) -> Spikes:
    """``sp`` with concrete traces cut to the analysed samples (before the NaN
    tail, rounded up like in ``detect_spikes``): features do not depend on the
    tail, so they need not process it."""
    if isinstance(sp.t, jax.core.Tracer) or jnp.ndim(sp.t) != 1:
        return sp
    size = sp.t.shape[0]
    n = max(int(np.searchsorted(np.asarray(sp.t), np.inf)), 1)  # NaN times sort last
    n = int(min(size, np.ceil(n / _LENGTH_STEP) * _LENGTH_STEP))
    if n == size:
        return sp

    def cut(x):
        return None if x is None else x[:n]

    return replace(sp, t=cut(sp.t), v=cut(sp.v), i=cut(sp.i), dvdt=cut(sp.dvdt))


def _pad(sp: Spikes, size: int) -> Spikes:
    """``sp`` with its traces padded with NaN to ``size`` samples."""

    def pad(x):
        if x is None:
            return None
        return jnp.concatenate([x, jnp.full(size - x.shape[0], jnp.nan, x.dtype)])

    return replace(sp, t=pad(sp.t), v=pad(sp.v), i=pad(sp.i), dvdt=pad(sp.dvdt))


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
    t_cutoff: float | Array | None = None,
    max_spikes: int | None = 256,
    interpolate: bool = False,
    count_width: float | Array = 5.0,
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
        t_cutoff: Time (s) after which samples are not analysed, as if the
            recording ended there; they are NaN in ``Spikes``. Default: 20 ms
            after ``end``, the reach of the peak search and of the filters'
            transients, so that no result changes (results match Allen for any
            later cutoff). ``float("inf")`` analyses the whole sweep, as does the
            default if ``t`` is traced. Trailing NaN times (e.g. padding to batch
            sweeps of unequal length) are treated the same way.
        max_spikes: Static capacity K of the per-spike buffers. It must hold every
            upward crossing of ``dv_cutoff`` in the window (the putative spikes,
            before they are merged and filtered). ``None`` starts at 256 and grows
            K 4x until they fit; this needs concrete (non-traced) inputs.
        interpolate: Event times (thresholds, peaks, troughs, ...) between
            samples: where the defining level is crossed (linear interpolation)
            or at the vertex of a parabola through the extremum. They change
            smoothly with ``v``, within a sample of Allen's. ``False`` (the
            default) keeps Allen's sample times; their gradients are still those
            of the interpolated times. All features of the ``Spikes`` use them.
        count_width: Width (mV) of the sigmoid in the soft spike count that
            gives spike counts (``avg_rate`` and the f-I slope) their gradients:
            the number of upward crossings of ``min_peak``, counted smoothly.
            Wider reaches models further from spiking; it never changes values.

    Returns:
        ``Spikes`` with ``[K]`` buffers. If K is too small, a ``ValueError`` is
        raised outside of ``jit``/``vmap``; inside, ``Spikes.overflow`` is set and
        the results are incomplete.

    Samples after ``t_cutoff`` are skipped if ``t`` and ``t_cutoff`` are concrete
    (not traced), which makes long sweeps with short windows fast; otherwise they
    are masked.

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
    size = n = shapes[0][0]
    if not any(isinstance(x, jax.core.Tracer) for x in (t, start, end)):
        t_host, rounding = _host_time(t)
        if filter:
            _check_cutoff(t_host, rounding, start, end, filter)
        if t_cutoff is None and not isinstance(max_interval, jax.core.Tracer):
            t_cutoff = _default_cutoff(
                t_host, rounding, end, filter, float(max_interval)
            )
        if t_cutoff is not None and not isinstance(t_cutoff, jax.core.Tracer):
            # Skip the samples after the cutoff (and pad the results back).
            n = _analysed_length(t_host, float(t_cutoff))
            t, v, i = t[:n], v[:n], None if i is None else i[:n]
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
        t_cutoff=jnp.inf if t_cutoff is None else t_cutoff,
        interpolate=interpolate,
        count_width=count_width,
    )
    if max_spikes is not None:
        sp = detect(max_spikes=max_spikes)
        if not isinstance(sp.overflow, jax.core.Tracer) and sp.overflow:
            raise ValueError(
                f"more than max_spikes={max_spikes} putative spikes; "
                "increase max_spikes or use max_spikes=None"
            )
    else:
        capacity = AUTO_MAX_SPIKES
        while True:
            sp = detect(max_spikes=capacity)
            if isinstance(sp.overflow, jax.core.Tracer):
                raise ValueError(
                    "max_spikes=None needs concrete inputs; "
                    "pass an int inside jit, vmap or grad"
                )
            if not sp.overflow:
                break
            capacity *= 4
    return _pad(sp, size) if n < size else sp


def _crossing(s: Array, k: Array, level: Array) -> Array:
    """Where ``s`` crosses ``level`` between samples ``k`` and ``k + 1``, as a
    fraction of the interval (linear interpolation, clipped to [0, 1]); 0 where
    undefined. Undefined inputs are replaced before dividing, so that gradients
    stay finite."""
    s0, s1 = take(s, k), take(s, k + 1)
    ok = (k >= 0) & jnp.isfinite(s0) & jnp.isfinite(s1) & jnp.isfinite(level)
    ok = ok & (s0 != s1)
    s0, s1 = jnp.where(ok, s0, 0.0), jnp.where(ok, s1, 1.0)
    frac = (jnp.where(ok, level, 0.0) - s0) / (s1 - s0)
    return jnp.where(ok, jnp.clip(frac, 0.0, 1.0), 0.0)


def _vertex(s: Array, k: Array) -> Array:
    """Offset (samples, clipped to +-0.5) of the vertex of the parabola through
    ``s[k - 1]``, ``s[k]`` and ``s[k + 1]``, i.e. of the extremum at ``k``
    between samples; 0 where undefined."""
    a, b, c = take(s, k - 1), take(s, k), take(s, k + 1)
    curvature = a - 2.0 * b + c
    ok = (k >= 1) & jnp.isfinite(curvature) & (curvature != 0)
    a, c = jnp.where(ok, a, 0.0), jnp.where(ok, c, 0.0)
    offset = 0.5 * (a - c) / jnp.where(ok, curvature, 1.0)
    return jnp.where(ok, jnp.clip(offset, -0.5, 0.5), 0.0)


def _event_time(sp: Spikes, k: Array, offset: Array) -> Array:
    """``t[k]`` moved by ``offset`` samples: in value with ``sp.interpolate``,
    otherwise only in the gradient (the value stays Allen's ``t[k]``)."""
    dt = take(sp.t, k + 1) - take(sp.t, k)
    dt = jnp.where(jnp.isfinite(dt), dt, 0.0)
    if not sp.interpolate:
        offset = offset - lax.stop_gradient(offset)
    return take(sp.t, k) + dt * offset


def _spike_times(sp: Spikes, event: str) -> Array:
    """``[K]`` times (s) of one event of every spike: ``"threshold"``,
    ``"peak"``, ``"trough"``, ``"upstroke"``, ``"downstroke"``, ``"fast_trough"``,
    ``"adp"`` or ``"slow_trough"``; NaN where missing.

    The values are Allen's sample times, or with ``sp.interpolate`` the times
    between samples: where dV/dt crosses the threshold level (``thresh_frac``
    times the mean upstroke) or the fast-trough level (1% of the downstroke), and
    the vertex of a parabola through the extremum for the others (of dV/dt for
    up- and downstrokes, of ``v`` otherwise; for the afterdepolarization this
    approximates its definition on the 1 kHz dV/dt). Either way the gradients are
    those of the times between samples.
    """
    k = getattr(sp, event)
    if event == "threshold":
        level = _upstroke_target(sp.dvdt, sp.upstroke, sp.thresh_frac)
        offset = _crossing(sp.dvdt, k, level)
    elif event == "fast_trough":
        level = 0.01 * take(sp.dvdt, sp.downstroke)
        offset = _crossing(sp.dvdt, k - 1, level) - 1.0
    elif event in ("upstroke", "downstroke"):
        offset = _vertex(sp.dvdt, k)
    elif event in ("peak", "trough", "adp", "slow_trough"):
        offset = _vertex(sp.v, k)
    else:
        raise ValueError(f"unknown spike event {event!r}")
    return _event_time(sp, k, offset)


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
    # Times of the half-height crossings, between samples (see _spike_times).
    t_left = _event_time(sp, left, _crossing(sp.v, left, level))
    t_right = _event_time(sp, right, _crossing(sp.v, right - 1, level) - 1.0)
    return jnp.where(ok, t_right - t_left, jnp.nan)


def _point_features(sp: Spikes, name: str, idx: Array) -> dict[str, Array]:
    out = {
        f"{name}_index": idx,
        f"{name}_t": _spike_times(sp, name),
        f"{name}_v": take(sp.v, idx),
    }
    if sp.i is not None:
        out[f"{name}_i"] = take(sp.i, idx)
    return out


def spike_features(sp: Spikes, *, nan_to: float | None = None) -> dict[str, Array]:
    """The columns of Allen's spike table, as ``[K]`` arrays.

    ``*_index`` are sample indices (-1 where undefined), ``*_t`` times (s),
    ``*_v`` voltages (mV), ``*_i`` currents (pA, only if ``sp.i`` is given),
    ``upstroke``/``downstroke`` dV/dt (V/s), ``width`` the width at half height
    (s); undefined values are NaN. ``clipped`` is boolean, ``isi_type`` uses
    ``ISI_DIRECT``, ``ISI_DETOUR`` and ``ISI_MISSING``.

    ``nan_to`` replaces NaN values by the given value, e.g. ``0.0`` in a loss for
    gradient-based fitting, where a NaN would make all gradients NaN; ``None``
    (the default) keeps NaN.
    """
    return _spike_features(_analysed(sp), nan_to)


@partial(jax.jit, static_argnames=("nan_to",))
def _spike_features(sp: Spikes, nan_to: float | None) -> dict[str, Array]:
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
        out[f"{name}_t"] = _spike_times(sp, name)
        out[f"{name}_v"] = take(sp.v, idx)
    out["isi_type"] = sp.isi_type
    for name in ("fast_trough", "adp", "slow_trough"):
        out |= _point_features(sp, name, getattr(sp, name))
    out["width"] = _widths(sp)
    # NaN unless both are defined; dividing only then keeps gradients finite.
    up, down = out["upstroke"], out["downstroke"]
    both = jnp.isfinite(up) & jnp.isfinite(down)
    ratio = jnp.where(both, up, 1.0) / -jnp.where(both, down, -1.0)
    out["upstroke_downstroke_ratio"] = jnp.where(both, ratio, jnp.nan)
    return fill_nan(out, nan_to)
