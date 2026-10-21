"""Sweepset features: the sweeps of one stimulus type combined, as in Allen's
``EphysCellFeatureExtractor`` and ``extract_cell_features``.

The inputs are dicts of ``[S]`` arrays with one entry per sweep: ``sweeps`` holds
the output of ``sweep_features`` and ``first_spikes`` the first spike of
``spike_features`` (``{name: values[0]}``), stacked over the sweeps, e.g. with
``jax.tree.map(lambda *x: jnp.stack(x), *per_sweep)``. Nothing here depends on
the length of the traces. ``valid`` marks the real sweeps when sweep sets are
padded to a common size, e.g. to ``jax.vmap`` over cells.

Selected sweeps are returned as ``[S]`` masks or as indices into the sweep axis
(-1 if there is none). Where Allen raises (no spiking or subthreshold sweeps, no
hero sweep, too few sweeps for a fit), the features are NaN.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ._ops import fill_nan, masked_mean, take
from .dvdt import calculate_dvdt
from .spikes import AUTO_MAX_SPIKES, _putative_upstrokes

SAG_TARGET = -100.0  # mV
SUBTHRESHOLD_MIN_AMP = -100.0  # pA, Allen's default ``subthresh_min_amp``
SUBTHRESHOLD_MAX_AMP = 0.0  # pA
HERO_AMP_OFFSETS = (39.0, 61.0)  # pA above rheobase
SHORT_SQUARE_WINDOW = (1.02, 1.021)  # s, passive response to the pulse
SHORT_SQUARE_MIN_THRESH_FRAC = 0.1
# First-spike features averaged over short squares and ramps (Allen's MEAN_FEATURES).
MEAN_SPIKE_0_FEATURES = (
    "upstroke_downstroke_ratio",
    "peak_v",
    "peak_t",
    "trough_v",
    "trough_t",
    "fast_trough_v",
    "fast_trough_t",
    "slow_trough_v",
    "slow_trough_t",
    "threshold_v",
    "threshold_i",
    "threshold_t",
)

Features = dict[str, Array]


def _valid(sweeps: Features, valid: Array | None) -> Array:
    if valid is None:
        return jnp.ones(jnp.shape(sweeps["avg_rate"]), dtype=bool)
    return jnp.asarray(valid, dtype=bool)


def _line_slope(x: Array, y: Array, mask: Array) -> Array:
    """Least-squares slope of ``y`` over ``x`` where ``mask`` (Allen fits a line
    with ``np.linalg.lstsq``); NaN without spread in ``x``."""
    x_mean, y_mean = masked_mean(x, mask), masked_mean(y, mask)
    dx = jnp.where(mask, x - x_mean, 0.0)
    spread = jnp.sum(dx * dx)
    ok = spread > 0  # dividing only then keeps gradients finite
    slope = jnp.sum(dx * jnp.where(mask, y - y_mean, 0.0)) / jnp.where(ok, spread, 1.0)
    return jnp.where(ok, slope, jnp.nan)


# Rate scale of the soft spike presence behind the rheobase's gradient (1/s): one
# spike in a 1 s long square.
_PRESENCE_SCALE = 1.0


def _soft_rheobase(amp: Array, rate: Array, valid: Array) -> Array:
    """Amplitude of the first spiking sweep in order of amplitude, from spike
    presences ``1 - exp(-rate * 1 s)`` whose values are exactly 0 or 1 and whose
    gradients are those of the soft spike counts in ``rate``. The gradient path of
    the rheobase: e.g. a lower sweep that starts spiking lowers it."""
    ok = valid & jnp.isfinite(amp) & jnp.isfinite(rate)
    smooth = 1.0 - jnp.exp(-jnp.where(ok, rate, 0.0) * _PRESENCE_SCALE)
    present = (
        jnp.where(ok & (rate > 0), 1.0, 0.0) + smooth - jax.lax.stop_gradient(smooth)
    )
    order = jnp.argsort(jnp.where(ok, amp, jnp.inf), stable=True)
    p = present[order]
    none_before = jnp.concatenate([jnp.ones(1, p.dtype), jnp.cumprod(1.0 - p)[:-1]])
    return jnp.sum(jnp.where(ok, amp, 0.0)[order] * p * none_before)


def _pick(values: Features, index: Array) -> Features:
    """Entry ``index`` of every ``[S]`` array, NaN for ``index == -1``."""
    return {
        name: take(jnp.asarray(x, dtype=float), index) for name, x in values.items()
    }


def _first_index(mask: Array) -> Array:
    return jnp.where(jnp.any(mask), jnp.argmax(mask), -1)


def mean_first_spike(first_spikes: Features, mask: Array) -> Features:
    """Mean first-spike features over the sweeps in ``mask`` (Allen's
    ``mean_features_spike_zero``; NaN if any of them is NaN or ``mask`` is empty)."""
    return {
        name: masked_mean(first_spikes[name], mask)
        for name in MEAN_SPIKE_0_FEATURES
        if name in first_spikes
    }


def long_square_features(
    sweeps: Features,
    first_spikes: Features,
    valid: Array | None = None,
    subthresh_min_amp: float = SUBTHRESHOLD_MIN_AMP,
    *,
    nan_to: float | None = None,
) -> Features:
    """Long-square features (Allen's ``_analyze_long_squares`` plus the hero sweep
    and baseline of ``extract_cell_features``).

    Requires ``sweep_features(sp, stimulus="long_square")`` for every sweep, so
    that ``stim_amp`` is the step amplitude. Returns:

    - ``rheobase_index``, ``rheobase_i`` (pA): the spiking sweep with the lowest
      amplitude (ties: highest rate, the vendored Allen rule), and
      ``rheobase_spike``, the first-spike features of that sweep;
    - ``fi_fit_slope`` (Hz/pA): slope of ``avg_rate`` over ``stim_amp`` of the
      spiking sweeps;
    - ``sag`` and ``vm_for_sag`` (mV) of the subthreshold sweep whose
      ``peak_deflect`` is closest to -100 mV;
    - ``input_resistance`` (MOhm): slope of the minimum voltage over the current
      at that time for the subthreshold sweeps with ``subthresh_min_amp <
      stim_amp < 0`` (``membrane_property``); with only one, its baseline at 0 pA
      is the second point;
    - ``tau`` (ms): mean of the finite ``tau`` of those sweeps;
    - ``v_baseline`` (mV): mean ``v_baseline`` of all sweeps;
    - ``hero_index``, ``hero_sweep``: the spiking sweep with the lowest amplitude
      in (rheobase + 39, rheobase + 61) pA and its sweep features;
    - ``[S]`` masks ``spiking``, ``subthreshold`` and ``membrane_property``.

    ``nan_to`` replaces NaN values in the result by the given value (e.g. ``0.0``
    in a loss for gradient-based fitting); ``None`` (the default) keeps NaN.
    """
    valid = _valid(sweeps, valid)
    amp, rate = sweeps["stim_amp"], sweeps["avg_rate"]

    # Allen's rheobase candidates are the sweeps with a nonzero rate.
    nonzero = valid & (rate != 0)
    min_amp = jnp.min(jnp.where(nonzero & ~jnp.isnan(amp), amp, jnp.inf))
    candidate = nonzero & jnp.isclose(amp, min_amp)
    max_rate = jnp.max(jnp.where(candidate & ~jnp.isnan(rate), rate, -jnp.inf))
    fastest = candidate & jnp.isclose(rate, max_rate)
    rheobase = _first_index(jnp.where(jnp.sum(candidate) == 1, candidate, fastest))
    # Allen's rheobase, with the gradient of a soft one (see _soft_rheobase).
    soft = _soft_rheobase(amp, rate, valid)
    rheobase_i = take(amp, rheobase) + soft - jax.lax.stop_gradient(soft)

    spiking = valid & (rate > 0)
    fi_fit_slope = jnp.where(
        jnp.sum(spiking) >= 2, _line_slope(amp, rate, spiking), jnp.nan
    )

    subthreshold = valid & (rate == 0)
    distance = jnp.abs(sweeps["peak_deflect"] - SAG_TARGET)
    closest = jnp.where(
        jnp.any(subthreshold),
        jnp.argmin(jnp.where(subthreshold, distance, jnp.inf)),
        -1,
    )

    membrane = subthreshold & (amp > subthresh_min_amp) & (amp < SUBTHRESHOLD_MAX_AMP)
    single = jnp.sum(membrane) == 1
    v_fit = jnp.append(
        sweeps["min_deflect"], sweeps["v_baseline"][jnp.argmax(membrane)]
    )
    i_fit = jnp.append(sweeps["min_deflect_i"], 0.0)
    resistance = 1e3 * _line_slope(i_fit, v_fit, jnp.append(membrane, single))
    tau = sweeps["tau"]

    lo, hi = rheobase_i + HERO_AMP_OFFSETS[0], rheobase_i + HERO_AMP_OFFSETS[1]
    hero_ok = spiking & (amp > lo) & (amp < hi)
    hero = jnp.where(jnp.any(hero_ok), jnp.argmin(jnp.where(hero_ok, amp, jnp.inf)), -1)
    out = {
        "rheobase_index": rheobase,
        "rheobase_i": rheobase_i,
        "rheobase_spike": _pick(first_spikes, rheobase),
        "fi_fit_slope": fi_fit_slope,
        "sag": take(sweeps["sag"], closest),
        "vm_for_sag": take(sweeps["peak_deflect"], closest),
        "input_resistance": jnp.where(jnp.any(membrane), resistance, jnp.nan),
        "tau": masked_mean(tau, membrane & jnp.isfinite(tau)),
        "v_baseline": masked_mean(sweeps["v_baseline"], valid),
        "hero_index": hero,
        "hero_sweep": _pick(sweeps, hero),
        "spiking": spiking,
        "subthreshold": subthreshold,
        "membrane_property": membrane,
    }
    return fill_nan(out, nan_to)


def short_square_features(
    sweeps: Features,
    first_spikes: Features,
    valid: Array | None = None,
    *,
    nan_to: float | None = None,
) -> Features:
    """Short-square features (Allen's ``_analyze_short_squares``).

    Requires ``sweep_features(sp, stimulus="short_square")`` for every sweep.
    Returns the most common amplitude of the spiking sweeps
    (``stimulus_amplitude``, pA; ties: the lowest), the ``[S]`` mask of the
    spiking sweeps at that amplitude (``common_amp``) and their mean first-spike
    features (``mean_spike_0``).

    ``nan_to`` replaces NaN values in the result by the given value (e.g. ``0.0``
    in a loss for gradient-based fitting); ``None`` (the default) keeps NaN.
    """
    valid = _valid(sweeps, valid)
    amp = sweeps["stim_amp"]
    spiking = valid & (sweeps["avg_rate"] > 0)
    count = jnp.sum((amp[:, None] == amp[None, :]) & spiking[None, :], axis=1)
    count = jnp.where(spiking, count, 0)
    common = jnp.min(jnp.where(spiking & (count == jnp.max(count)), amp, jnp.inf))
    common_amp = spiking & (amp == common)
    out = {
        "stimulus_amplitude": jnp.where(jnp.any(spiking), common, jnp.nan),
        "common_amp": common_amp,
        "mean_spike_0": mean_first_spike(first_spikes, common_amp),
    }
    return fill_nan(out, nan_to)


def ramp_features(
    sweeps: Features,
    first_spikes: Features,
    valid: Array | None = None,
    *,
    nan_to: float | None = None,
) -> Features:
    """Ramp features: the ``[S]`` mask of spiking ramps (``spiking``) and their
    mean first-spike features (``mean_spike_0``).

    ``nan_to`` replaces NaN values in the result by the given value (e.g. ``0.0``
    in a loss for gradient-based fitting); ``None`` (the default) keeps NaN.
    """
    spiking = _valid(sweeps, valid) & (sweeps["avg_rate"] > 0)
    out = {"spiking": spiking, "mean_spike_0": mean_first_spike(first_spikes, spiking)}
    return fill_nan(out, nan_to)


@partial(jax.jit, static_argnames=("filter",))
def _passive_dvdt(
    t: Array, v: Array, lo: int, hi: int, *, filter: float
) -> tuple[Array, Array, Array]:
    """dV/dt of a short square, its maximum in ``[lo, hi)`` and its value at ``hi``."""
    dvdt = calculate_dvdt(v, t, filter)
    x = jnp.arange(dvdt.shape[0])
    return dvdt, jnp.max(jnp.where((x >= lo) & (x < hi), dvdt, -jnp.inf)), dvdt[hi]


@partial(jax.jit, static_argnames=("max_spikes",))
def _upstroke_dvdt(
    v: Array, dvdt: Array, dv_cutoff: Array, *, max_spikes: int
) -> tuple[Array, Array, Array]:
    """dV/dt at the upstrokes of the putative spikes in the whole sweep."""
    v = jnp.asarray(v, dtype=float)
    end = v.shape[0] - 1
    _, _, ups, overflow = _putative_upstrokes(
        v, dvdt, dvdt, 0, end, dv_cutoff, 2.0, -30.0, max_spikes
    )
    return take(dvdt, ups), ups >= 0, overflow


def short_square_detection_params(
    t_set: list[Array],
    v_set: list[Array],
    window: tuple[float, float] = SHORT_SQUARE_WINDOW,
    filter: float = 10.0,
) -> tuple[Array, Array]:
    """``dv_cutoff`` (V/s) and ``thresh_frac`` for detecting short-square spikes.

    Allen's ``estimate_adjusted_detection_parameters`` as used by
    ``cell_extractor_for_nwb``: the strong, brief current makes the passive
    response steep, so the cutoff is 1.1x the median (over sweeps) of the maximum
    dV/dt in ``window`` (s, sample indices taken from the first sweep). The
    threshold fraction is 1.2x the median dV/dt at the end of ``window`` over the
    mean upstroke of all putative spikes found with that cutoff, at least 0.1.

    Takes lists of ``[T]`` arrays (lengths may differ); runs eagerly.
    """
    if len(t_set) != len(v_set) or not t_set:
        raise ValueError("t_set and v_set must be non-empty lists of equal length")
    t0 = np.asarray(t_set[0], dtype=float)
    lo, hi = (min(int(np.searchsorted(t0, x)), t0.size - 1) for x in window)
    passive = [_passive_dvdt(t, v, lo, hi, filter=filter) for t, v in zip(t_set, v_set)]
    dv_cutoff = 1.1 * jnp.median(jnp.stack([p[1] for p in passive]))
    min_thresh = 1.2 * jnp.median(jnp.stack([p[2] for p in passive]))

    upstrokes, found = [], []
    for v, (dvdt, _, _) in zip(v_set, passive):
        capacity = AUTO_MAX_SPIKES
        while True:
            values, ok, overflow = _upstroke_dvdt(
                v, dvdt, dv_cutoff, max_spikes=capacity
            )
            if not overflow:
                break
            capacity *= 4
        upstrokes.append(values)
        found.append(ok)
    upstroke = masked_mean(jnp.concatenate(upstrokes), jnp.concatenate(found))
    return dv_cutoff, jnp.maximum(min_thresh / upstroke, SHORT_SQUARE_MIN_THRESH_FRAC)
