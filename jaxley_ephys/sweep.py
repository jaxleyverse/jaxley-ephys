"""Sweep-level features (Allen's ``EphysSweepFeatureExtractor.sweep_feature``).

Every feature takes the ``Spikes`` of one sweep (see ``detect_spikes``) and returns
a scalar, NaN where it is undefined. Units: s for ``latency``, ms for the
interspike intervals and ``tau``, mV for voltages, pA for ``stim_amp``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax

from ._ops import masked_mean, take
from .spikes import Spikes, find_time_index

BASELINE_INTERVAL = 0.1  # s
SAG_PEAK_WIDTH = 0.005  # s
TAU_MAX_RMSE = 1.0  # mV


def _isis(sp: Spikes) -> tuple[Array, Array]:
    """Interspike intervals (s) between consecutive thresholds and their mask."""
    thr_t = take(sp.t, sp.threshold)
    return thr_t[1:] - thr_t[:-1], sp.valid[1:]


@jax.jit
def avg_rate(sp: Spikes) -> Array:
    """Firing rate (Hz): spikes with threshold in ``[start, end]`` per second."""
    thr_t = take(sp.t, sp.threshold)
    n = jnp.sum(sp.valid & (thr_t >= sp.start) & (thr_t <= sp.end))
    return n / (sp.end - sp.start)


@jax.jit
def latency(sp: Spikes) -> Array:
    """Time from ``start`` to the first threshold (s)."""
    return take(sp.t, sp.threshold[0]) - sp.start


@jax.jit
def adapt(sp: Spikes) -> Array:
    """Adaptation index: mean of ``(isi[k+1] - isi[k]) / (isi[k+1] + isi[k])``."""
    isi, valid = _isis(sp)
    norm = (isi[1:] - isi[:-1]) / (isi[1:] + isi[:-1])
    return masked_mean(norm, valid[1:])


@jax.jit
def isi_cv(sp: Spikes) -> Array:
    """Coefficient of variation of the interspike intervals."""
    isi, valid = _isis(sp)
    mean = masked_mean(isi, valid)
    return jnp.sqrt(masked_mean((isi - mean) ** 2, valid)) / mean


@jax.jit
def mean_isi(sp: Spikes) -> Array:
    """Mean interspike interval (ms)."""
    isi, valid = _isis(sp)
    return 1e3 * masked_mean(isi, valid)


@jax.jit
def median_isi(sp: Spikes) -> Array:
    """Median interspike interval (ms)."""
    isi, valid = _isis(sp)
    n = jnp.sum(valid)
    isi = jnp.sort(jnp.where(valid, isi, jnp.inf))
    lo, hi = jnp.maximum((n - 1) // 2, 0), jnp.maximum(n // 2, 0)
    return jnp.where(n > 0, 1e3 * (isi[lo] + isi[hi]) / 2.0, jnp.nan)


@jax.jit
def first_isi(sp: Spikes) -> Array:
    """First interspike interval (ms)."""
    isi, valid = _isis(sp)
    return jnp.where(valid[0], 1e3 * isi[0], jnp.nan)


def _average_voltage(sp: Spikes, start: Array, end: Array) -> Array:
    """Mean voltage from ``start`` (inclusive) to ``end`` (exclusive)."""
    x = jnp.arange(sp.t.shape[0])
    inside = (x >= find_time_index(sp.t, start)) & (x < find_time_index(sp.t, end))
    return masked_mean(sp.v, inside)


@jax.jit
def v_baseline(sp: Spikes) -> Array:
    """Mean voltage in the 100 ms before ``start`` (mV).

    NaN without a baseline before ``start`` (e.g. ``start=None``); Allen then
    searches for a flat interval instead (see ``allen_ephys/ALLEN_EPHYS_FIXES.md``).
    """
    return _average_voltage(sp, sp.start - BASELINE_INTERVAL, sp.start)


def _halfway_current(sp: Spikes) -> Array:
    return sp.i[find_time_index(sp.t, (sp.end - sp.start) / 2.0 + sp.start)]


@partial(jax.jit, static_argnames=("deflect_type",))
def voltage_deflection(
    sp: Spikes, deflect_type: str | None = None
) -> tuple[Array, Array]:
    """Minimum or maximum voltage in ``[start, end)`` (mV) and its index.

    ``deflect_type`` is ``"min"``, ``"max"`` or ``None``: then the maximum is used
    if the current halfway through the window is >= 0, else the minimum (the
    minimum if there is no current). NaN and -1 for an empty window.
    """
    if deflect_type not in (None, "min", "max"):
        raise ValueError(
            f"deflect_type must be None, 'min' or 'max', got {deflect_type!r}"
        )
    x = jnp.arange(sp.t.shape[0])
    start_idx, end_idx = find_time_index(sp.t, sp.start), find_time_index(sp.t, sp.end)
    inside = (x >= start_idx) & (x < end_idx)
    idx_min = jnp.argmin(jnp.where(inside, sp.v, jnp.inf))
    idx_max = jnp.argmax(jnp.where(inside, sp.v, -jnp.inf))
    if deflect_type == "min" or (deflect_type is None and sp.i is None):
        idx = idx_min
    elif deflect_type == "max":
        idx = idx_max
    else:
        idx = jnp.where(_halfway_current(sp) >= 0, idx_max, idx_min)
    idx = jnp.where(jnp.any(inside), idx, -1)
    return take(sp.v, idx), idx


@jax.jit
def peak_deflect(sp: Spikes) -> Array:
    """Voltage of the peak deflection (mV), see ``voltage_deflection``."""
    return voltage_deflection(sp)[0]


@jax.jit
def peak_deflect_index(sp: Spikes) -> Array:
    """Sample index of the peak deflection, see ``voltage_deflection``."""
    return voltage_deflection(sp)[1]


@jax.jit
def min_deflect(sp: Spikes) -> Array:
    """Minimum voltage in ``[start, end)`` (mV)."""
    return voltage_deflection(sp, "min")[0]


@jax.jit
def min_deflect_i(sp: Spikes) -> Array:
    """Injected current at the minimum voltage (pA); NaN without current."""
    if sp.i is None:
        return jnp.asarray(jnp.nan, dtype=sp.v.dtype)
    return take(sp.i, voltage_deflection(sp, "min")[1])


def _is_hyperpolarizing(sp: Spikes) -> Array:
    """Negative current halfway through ``[start, end]``."""
    if sp.i is None:
        return jnp.asarray(False)
    i_half = _halfway_current(sp)
    return (sp.end > sp.start) & jnp.isfinite(i_half) & (i_half < 0)


def _fit_exp(
    x: Array,
    y: Array,
    mask: Array,
    k0: float = 50.0,
    max_iters: int = 100,
    tol: float = 1e-10,
) -> Array:
    """Least-squares fit of ``y0 + a * exp(-k * x)`` over ``mask``; returns ``(a, k, y0)``.

    Like Allen's ``curve_fit`` call this is a local search started at ``k = 50``:
    ``a`` and ``y0`` are solved exactly for each ``k`` (variable projection) and ``k``
    is updated with damped Gauss-Newton steps on the reduced problem.
    """
    w = mask.astype(x.dtype)
    x = jnp.where(mask, x, 0.0)
    # Fit around the mean voltage: in float32, the sums of squares of voltages
    # around -70 mV would otherwise lose most of their precision.
    y_ref = jnp.sum(jnp.where(mask, y, 0.0)) / jnp.maximum(jnp.sum(w), 1.0)
    y = jnp.where(mask, y - y_ref, 0.0)
    n, sy, syy = jnp.sum(w), jnp.sum(w * y), jnp.sum(w * y * y)

    def solve(log_k):
        e = w * jnp.exp(-jnp.exp(log_k) * x)
        see, se, sey = jnp.sum(e * e), jnp.sum(e), jnp.sum(e * y)
        sxee, sxe = jnp.sum(x * e * e), jnp.sum(x * e)
        sxxee, sxey = jnp.sum(x * x * e * e), jnp.sum(x * e * y)
        det = see * n - se * se
        a, y0 = (sey * n - sy * se) / det, (see * sy - se * sey) / det
        sse = syy - a * sey - y0 * sy
        # Gauss-Newton step in log(k) with a, y0 projected out (Kaufman).
        jr = -a * (a * sxee + y0 * sxe - sxey)
        je, j1 = -a * sxee, -a * sxe
        jj = a * a * sxxee - (n * je * je - 2 * se * je * j1 + see * j1 * j1) / det
        return a, y0, sse, -jr / jj / jnp.exp(log_k)

    def step(state):
        it, log_k, prev_log_k, prev_sse, max_step, _ = state
        _, _, sse, d = solve(log_k)
        worse = sse > prev_sse  # reject the last step and shrink the trust region
        log_k = jnp.where(worse, (log_k + prev_log_k) / 2, log_k)
        max_step = jnp.where(worse, max_step / 4, max_step)
        d = jnp.where(worse, 0.0, jnp.clip(d, -max_step, max_step))
        prev_log_k = jnp.where(worse, prev_log_k, log_k)
        prev_sse = jnp.where(worse, prev_sse, sse)
        converged = ~worse & (jnp.abs(d) < tol)
        return it + 1, log_k + d, prev_log_k, prev_sse, max_step, converged

    def keep_going(state):
        it, *_, converged = state
        return (it < max_iters) & ~converged

    log_k = jnp.log(jnp.asarray(k0, dtype=x.dtype))
    init = (0, log_k, log_k, jnp.inf, jnp.asarray(1.0, x.dtype), ~jnp.any(mask))
    _, log_k, *_ = lax.while_loop(keep_going, step, init)
    a, y0, _, _ = solve(log_k)
    return jnp.stack([a, jnp.exp(log_k), y0 + y_ref])


@jax.jit
def tau(sp: Spikes) -> Array:
    """Membrane time constant (ms) of a hyperpolarizing response (Allen's
    ``estimate_time_constant``).

    An exponential is fitted from where the voltage has dropped 10% of the way
    from baseline to its minimum until the minimum; fits with an RMSE above 1 mV
    give NaN. Allen's ``curve_fit`` stops before the least-squares optimum, so its
    values differ by up to ~1e-3 (relative) from this fully converged fit (see
    ``allen_ephys/ALLEN_EPHYS_FIXES.md``, section 6).
    """
    v_peak, peak_idx = voltage_deflection(sp, "min")
    base = v_baseline(sp)
    x = jnp.arange(sp.t.shape[0])
    start_idx = find_time_index(sp.t, sp.start)
    crossed = (x >= start_idx) & (sp.v <= 0.1 * (v_peak - base) + base)
    fit_start = jnp.argmax(crossed)
    valid = (
        _is_hyperpolarizing(sp)
        & jnp.isfinite(v_peak)
        & jnp.isfinite(base)
        & (v_peak < base)
        & jnp.any(crossed)
        & (peak_idx - fit_start >= 4)
    )
    mask = (x >= fit_start) & (x < peak_idx)
    t_rel = sp.t - sp.t[fit_start]
    a, inv_tau, y0 = _fit_exp(t_rel, sp.v, mask & valid)
    rmse = jnp.sqrt(masked_mean((y0 + a * jnp.exp(-inv_tau * t_rel) - sp.v) ** 2, mask))
    ok = valid & (rmse <= TAU_MAX_RMSE) & jnp.isfinite(inv_tau) & (inv_tau > 0)
    return jnp.where(ok, 1e3 / inv_tau, jnp.nan)


@jax.jit
def sag(sp: Spikes) -> Array:
    """Fraction by which a hyperpolarizing response relaxes from its minimum
    (averaged over 5 ms) back towards the steady state (last 100 ms)."""
    _, peak_idx = voltage_deflection(sp, "min")
    peak_t = take(sp.t, peak_idx)
    v_peak_avg = _average_voltage(
        sp, peak_t - SAG_PEAK_WIDTH / 2.0, peak_t + SAG_PEAK_WIDTH / 2.0
    )
    v_steady = _average_voltage(sp, sp.end - BASELINE_INTERVAL, sp.end)
    out = (v_peak_avg - v_steady) / (v_peak_avg - v_baseline(sp))
    return jnp.where(_is_hyperpolarizing(sp) & jnp.isfinite(out), out, jnp.nan)


def _require_current(sp: Spikes, name: str) -> None:
    if sp.i is None:
        raise ValueError(f"{name} needs the injected current i")


@jax.jit
def step_amplitude(sp: Spikes) -> Array:
    """Long-square stimulus amplitude (pA): the current right after ``start``."""
    _require_current(sp, "step_amplitude")
    return sp.i[find_time_index(sp.t, sp.start) + 1]


@jax.jit
def short_step_amplitude(sp: Spikes) -> Array:
    """Short-square stimulus amplitude (pA): the maximum current after ``start``."""
    _require_current(sp, "short_step_amplitude")
    x = jnp.arange(sp.t.shape[0])
    return jnp.max(jnp.where(x > find_time_index(sp.t, sp.start), sp.i, -jnp.inf))


SPIKE_TRAIN_FEATURES = {
    "adapt": adapt,
    "latency": latency,
    "isi_cv": isi_cv,
    "mean_isi": mean_isi,
    "median_isi": median_isi,
    "first_isi": first_isi,
    "avg_rate": avg_rate,
}
SUBTHRESHOLD_FEATURES = {
    "v_baseline": v_baseline,
    "tau": tau,
    "sag": sag,
    "peak_deflect": peak_deflect,
    "peak_deflect_index": peak_deflect_index,
    # Not sweep features in Allen; its cell-level input resistance needs them.
    "min_deflect": min_deflect,
    "min_deflect_i": min_deflect_i,
}
# How Allen's cell extractor measures ``stim_amp`` for each stimulus type.
STIMULUS_AMPLITUDE = {
    "long_square": step_amplitude,
    "short_square": short_step_amplitude,
    "ramp": None,
}


@partial(jax.jit, static_argnames=("stimulus",))
def sweep_features(sp: Spikes, stimulus: str | None = None) -> dict[str, Array]:
    """All sweep-level features as scalars (NaN where undefined).

    ``stimulus`` (``"long_square"``, ``"short_square"``, ``"ramp"`` or ``None``)
    selects how ``stim_amp`` is measured, as in Allen's cell extractor; it is NaN
    for ramps, ``None`` and sweeps without current.
    """
    if stimulus is not None and stimulus not in STIMULUS_AMPLITUDE:
        raise ValueError(
            f"stimulus must be None or one of {list(STIMULUS_AMPLITUDE)}, got {stimulus!r}"
        )
    out = {
        name: fn(sp)
        for name, fn in (SPIKE_TRAIN_FEATURES | SUBTHRESHOLD_FEATURES).items()
    }
    amplitude = STIMULUS_AMPLITUDE.get(stimulus)
    if amplitude is None or sp.i is None:
        out["stim_amp"] = jnp.asarray(jnp.nan, dtype=sp.v.dtype)
    else:
        out["stim_amp"] = amplitude(sp)
    return out
