"""Sweep-level features (Allen's ``EphysSweepFeatureExtractor.sweep_feature``).

Every feature takes the ``Spikes`` of one sweep (see ``detect_spikes``) and returns
a scalar, NaN where it is undefined. Units: s for ``latency``, ms for the
interspike intervals, ``tau`` and ``delay_tau``, mV for voltages, pA for
``stim_amp``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax

from ._ops import fill_nan, masked_mean, root_tangent_solve, take
from .patterns import burst_metrics, delay_metrics, pause_metrics
from .spikes import Spikes, _analysed, _spike_times, find_time_index

BASELINE_INTERVAL = 0.1  # s
SAG_PEAK_WIDTH = 0.005  # s
TAU_MAX_RMSE = 1.0  # mV


def _isis(sp: Spikes) -> tuple[Array, Array]:
    """Interspike intervals (s) between consecutive thresholds and their mask (0
    where undefined, which keeps the gradients of masked statistics finite)."""
    thr_t = _spike_times(sp, "threshold")
    valid = sp.valid[1:]
    return jnp.where(valid, thr_t[1:] - thr_t[:-1], 0.0), valid


def _soft_spike_count(sp: Spikes) -> Array:
    """Upward crossings of ``sp.min_peak`` in ``[start, end]``, counted smoothly:
    the rises of a sigmoid of the voltage of width ``sp.count_width``."""
    x = jnp.arange(sp.t.shape[0])
    inside = (x >= find_time_index(sp.t, sp.start)) & (
        x <= find_time_index(sp.t, sp.end)
    )
    below = sp.min_peak - 10.0 * sp.count_width  # outside: far below the level
    level = jax.nn.sigmoid(
        (jnp.where(inside, sp.v, below) - sp.min_peak) / sp.count_width
    )
    return jnp.sum(jax.nn.relu(level[1:] - level[:-1]))


@jax.custom_jvp
def _with_soft_count_derivative(count: Array, sp: Spikes) -> Array:
    """``count`` with the derivative of ``_soft_spike_count(sp)``; the soft count
    is only evaluated when differentiating."""
    return count


@_with_soft_count_derivative.defjvp
def _soft_count_jvp(primals, tangents):
    count, sp = primals
    _, sp_dot = tangents
    return count, jax.jvp(_soft_spike_count, (sp,), (sp_dot,))[1]


@jax.jit
def avg_rate(sp: Spikes) -> Array:
    """Firing rate (Hz): spikes with threshold in ``[start, end]`` per second.

    The count is exact; its gradient is that of a soft count of the upward
    crossings of ``min_peak`` (see ``detect_spikes``' ``count_width``).
    """
    thr_t = take(sp.t, sp.threshold)
    n = jnp.sum(sp.valid & (thr_t >= sp.start) & (thr_t <= sp.end))
    count = _with_soft_count_derivative(n.astype(sp.t.dtype), sp)
    return count / (sp.end - sp.start)


@jax.jit
def latency(sp: Spikes) -> Array:
    """Time from ``start`` to the first threshold (s)."""
    return _spike_times(sp, "threshold")[0] - sp.start


@jax.jit
def adapt(sp: Spikes) -> Array:
    """Adaptation index: mean of ``(isi[k+1] - isi[k]) / (isi[k+1] + isi[k])``."""
    isi, valid = _isis(sp)
    pair = valid[1:]
    total = jnp.where(pair, isi[1:] + isi[:-1], 1.0)
    return masked_mean((isi[1:] - isi[:-1]) / total, pair)


@jax.jit
def isi_cv(sp: Spikes) -> Array:
    """Coefficient of variation of the interspike intervals."""
    isi, valid = _isis(sp)
    some = jnp.any(valid)
    mean = jnp.where(some, masked_mean(isi, valid), 1.0)
    var = jnp.where(some, masked_mean((isi - mean) ** 2, valid), 0.0)
    # sqrt has an infinite derivative at 0 (all intervals equal).
    std = jnp.where(var > 0, jnp.sqrt(jnp.where(var > 0, var, 1.0)), 0.0)
    return jnp.where(some, std / mean, jnp.nan)


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
    ``a`` and ``y0`` are solved exactly for each ``k`` (variable projection) and
    ``log(k)`` takes Newton steps on the reduced problem within a trust region.
    The curvature is the secant of the exact gradient between the last two
    accepted points, else (or if not positive) the Gauss-Newton one, which alone
    converges only linearly when the exponential fits the response poorly. The
    search is not differentiated: gradients follow from the optimality condition
    (implicit differentiation, ``lax.custom_root``).
    """
    w = mask.astype(x.dtype)
    x = jnp.where(mask, x, 0.0)
    # Fit around the mean voltage: in float32, the sums of squares of voltages
    # around -70 mV would otherwise lose most of their precision.
    y_ref = jnp.sum(jnp.where(mask, y, 0.0)) / jnp.maximum(jnp.sum(w), 1.0)
    y = jnp.where(mask, y - y_ref, 0.0)
    n, sy, syy = jnp.sum(w), jnp.sum(w * y), jnp.sum(w * y * y)
    span = jnp.max(x)  # length of the fit window
    first = jnp.min(jnp.where(mask & (x > 0), x, jnp.inf))  # second sample

    def solve(log_k):
        """Linear parameters, SSE, and the gradient and Gauss-Newton curvature of
        the (halved) SSE in log(k), with a and y0 projected out (Kaufman)."""
        k = jnp.exp(log_k)
        e = w * jnp.exp(-k * x)
        see, se, sey = jnp.sum(e * e), jnp.sum(e), jnp.sum(e * y)
        sxee, sxe = jnp.sum(x * e * e), jnp.sum(x * e)
        sxxee, sxey = jnp.sum(x * x * e * e), jnp.sum(x * e * y)
        det = see * n - se * se
        det = jnp.where(det != 0, det, 1.0)  # empty window: keeps gradients finite
        a, y0 = (sey * n - sy * se) / det, (see * sy - se * sey) / det
        sse = syy - a * sey - y0 * sy
        jr = -a * (a * sxee + y0 * sxe - sxey)
        je, j1 = -a * sxee, -a * sxe
        jj = a * a * sxxee - (n * je * je - 2 * se * je * j1 + see * j1 * j1) / det
        return a, y0, sse, jr * k, jj * k * k

    def step(state):
        it, log_k, prev_log_k, prev_sse, prev_grad, max_step, _ = state
        _, _, sse, grad, curvature = solve(log_k)
        secant = (grad - prev_grad) / (log_k - prev_log_k)
        curvature = jnp.where(jnp.isfinite(secant) & (secant > 0), secant, curvature)
        d = -grad / curvature
        # Reject the last step and shrink the trust region if the SSE grew by more
        # than rounding (near the optimum, smaller changes are noise; comparing
        # them would make the search wander until max_iters); grow it otherwise.
        worse = sse > prev_sse + 1e-10 * jnp.abs(prev_sse)
        max_step = jnp.where(worse, max_step / 4, jnp.minimum(2 * max_step, 1.0))
        # Converged, or the trust region collapsed (steps would change k by less
        # than 1e-6; happens only on ill-posed fits, where steps keep failing).
        converged = (~worse & (jnp.abs(d) < tol)) | (max_step < 1e-6)
        next_log_k = jnp.where(
            worse,
            (log_k + prev_log_k) / 2,
            log_k + jnp.clip(d, -max_step, max_step),
        )
        prev_log_k = jnp.where(worse, prev_log_k, log_k)
        prev_sse = jnp.where(worse, prev_sse, sse)
        prev_grad = jnp.where(worse, prev_grad, grad)
        return it + 1, next_log_k, prev_log_k, prev_sse, prev_grad, max_step, converged

    def keep_going(state):
        it, log_k, *_, converged = state
        # Ill-posed fits run off towards a straight line (k -> 0) or a step
        # (k -> inf) without converging: stop once indistinguishable from those
        # (under vmap, one such sweep would otherwise hold up the whole batch).
        k = jnp.exp(log_k)
        degenerate = (k * span < 1e-3) | (k * first > 50.0)
        return (it < max_iters) & ~converged & ~degenerate

    zero, one = jnp.zeros((), x.dtype), jnp.ones((), x.dtype)

    def search(_, log_k):
        init = (0, log_k, log_k, jnp.inf, zero, one, ~jnp.any(mask))
        return lax.while_loop(keep_going, step, init)[1]

    def optimality(log_k):  # zero gradient of the SSE in log(k)
        return solve(log_k)[3]

    log_k0 = jnp.log(jnp.asarray(k0, dtype=x.dtype))
    log_k = lax.custom_root(optimality, log_k0, search, root_tangent_solve)
    a, y0, *_ = solve(log_k)
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
    # Skip the fit for sweeps that are not hyperpolarizing (under vmap, both run).
    return lax.cond(_is_hyperpolarizing(sp), _tau, _nan, sp)


def _nan(sp: Spikes) -> Array:
    return jnp.asarray(jnp.nan, dtype=sp.v.dtype)


def _tau(sp: Spikes) -> Array:
    v_peak, peak_idx = voltage_deflection(sp, "min")
    base = v_baseline(sp)
    x = jnp.arange(sp.t.shape[0])
    start_idx = find_time_index(sp.t, sp.start)
    crossed = (x >= start_idx) & (sp.v <= 0.1 * (v_peak - base) + base)
    fit_start = jnp.argmax(crossed)
    valid = (
        # Also checked here: under vmap, lax.cond runs this for every sweep, and an
        # empty fit window makes the fit return at once.
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
    return jnp.where(ok, 1e3 / jnp.where(ok, inv_tau, 1.0), jnp.nan)


@jax.jit
def sag(sp: Spikes) -> Array:
    """Fraction by which a hyperpolarizing response relaxes from its minimum
    (averaged over 5 ms) back towards the steady state (last 100 ms)."""
    return lax.cond(_is_hyperpolarizing(sp), _sag, _nan, sp)


def _sag(sp: Spikes) -> Array:
    _, peak_idx = voltage_deflection(sp, "min")
    peak_t = take(sp.t, peak_idx)
    v_peak_avg = _average_voltage(
        sp, peak_t - SAG_PEAK_WIDTH / 2.0, peak_t + SAG_PEAK_WIDTH / 2.0
    )
    v_steady = _average_voltage(sp, sp.end - BASELINE_INTERVAL, sp.end)
    out = (v_peak_avg - v_steady) / (v_peak_avg - v_baseline(sp))
    return jnp.where(jnp.isfinite(out), out, jnp.nan)


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
    after = (x > find_time_index(sp.t, sp.start)) & ~jnp.isnan(sp.i)
    return jnp.max(jnp.where(after, sp.i, -jnp.inf))


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


def sweep_features(
    sp: Spikes,
    stimulus: str | None = None,
    *,
    bursts: bool = False,
    pauses: bool = False,
    delays: bool = False,
    nan_to: float | None = None,
) -> dict[str, Array]:
    """All sweep-level features as scalars (NaN where undefined).

    ``stimulus`` (``"long_square"``, ``"short_square"``, ``"ramp"`` or ``None``)
    selects how ``stim_amp`` is measured, as in Allen's cell extractor; it is NaN
    for ramps, ``None`` and sweeps without current.

    Allen's firing-pattern metrics are added on request, with Allen's defaults:
    ``bursts`` adds ``max_burstiness`` and ``n_bursts`` (``burst_metrics``),
    ``pauses`` adds ``n_pauses`` and ``pause_frac`` (``pause_metrics``) and
    ``delays`` adds ``delay_ratio`` and ``delay_tau`` (ms; ``delay_metrics``).
    The delay fits exponentials to the rise before the first spike: it costs a
    few ms per sweep, and it differs from Allen where Allen's fit misses a
    double exponential.

    ``nan_to`` replaces NaN values by the given value, e.g. ``0.0`` in a loss for
    gradient-based fitting, where a NaN would make all gradients NaN; ``None``
    (the default) keeps NaN.
    """
    return _sweep_features(_analysed(sp), stimulus, bursts, pauses, delays, nan_to)


@partial(jax.jit, static_argnames=("stimulus", "bursts", "pauses", "delays", "nan_to"))
def _sweep_features(
    sp: Spikes,
    stimulus: str | None,
    bursts: bool,
    pauses: bool,
    delays: bool,
    nan_to: float | None,
) -> dict[str, Array]:
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
    if bursts:
        out["max_burstiness"], out["n_bursts"] = burst_metrics(sp)
    if pauses:
        out["n_pauses"], out["pause_frac"] = pause_metrics(sp)
    if delays:
        out["delay_ratio"], out["delay_tau"] = delay_metrics(sp)
    return fill_nan(out, nan_to)
