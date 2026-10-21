"""Firing-pattern metrics of a sweep: pauses, bursts and delayed firing.

These are Allen's ``EphysSweepFeatureExtractor.pause_metrics``, ``burst_metrics``
and ``delay_metrics``, computed on request only, as in Allen: each takes the
``Spikes`` of one sweep (see ``detect_spikes``) and returns a pair of scalars, and
``sweep_features(sp, bursts=True, pauses=True, delays=True)`` adds them to the
sweep features.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from ._ops import masked_mean, next_true, prev_true, shift_left, shift_right, take
from .spikes import ISI_DETOUR, ISI_DIRECT, ISI_MISSING, Spikes, find_time_index

# ISI types set during burst detection, besides ISI_DIRECT and ISI_DETOUR.
_PAUSELIKE, _MIDBURST = 2, 3
# Costs of a burst's boundaries and of its length (Allen's _score_burst_set).
_BURST_TRANSITION_COST, _BURST_LENGTH_COST = 0.01, 0.1
# Starts of the double-exponential fits: log-rates relative to the single
# exponential's, i.e. a component 100x faster and one at the same rate or 100x slower.
_DOUBLE_EXP_STARTS = ((np.log(100.0), 0.0), (np.log(100.0), -np.log(100.0)))
# The exponential fits sum over chunks of this many samples covering the fit
# window: their cost scales with the window, not with the sweep.
_CHUNK = 1024


def _spike_train(sp: Spikes) -> tuple[Array, Array, Array]:
    """``[K - 1]`` interspike intervals (s) between consecutive thresholds, their
    mask and the ISI type of the spike before each interval."""
    thr_t = take(sp.t, sp.threshold)
    valid = sp.valid[1:]
    isi = jnp.where(valid, thr_t[1:] - thr_t[:-1], jnp.nan)
    return isi, valid, jnp.where(valid, sp.isi_type[:-1], ISI_MISSING)


def _masked_median(values: Array, mask: Array) -> Array:
    n = jnp.sum(mask)
    values = jnp.sort(jnp.where(mask, values, jnp.inf))
    lo, hi = jnp.maximum((n - 1) // 2, 0), jnp.maximum(n // 2, 0)
    return jnp.where(n > 0, (values[lo] + values[hi]) / 2.0, jnp.nan)


def _mean_std(values: Array, mask: Array) -> tuple[Array, Array]:
    mean = masked_mean(values, mask)
    return mean, jnp.sqrt(masked_mean((values - mean) ** 2, mask))


def _detect_pauses(isi: Array, valid: Array, types: Array, cost_weight: Array) -> Array:
    """Allen's ``detect_pauses``: mask of the ISIs that are pauses.

    Candidates (detour ISIs, then direct ISIs longer than 3x the median direct
    one) are added in turn until their net benefit falls again: how much less the
    other ISIs vary (coefficient of variation) without them, minus ``cost_weight``
    times a cost that grows as they approach the other ISIs' mean. No pauses if
    the net benefit never becomes positive.
    """
    m = isi.shape[0]
    idx = jnp.arange(m)
    direct, detour = types == ISI_DIRECT, types == ISI_DETOUR
    long_direct = direct & (isi > 3.0 * _masked_median(isi, direct))
    order = jnp.argsort(jnp.where(detour, idx, jnp.where(long_direct, m, 2 * m) + idx))
    n_candidates = jnp.sum(detour | long_direct)
    mean, std = _mean_std(isi, valid)
    all_cv = std / mean

    def try_next(state):
        j, pauses, best_net, _ = state
        trial = pauses.at[order[j]].set(True)
        rest = valid & ~trial
        mean, std = _mean_std(isi, rest)
        cost = jnp.sum(jnp.where(trial, std / jnp.abs(mean - isi), 0.0))
        net = all_cv - std / mean - cost_weight * cost
        stop = (jnp.sum(rest) < 2) | ((net > 0) & (net < best_net))
        best_net = jnp.where(~stop & (net > best_net), net, best_net)
        return j + 1, jnp.where(stop, pauses, trial), best_net, stop

    def keep_going(state):
        j, *_, stop = state
        return (j < n_candidates) & ~stop

    # Pauses need direct-type firing around them.
    init = (0, jnp.zeros(m, bool), jnp.zeros((), isi.dtype), ~jnp.any(direct))
    _, pauses, best_net, _ = lax.while_loop(keep_going, try_next, init)
    return pauses & (best_net > 0)


@jax.jit
def pause_metrics(sp: Spikes, cost_weight: float = 1.0) -> tuple[Array, Array]:
    """Number of pauses and the fraction of ``[start, end]`` spent in them.

    Pauses are unusually long interspike intervals among regular firing (Allen's
    ``detect_pauses``); a higher ``cost_weight`` finds fewer. ``(0, 0.0)`` without
    pauses.
    """
    isi, valid, types = _spike_train(sp)
    pauses = _detect_pauses(isi, valid, types, cost_weight)
    pause_time = jnp.sum(jnp.where(pauses, isi, 0.0))
    return jnp.sum(pauses), pause_time / (sp.end - sp.start)


def _burstiness(min_in: Array, min_out: Array) -> Array:
    """Allen's ``_burstiness_index`` from the shortest ISI in and out of a burst."""
    rate_in, rate_out = 1.0 / min_in, 1.0 / min_out
    return (rate_in - rate_out) / (rate_in + rate_out)


def _detect_bursts(sp: Spikes, tol: Array, pause_cost: Array) -> tuple[Array, Array]:
    """Allen's ``detect_bursts``: ``[K - 1]`` mask of the ISIs that start a burst
    and the burstiness index of each burst."""
    isi, valid, types = _spike_train(sp)
    m = isi.shape[0]
    idx = jnp.arange(m)
    n_isi = jnp.sum(valid)
    fast_t = take(sp.t, sp.fast_trough)[:-1]
    slow_t, slow_v = take(sp.t, sp.slow_trough)[:-1], take(sp.v, sp.slow_trough)[:-1]
    thr_v = take(sp.v, sp.threshold)[:-1]

    # Burst boundaries cannot be at pauses, and bursts need direct and detour ISIs.
    types = jnp.where(_detect_pauses(isi, valid, types, pause_cost), _PAUSELIKE, types)
    possible = (
        (n_isi >= 2) & jnp.any(types == ISI_DIRECT) & jnp.any(types == ISI_DETOUR)
    )
    # A detour whose slow trough stays above the spike's threshold (within tol) is
    # part of a burst.
    midburst = (types == ISI_DETOUR) & (thr_v < slow_v + tol)
    types = jnp.where(midburst, _MIDBURST, types)
    detour = types == ISI_DETOUR
    # Candidates start at the first direct ISI after a detour (or at the first ISI)
    # and end at the next detour ISI; those running into a pause are dropped.
    start = possible & (types == ISI_DIRECT) & ((idx == 0) | shift_right(detour, False))
    next_start = jnp.minimum(shift_left(next_true(start), m), n_isi)
    stop = shift_left(next_true(detour | (types == _PAUSELIKE)), m)
    found = stop < next_start
    start = start & ~(found & (types[jnp.minimum(stop, m - 1)] == _PAUSELIKE))
    end = jnp.where(found, stop, n_isi)  # first ISI after each burst
    delta_frac = (slow_t - fast_t) / isi

    def in_burst(bursts):
        """For each ISI: the start of the last burst before it, and whether it is
        inside that burst."""
        owner = prev_true(bursts)
        return owner, (owner >= 0) & (idx < end[jnp.maximum(owner, 0)])

    def score(bursts):
        owner, inside = in_burst(bursts)
        rest = valid & ~inside
        min_in = jax.ops.segment_min(isi, jnp.where(inside, owner, m), m)
        s = _burstiness(min_in, jnp.min(jnp.where(rest, isi, jnp.inf)))
        cost_end = _BURST_TRANSITION_COST / delta_frac[jnp.minimum(end, m - 1)]
        cost_start = _BURST_TRANSITION_COST / shift_right(delta_frac, jnp.nan)
        s = s - jnp.where(end < n_isi, cost_end, 0.0)
        s = s - jnp.where(idx > 0, cost_start, 0.0)
        s = s - _BURST_LENGTH_COST * (end - idx - 1)
        # Bursts covering all ISIs get the worst score.
        return jnp.where(jnp.any(rest), s, -1e12)

    def mean_score(s, bursts):
        return jnp.sum(jnp.where(bursts, s, 0.0)) / jnp.sum(bursts)

    def without_worst(s, bursts):
        return bursts & (idx != jnp.argmin(jnp.where(bursts, s, jnp.inf)))

    # Drop the worst-scoring burst as long as that raises the mean score.
    s = score(start)
    init = (start, without_worst(s, start), mean_score(s, start), False)

    def prune(state):
        bursts, test, best, _ = state
        s = score(test)
        mean = mean_score(s, test)
        better = mean > best
        bursts = jnp.where(better, test, bursts)
        test = jnp.where(better, without_worst(s, test), test)
        return bursts, test, jnp.where(better, mean, best), ~better

    def keep_pruning(state):
        _, test, _, done = state
        return jnp.any(test) & ~done

    bursts, _, best, _ = lax.while_loop(keep_pruning, prune, init)
    bursts = bursts & ~(best < 0)

    # Each burst is compared with the ISIs up to the next burst; the last one with
    # the ISIs after it, or before it if it ends the spike train.
    owner, inside = in_burst(bursts)
    min_in = jax.ops.segment_min(isi, jnp.where(inside, owner, m), m)
    gap = jnp.cumsum(bursts)  # outside bursts: 0 before the first one, 1 after it, ...
    gap_min = jax.ops.segment_min(isi, jnp.where(valid & ~inside, gap, m + 1), m + 1)
    rank = gap - 1
    last = rank == jnp.sum(bursts) - 1
    after, before = gap_min[jnp.minimum(rank + 1, m)], gap_min[jnp.maximum(rank, 0)]
    min_out = jnp.where(~last | (end < n_isi), after, before)
    return bursts, jnp.where(bursts, _burstiness(min_in, min_out), jnp.nan)


@jax.jit
def burst_metrics(
    sp: Spikes, tol: float = 0.5, pause_cost: float = 1.0
) -> tuple[Array, Array]:
    """Largest burstiness index of the detected bursts and their number.

    Bursts are runs of direct-reset ISIs between detour resets (Allen's
    ``detect_bursts``). The burstiness index compares the highest rate inside a
    burst with the highest rate outside it, in ``(-1, 1)``. ``tol`` (mV) and
    ``pause_cost`` are Allen's ``_process_bursts`` arguments. ``(0.0, 0)``
    without bursts.
    """
    bursts, index = _detect_bursts(sp, tol, pause_cost)
    n = jnp.sum(bursts)
    largest = jnp.max(jnp.where(bursts, index, -jnp.inf))
    return jnp.where(n > 0, largest, 0.0), n


def _solve(a: Array, b: Array) -> Array:
    """``a^-1 @ b`` for a 1x1 or 2x2 matrix ``a``."""
    if a.shape[0] == 1:
        return b / a[0, 0]
    det = a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]
    inv = jnp.stack([jnp.stack([a[1, 1], -a[0, 1]]), jnp.stack([-a[1, 0], a[0, 0]])])
    return inv @ b / det


def _window_reduce(fn, t: Array, v: Array, lo: Array, hi: Array, init, combine=jnp.add):
    """Combine ``fn(t, v, inside)`` over chunks of the window ``[lo, hi)`` of the
    traces; ``fn`` gets one chunk of each and its mask of window samples."""
    size = t.shape[0]
    chunk = min(_CHUNK, size)

    def add(c, total):
        start = jnp.minimum(c * chunk, size - chunk)  # the last chunk moves back
        idx = start + jnp.arange(chunk)
        inside = (idx >= jnp.maximum(lo, c * chunk)) & (idx < hi)
        t_c = lax.dynamic_slice(t, (start,), (chunk,))
        v_c = lax.dynamic_slice(v, (start,), (chunk,))
        return jax.tree.map(combine, total, fn(t_c, v_c, inside))

    return lax.fori_loop(lo // chunk, (hi - 1) // chunk + 1, add, init)


def _window_sums(k: Array, window: tuple) -> tuple[Array, ...]:
    """``e^T e``, ``e^T d``, ``(x e)^T e``, ``(x e)^T d`` and ``(x e)^T (x e)`` over
    the fit window, with ``e[:, j] = exp(-k[j] * x)``, ``x = t - t[lo]`` and
    ``d = v - y0``."""
    t, v, lo, hi, y0, _ = window

    def chunk_sums(t_c, v_c, inside):
        # Samples on the last axis: reductions along it vectorize best on CPU.
        x = jnp.where(inside, t_c - t[lo], 0.0)
        d = jnp.where(inside, v_c - y0, 0.0)
        e = jnp.where(inside, jnp.exp(-k[:, None] * x), 0.0)
        ee = e[:, None, :] * e[None, :, :]
        return (
            ee.sum(-1),
            (e * d).sum(-1),
            (x * ee).sum(-1),
            (x * e * d).sum(-1),
            (x * x * ee).sum(-1),
        )

    p = k.shape[0]
    square, vector = jnp.zeros((p, p), t.dtype), jnp.zeros(p, t.dtype)
    return _window_reduce(chunk_sums, t, v, lo, hi, (square, vector) * 2 + (square,))


def _exp_sums(log_k: Array, window: tuple) -> tuple[Array, Array, Array, Array]:
    """Least-squares fit of ``d`` by ``sum_j a_j * exp(-k_j * x)`` over the fit
    window for fixed rates ``k``: SSE, amplitudes ``a``, and the gradient and
    Gauss-Newton curvature of the (halved) SSE in ``log(k)`` with ``a`` projected
    out (Kaufman's variable projection)."""
    k = jnp.exp(log_k)
    gram, ed, xee, xed, xexe = _window_sums(k, window)
    a = _solve(gram, ed)
    ak = a * k
    curvature = jnp.outer(ak, ak) * (xexe - xee @ _solve(gram, xee))
    return window[-1] - ed @ a, a, ak * (xed - xee @ a), curvature


# Status of _fit_exps.
_RUNNING, _CONVERGED, _DEGENERATE = 0, 1, 2


def _fit_exps(log_k: Array, window: tuple, max_iters: int = 200):
    """Levenberg-Marquardt fit of ``d ~ sum_j a_j * exp(-k_j * x)`` over the rates
    (one or two), started at ``log_k``; returns ``(log_k, a, sse, status)``.

    ``status`` is ``_CONVERGED``, ``_RUNNING`` (``max_iters`` reached) or
    ``_DEGENERATE``: two rates merged or a rate ran off towards a constant
    (``k * span < 1e-6``) or a step at the first sample, where the fit has no
    least-squares optimum.
    """
    t, _, lo, hi, *_ = window
    span, first = t[hi - 1] - t[lo], t[lo + 1] - t[lo]
    eps = jnp.finfo(t.dtype).eps
    # Rates to ~1e-6 (Allen's optimizer stops at ~1e-5), or the SSE no longer drops.
    step_tol, sse_tol = jnp.maximum(1e-6, jnp.sqrt(eps)), jnp.maximum(1e-12, eps)
    sse, a, grad, curvature = _exp_sums(log_k, window)

    def step(state):
        it, log_k, sse, a, grad, curvature, damping, _ = state
        damped = curvature + damping * jnp.diag(jnp.diag(curvature))
        delta = jnp.clip(-_solve(damped, grad), -2.0, 2.0)
        new = _exp_sums(log_k + delta, window)
        accept = new[0] <= sse
        done = accept & (
            (jnp.max(jnp.abs(delta)) < step_tol) | (sse - new[0] <= sse_tol * sse)
        )
        log_k = jnp.where(accept, log_k + delta, log_k)
        sse, a, grad, curvature = (
            jnp.where(accept, n, o) for n, o in zip(new, (sse, a, grad, curvature))
        )
        damping = jnp.where(accept, jnp.maximum(damping / 3, 1e-15), damping * 4)
        k = jnp.exp(log_k)
        degenerate = jnp.any(k * span < 1e-6) | jnp.any(k * first > 50.0)
        if log_k.shape[0] == 2:
            degenerate = degenerate | (jnp.abs(log_k[0] - log_k[1]) < 1e-3)
        status = jnp.where(done | (damping > 1e12), _CONVERGED, _RUNNING)
        status = jnp.where(degenerate, _DEGENERATE, status)
        return it + 1, log_k, sse, a, grad, curvature, damping, status

    def keep_going(state):
        it, *_, status = state
        return (it < max_iters) & (status == _RUNNING)

    damping = jnp.asarray(1e-3, t.dtype)
    init = (0, log_k, sse, a, grad, curvature, damping, _RUNNING)
    _, log_k, sse, a, *_, status = lax.while_loop(keep_going, step, init)
    return log_k, a, sse, status


def _dominant_tau(tau: Array, a: Array, tau_limit: Array) -> Array:
    """Allen's choice between the time constants of a double exponential: the
    faster one if it has the larger weight, else the slower one if they are within
    10%, else the faster one if the slower exceeds ``tau_limit`` without twice its
    weight, else the slower one."""
    first_faster = tau[0] < tau[1]
    fast_tau, slow_tau = jnp.where(first_faster, tau, tau[::-1])
    fast_a, slow_a = jnp.where(first_faster, a, a[::-1])
    return jnp.where(
        jnp.abs(fast_a) > jnp.abs(slow_a),
        fast_tau,
        jnp.where(
            (slow_tau - fast_tau) / slow_tau <= 0.1,
            slow_tau,
            jnp.where(
                (slow_tau > tau_limit) & (slow_a / fast_a < 2.0), fast_tau, slow_tau
            ),
        ),
    )


def _prespike_fit(
    t: Array, v: Array, lo: Array, hi: Array, tau_limit: Array
) -> tuple[Array, Array, Array]:
    """Allen's ``fit_prespike_time_constant`` on ``[lo, hi)``: the dominant time
    constant (s) of the rise of ``v`` towards just above its maximum, the single
    exponential's time constant and the SSE of the fit used."""
    zero = jnp.zeros((), t.dtype)

    def extremes(t_c, v_c, inside):
        return tuple(jnp.max(jnp.where(inside, x, -jnp.inf)) for x in (v_c, -v_c))

    v_max, minus_v_min = _window_reduce(
        extremes, t, v, lo, hi, (-jnp.inf, -jnp.inf), jnp.maximum
    )
    y0 = v_max + 5e-6

    def line_sums(t_c, v_c, inside):
        x = jnp.where(inside, t_c - t[lo], 0.0)
        d = jnp.where(inside, v_c - y0, 0.0)
        log_d = jnp.where(inside, jnp.log(jnp.where(inside, y0 - v_c, 1.0)), 0.0)
        return tuple(jnp.sum(s) for s in (d * d, x, x * x, log_d, x * log_d))

    dd, sx, sxx, sy, sxy = _window_reduce(line_sums, t, v, lo, hi, (zero,) * 5)
    window = (t, v, lo, hi, y0, dd)
    span, first = t[hi - 1] - t[lo], t[lo + 1] - t[lo]

    # Single exponential, started like Allen's at a straight-line fit of log(y0 - v).
    n = hi - lo
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    log_k0 = jnp.log(jnp.maximum(-slope, 1.0 / span))[None]
    log_k, _, sse, _ = _fit_exps(log_k0, window)
    single_tau = jnp.exp(-log_k[0])

    # Double exponentials (Allen starts both components at the single one, where
    # its optimizer usually stays). Fits towards merged time constants (the
    # components then cancel each other with amplitudes far above the voltage
    # range), or with a time constant above 100x the window (a constant: v would
    # approach a level above y0), are not double exponentials; without a
    # better-fitting one, the single exponential is used.
    starts = log_k + jnp.asarray(_DOUBLE_EXP_STARTS, t.dtype)
    log_k2, a2, sse2, status = jax.vmap(_fit_exps, (0, None))(starts, window)
    k2 = jnp.exp(log_k2)
    double = (
        (status != _DEGENERATE)
        & jnp.all(k2 * span > 1e-2, axis=1)
        & jnp.all(k2 * first <= 50.0, axis=1)
        & (jnp.abs(log_k2[:, 0] - log_k2[:, 1]) > 1e-2)
        & jnp.all(jnp.abs(a2) <= 2.0 * (y0 + minus_v_min), axis=1)
        & (sse2 < sse)
    )
    best = jnp.argmin(jnp.where(double, sse2, jnp.inf))
    tau = _dominant_tau(1.0 / k2[best], a2[best], tau_limit)
    use_double = jnp.any(double)
    return (
        jnp.where(use_double, tau, single_tau),
        single_tau,
        jnp.where(use_double, sse2[best], sse),
    )


@jax.jit
def _delay_fit(sp: Spikes, tau_limit: float) -> Array:
    """``_prespike_fit`` from ``start`` to the first threshold, NaN where Allen's
    fit fails: without spikes or with fewer than 4 samples (one per parameter)."""
    t = sp.t
    lo = find_time_index(t, sp.start)
    hi = find_time_index(t, take(t, sp.threshold[0]))
    fits = (sp.threshold[0] >= 0) & (hi - lo >= 4)
    # Under vmap both branches run: sweeps without a fit get a short window.
    hi = jnp.where(fits, hi, lo + 4)
    return lax.cond(
        fits,
        lambda *args: jnp.stack(_prespike_fit(*args)),
        lambda *_: jnp.full(3, jnp.nan, t.dtype),
        t,
        sp.v,
        lo,
        hi,
        jnp.asarray(tau_limit, t.dtype),
    )


@jax.jit
def delay_metrics(sp: Spikes, tau_limit: float = 0.3) -> tuple[Array, Array]:
    """Delay ratio and the dominant time constant (ms) of the rise before the first
    spike (Allen's ``delay_metrics``).

    The voltage from ``start`` to the first threshold is fitted with a double
    exponential approaching just above its maximum, and the dominant time
    constant is chosen with Allen's rules (``tau_limit`` in s). The delay ratio is
    the latency over it: high for cells that start firing long after their
    membrane has charged. Meaningful for long squares; on ramps and short squares
    the rise is not exponential. NaN without spikes or with fewer than 4 samples
    before the first one (Allen: 0).

    Differs from Allen where Allen's fit misses the double exponential: Allen
    starts both components at the same time constant, and its optimizer mostly
    stays there, i.e. returns a single exponential. jaxley_ephys also fits double
    exponentials from two other starts and uses the best one if its time
    constants are distinct and it fits better than the single exponential. On the
    cached Allen cells this changes 29 of 85 long squares, e.g. a delay ratio of
    130 instead of 1; where Allen's fit separates the components, the results
    agree. See ``allen_ephys/ALLEN_EPHYS_FIXES.md``, section 6.
    """
    tau = _delay_fit(sp, tau_limit)[0]
    latency = take(sp.t, sp.threshold[0]) - sp.start
    return latency / tau, 1e3 * tau
