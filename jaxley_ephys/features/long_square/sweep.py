from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array


BASELINE_INTERVAL = 0.1
SAG_PEAK_WIDTH = 0.005
STEP_STIM_TOL = 1e-6
TAU_RMSE_MAX = 1.0
TAU_INV_GRID_SIZE = 256


def _analysis_window(t: Array, ev) -> tuple[Array, Array, Array]:
    window = ev.window
    has_stim = jnp.any(window)
    start_idx = jnp.argmax(window)
    end_idx = (t.shape[0] - 1) - jnp.argmax(window[::-1])
    has_stim = has_stim & (end_idx > start_idx)
    return start_idx, end_idx, has_stim


def _mean_in_window(t: Array, values: Array, start_t: Array, end_t: Array) -> Array:
    mask = (t >= start_t) & (t < end_t)
    count = jnp.sum(mask)
    vals = jnp.where(mask, values, jnp.nan)
    return jnp.where(count > 0, jnp.nanmean(vals), jnp.nan)


def _first_valid(values: Array, valid: Array) -> Array:
    idx = jnp.argmax(valid)
    return jnp.where(jnp.any(valid), values[idx], jnp.nan)


def _masked_median(values: Array, valid: Array) -> Array:
    count = jnp.sum(valid)
    padded = jnp.where(valid, values, jnp.inf)
    sorted_vals = jnp.sort(padded)
    lo = jnp.maximum((count - 1) // 2, 0)
    hi = jnp.maximum(count // 2, 0)
    med = (sorted_vals[lo] + sorted_vals[hi]) / 2.0
    return jnp.where(count > 0, med, jnp.nan)


def _threshold_times(t: Array, ev) -> Array:
    return jnp.where(
        ev.spike_valid,
        t[jnp.where(ev.spike_valid, ev.threshold_idxs, 0)],
        jnp.nan,
    )


def _isi_values(t: Array, ev) -> tuple[Array, Array]:
    threshold_t = _threshold_times(t, ev)
    isis = threshold_t[1:] - threshold_t[:-1]
    valid = ev.spike_valid[1:] & ev.spike_valid[:-1]
    return isis, valid


def _adaptation_index(isis: Array, valid: Array) -> Array:
    pair_valid = valid[1:] & valid[:-1]
    num = isis[1:] - isis[:-1]
    den = isis[1:] + isis[:-1]
    norm = jnp.where(pair_valid & (den != 0.0), num / den, jnp.nan)
    norm = jnp.where(pair_valid & (den == 0.0), 0.0, norm)
    return jnp.where(jnp.sum(pair_valid) > 0, jnp.nanmean(norm), jnp.nan)


def _voltage_deflection(t: Array, v: Array, i: Array, ev, _dvdt) -> tuple[Array, Array, Array]:
    start_idx, end_idx, has_stim = _analysis_window(t, ev)
    x = jnp.arange(t.shape[0])
    in_window = has_stim & (x >= start_idx) & (x < end_idx)
    count = jnp.sum(in_window)

    mid_idx = (start_idx + end_idx) // 2
    hyper = has_stim & (i[mid_idx] < 0.0)
    min_cand = jnp.where(in_window, v, jnp.inf)
    max_cand = jnp.where(in_window, v, -jnp.inf)
    deflect_idx = jnp.where(hyper, jnp.argmin(min_cand), jnp.argmax(max_cand))
    valid = count > 0
    deflect_v = jnp.where(valid, v[deflect_idx], jnp.nan)
    deflect_idx = jnp.where(valid, deflect_idx, -1)
    return deflect_v, deflect_idx, valid


def adapt(t: Array, v: Array, i: Array, ev, dvdt: Array) -> Array:
    del v, i, dvdt
    isis, isis_valid = _isi_values(t, ev)
    return _adaptation_index(isis, isis_valid)


def latency(t: Array, v: Array, i: Array, ev, dvdt: Array) -> Array:
    del v, dvdt
    start_idx, _, has_stim = _analysis_window(t, ev)
    threshold_t = _threshold_times(t, ev)
    return jnp.where(
        has_stim & (jnp.sum(ev.spike_valid) > 0),
        threshold_t[0] - t[start_idx],
        jnp.nan,
    )


def isi_cv(t: Array, v: Array, i: Array, ev, dvdt: Array) -> Array:
    del v, i, dvdt
    isis, isis_valid = _isi_values(t, ev)
    vals = jnp.where(isis_valid, isis, jnp.nan)
    n_isis = jnp.sum(isis_valid)
    mean = jnp.where(n_isis > 0, jnp.nanmean(vals), jnp.nan)
    std = jnp.where(n_isis > 0, jnp.nanstd(vals), jnp.nan)
    return jnp.where(n_isis > 0, std / mean, jnp.nan)


def mean_isi(t: Array, v: Array, i: Array, ev, dvdt: Array) -> Array:
    del v, i, dvdt
    isis, isis_valid = _isi_values(t, ev)
    vals = jnp.where(isis_valid, isis, jnp.nan)
    return jnp.where(jnp.sum(isis_valid) > 0, 1e3 * jnp.nanmean(vals), jnp.nan)


def median_isi(t: Array, v: Array, i: Array, ev, dvdt: Array) -> Array:
    del v, i, dvdt
    isis, isis_valid = _isi_values(t, ev)
    return _masked_median(isis, isis_valid)


def first_isi(t: Array, v: Array, i: Array, ev, dvdt: Array) -> Array:
    del v, i, dvdt
    isis, isis_valid = _isi_values(t, ev)
    return _first_valid(isis, isis_valid)


def avg_rate(t: Array, v: Array, i: Array, ev, dvdt: Array) -> Array:
    del v, i, dvdt
    start_idx, end_idx, has_stim = _analysis_window(t, ev)
    duration = t[end_idx] - t[start_idx]
    n_spikes = jnp.sum(ev.spike_valid)
    return jnp.where(has_stim & (duration > 0.0), n_spikes / duration, jnp.nan)


def v_baseline(t: Array, v: Array, i: Array, ev, dvdt: Array) -> Array:
    del i, dvdt
    start_idx, _, has_stim = _analysis_window(t, ev)
    start_t = t[start_idx]
    baseline = _mean_in_window(t, v, start_t - BASELINE_INTERVAL, start_t)
    return jnp.where(has_stim, baseline, jnp.nan)


def tau(t: Array, v: Array, i: Array, ev, dvdt: Array) -> Array:
    start_idx, end_idx, has_stim = _analysis_window(t, ev)
    mid_idx = start_idx + 1
    mid_idx = jnp.minimum(mid_idx, end_idx)
    hyper = has_stim & (i[mid_idx] < 0.0)

    v0, peak_idx, peak_valid = _voltage_deflection(t, v, i, ev, dvdt)
    baseline = v_baseline(t, v, i, ev, dvdt)
    finite_inputs = jnp.isfinite(v0) & jnp.isfinite(baseline)
    has_deflection = v0 < baseline

    frac_level = 0.1 * (v0 - baseline) + baseline
    x = jnp.arange(t.shape[0])
    after_start = x >= start_idx
    crossed = after_start & (v <= frac_level)
    has_cross = jnp.any(crossed)
    fit_start_idx = jnp.where(has_cross, jnp.argmax(crossed), start_idx)

    fit_mask = (x >= fit_start_idx) & (x < peak_idx)
    t_rel = t - t[fit_start_idx]
    reg_mask = fit_mask
    n = jnp.sum(reg_mask)

    valid = hyper & peak_valid & finite_inputs & has_deflection & has_cross & (n >= 4)

    def _fit_tau(_: None) -> Array:
        t_fit = jnp.where(reg_mask, t_rel, 0.0)
        v_fit = jnp.where(reg_mask, v, 0.0)
        mask_f = reg_mask.astype(t.dtype)
        n_f = jnp.maximum(jnp.sum(mask_f), 1.0)

        inv_grid = jnp.logspace(
            0.0,
            4.0,
            num=TAU_INV_GRID_SIZE,
            dtype=t.dtype,
        )
        e = jnp.exp(-inv_grid[:, None] * t_fit[None, :])
        m = mask_f[None, :]
        em = e * m
        vv = v_fit[None, :]

        s_ee = jnp.sum(em * e, axis=1)
        s_e1 = jnp.sum(em, axis=1)
        s_11 = jnp.sum(m, axis=1)
        s_ev = jnp.sum(em * vv, axis=1)
        s_1v = jnp.sum(m * vv, axis=1)

        det = s_ee * s_11 - s_e1 * s_e1
        valid_det = det > 0.0
        safe_det = jnp.where(valid_det, det, 1.0)

        a = (s_ev * s_11 - s_1v * s_e1) / safe_det
        y0 = (s_ee * s_1v - s_e1 * s_ev) / safe_det
        pred = a[:, None] * e + y0[:, None]
        resid = (pred - vv) * m
        rmse = jnp.sqrt(jnp.sum(resid * resid, axis=1) / n_f)
        rmse = jnp.where(valid_det, rmse, jnp.inf)

        best = jnp.argmin(rmse)
        best_rmse = rmse[best]
        best_inv = inv_grid[best]
        return jnp.where(best_rmse <= TAU_RMSE_MAX, 1e3 / best_inv, jnp.nan)

    return jax.lax.cond(valid, _fit_tau, lambda _: jnp.nan, operand=None)


def sag(t: Array, v: Array, i: Array, ev, dvdt: Array) -> Array:
    start_idx, end_idx, has_stim = _analysis_window(t, ev)
    mid_idx = (start_idx + end_idx) // 2
    hyper = has_stim & (i[mid_idx] < 0.0)

    _, peak_idx, peak_valid = _voltage_deflection(t, v, i, ev, dvdt)
    peak_t = jnp.where(peak_valid, t[peak_idx], 0.0)
    v_peak_avg = _mean_in_window(
        t,
        v,
        peak_t - SAG_PEAK_WIDTH / 2.0,
        peak_t + SAG_PEAK_WIDTH / 2.0,
    )
    vb = v_baseline(t, v, i, ev, dvdt)
    v_steady = _mean_in_window(t, v, t[end_idx] - BASELINE_INTERVAL, t[end_idx])
    den = v_peak_avg - vb
    out = jnp.where(den != 0.0, (v_peak_avg - v_steady) / den, jnp.nan)
    return jnp.where(hyper & peak_valid, out, jnp.nan)


def peak_deflect(t: Array, v: Array, i: Array, ev, dvdt: Array) -> Array:
    deflect_v, _, _ = _voltage_deflection(t, v, i, ev, dvdt)
    return deflect_v


def peak_deflect_index(t: Array, v: Array, i: Array, ev, dvdt: Array) -> Array:
    _, deflect_idx, _ = _voltage_deflection(t, v, i, ev, dvdt)
    return deflect_idx


def stim_amp(t: Array, v: Array, i: Array, ev, dvdt: Array) -> Array:
    del v, dvdt
    start_idx, end_idx, has_stim = _analysis_window(t, ev)
    mid_idx = (start_idx + end_idx) // 2
    x = jnp.arange(t.shape[0])
    in_window = has_stim & (x >= start_idx) & (x < end_idx)
    amp = i[mid_idx]
    max_dev = jnp.max(jnp.where(in_window, jnp.abs(i - amp), 0.0))
    is_step = has_stim & (max_dev <= STEP_STIM_TOL)
    return jnp.where(is_step, amp, jnp.nan)


SWEEP_FEATURE_FUNCTIONS: dict[str, Callable[[Array, Array, Array, object, Array], Array]] = {
    "adapt": adapt,
    "latency": latency,
    "isi_cv": isi_cv,
    "mean_isi": mean_isi,
    "median_isi": median_isi,
    "first_isi": first_isi,
    "avg_rate": avg_rate,
    "v_baseline": v_baseline,
    "tau": tau,
    "sag": sag,
    "peak_deflect": peak_deflect,
    "peak_deflect_index": peak_deflect_index,
    "stim_amp": stim_amp,
}


def compute_sweep_features(t: Array, v: Array, i: Array, ev, dvdt: Array) -> dict[str, Array]:
    return {
        feature_name: feature_fn(t, v, i, ev, dvdt)
        for feature_name, feature_fn in SWEEP_FEATURE_FUNCTIONS.items()
    }
