from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from typing import Optional

from .bessel import allen_bessel_sosfiltfilt


def _raise_if_overflow(overflow: Array, max_spikes: int) -> None:
    if isinstance(overflow, jax.core.Tracer):
        return
    if bool(overflow):
        raise ValueError(
            "detect_events overflowed max_spikes="
            f"{max_spikes}; increase max_spikes for this sweep"
        )


@jax.tree_util.register_dataclass
@dataclass
class Events:
    spikes: Array # (T,) int array of spike labels, -1 for non-spike timepoints
    thresholds: Array # (T,) bool array of threshold crossings
    peaks: Array # (T,) bool array of peak timepoints
    troughs: Array # (T,) bool array of trough timepoints

    n_spikes: Array # scalar int number of detected spikes
    overflow: Array # scalar bool whether overflow occurred
    start_idx: Array # scalar int start index of analysis window 
    end_idx: Array # scalar int end index of analysis window

    # mask
    spike_valid: Array # (max_spikes,) bool array of which spike labels are valid; rest are -1
    
    # indices of events
    threshold_idxs: Array # (max_spikes,) int array of threshold crossing indices
    peak_idxs: Array # (max_spikes,) int array of peak indices
    trough_idxs: Array # (max_spikes,) int array of trough indices
    upstroke_idxs: Array # (max_spikes,) int array of upstroke indices
    downstroke_idxs: Array # (max_spikes,) int array of downstroke indices

    # Needed because last AP downstroke can be cut off at window end.
    trough_valid: Array # (max_spikes,) bool array of which trough indices are valid
    downstroke_valid: Array # (max_spikes,) bool array of which downstroke indices are valid
    clipped: Array # (max_spikes,) bool array of which spikes did not return to baseline

def _smooth5(v: Array) -> Array:
    kernel = jnp.asarray([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel = kernel / jnp.sum(kernel)
    return jnp.convolve(v, kernel, mode="same")


def compute_dvdt(
    t: Array,
    v: Array,
    *,
    filter: float | None = None,
) -> Array:
    if t.ndim != 1 or v.ndim != 1 or t.shape[0] != v.shape[0]:
        raise ValueError("compute_dvdt expects 1D aligned t and v")
    if t.shape[0] < 2:
        return jnp.full(t.shape, jnp.nan)

    if filter is None:
        v_used = v
    else:
        dt = jnp.diff(t)
        fixed_dt = jnp.all(jnp.isclose(dt, dt[0], rtol=1e-3, atol=1e-9))
        sample_freq = 1.0 / dt[0]
        filt_coeff = (filter * 1e3) / (sample_freq / 2.0)
        valid_coeff = (filt_coeff >= 0) & (filt_coeff < 1)
        use_bessel = fixed_dt & valid_coeff
        v_used = jax.lax.cond(
            use_bessel,
            lambda _: allen_bessel_sosfiltfilt(v, t, cutoff_khz=filter),
            lambda _: _smooth5(v),
            operand=None,
        )

    dv = jnp.diff(v_used)

    dt = jnp.diff(t)
    short = jnp.where(dt != 0.0, 1e-3 * dv / dt, jnp.nan)
    out = jnp.full(t.shape, jnp.nan)
    return out.at[:-1].set(short)


def _first_index_at_or_after(t: Array, t0: float | Array) -> Array:
    mask = t >= t0
    idx = jnp.argmax(mask)
    return jnp.where(jnp.any(mask), idx, t.shape[0] - 1)


def _compact(values: Array, valid: Array) -> tuple[Array, Array]:
    n = values.shape[0]
    order = jnp.argsort(jnp.where(valid, jnp.arange(n), n), stable=True)
    sorted_vals = values[order]
    n_valid = jnp.sum(valid)
    sorted_valid = jnp.arange(n) < n_valid
    sorted_vals = jnp.where(sorted_valid, sorted_vals, -1)
    return sorted_vals, sorted_valid


def _masked_take(values: Array, idx: Array, valid: Array, fill: float) -> Array:
    safe_idx = jnp.where(valid, idx, 0)
    return jnp.where(valid, values[safe_idx], fill)


def _indices_to_mask(length: int, idx: Array) -> Array:
    x = jnp.arange(length)
    return jnp.any(x[:, None] == idx[None, :], axis=1)


def _next_valid_or_default(values: Array, valid: Array, default: Array) -> Array:
    shifted_values = jnp.concatenate([values[1:], jnp.asarray([default])])
    shifted_valid = jnp.concatenate([valid[1:], jnp.asarray([False])])
    return jnp.where(shifted_valid, shifted_values, default)


def _detect_putative_spikes(
    dvdt: Array,
    start_idx: Array,
    end_idx: Array,
    max_spikes: int,
    dv_cutoff: float,
) -> tuple[Array, Array, Array]:
    t_len = dvdt.shape[0]
    ge = dvdt >= dv_cutoff
    crossing_pos = jnp.arange(t_len - 1)
    crossing_mask = (~ge[:-1]) & ge[1:]
    crossing_in_window = (
        crossing_mask
        & (crossing_pos >= start_idx)
        & (crossing_pos <= end_idx)
        & (start_idx < end_idx)
    )
    overflow = jnp.sum(crossing_in_window) > max_spikes

    crossings = jnp.where(crossing_in_window, size=max_spikes, fill_value=-1)[0]
    spike_idx = crossings
    spike_valid = spike_idx >= 0
    return spike_idx, spike_valid, overflow


def _find_peak_indexes(
    v: Array,
    spike_idx: Array,
    spike_valid: Array,
    end_idx: Array,
) -> tuple[Array, Array]:
    max_spikes = spike_idx.shape[0]
    t_len = v.shape[0]
    x = jnp.arange(t_len)
    next_spike = _next_valid_or_default(spike_idx, spike_valid, end_idx)

    def peak_for(k: Array) -> Array:
        s = spike_idx[k]
        n = next_spike[k]
        valid = spike_valid[k] & (n > s)
        m = valid & (x >= s) & (x < n)
        vals = jnp.where(m, v, -jnp.inf)
        return jnp.where(valid, jnp.argmax(vals), -1)

    peak_idx = jax.vmap(peak_for)(jnp.arange(max_spikes))
    peak_valid = peak_idx >= 0
    return peak_idx, peak_valid


def _filter_putative_spikes(
    v: Array,
    spike_idx: Array,
    spike_valid: Array,
    peak_idx: Array,
    peak_valid: Array,
    min_height: float,
    min_peak: float,
) -> tuple[Array, Array, Array, Array]:
    peak_v = _masked_take(v, peak_idx, peak_valid, fill=-jnp.inf)
    spike_v = _masked_take(v, spike_idx, spike_valid, fill=jnp.inf)
    keep = spike_valid & peak_valid
    keep = keep & (peak_v >= min_peak)
    keep = keep & ((peak_v - spike_v) >= min_height)
    spike_idx, spike_valid = _compact(spike_idx, keep)
    peak_idx, peak_valid = _compact(peak_idx, keep)
    return spike_idx, spike_valid, peak_idx, peak_valid


def _find_upstroke_indexes(
    dvdt: Array,
    spike_idx: Array,
    spike_valid: Array,
    peak_idx: Array,
    peak_valid: Array,
) -> tuple[Array, Array]:
    max_spikes = spike_idx.shape[0]
    t_len = dvdt.shape[0]
    x = jnp.arange(t_len)

    def upstroke_for(k: Array) -> Array:
        s = spike_idx[k]
        p = peak_idx[k]
        valid = spike_valid[k] & peak_valid[k] & (p > s)
        m = valid & (x >= s) & (x < p)
        vals = jnp.where(m, dvdt, -jnp.inf)
        return jnp.where(valid, jnp.argmax(vals), -1)

    upstroke_idx = jax.vmap(upstroke_for)(jnp.arange(max_spikes))
    up_valid = upstroke_idx >= 0
    return upstroke_idx, up_valid


def _refine_threshold_indexes(
    dvdt: Array,
    upstroke_idx: Array,
    up_valid: Array,
    thresh_frac: float = 0.05,
) -> tuple[Array, Array]:
    max_spikes = upstroke_idx.shape[0]
    t_len = dvdt.shape[0]
    x = jnp.arange(t_len)
    xr = x[::-1]
    up_vals = _masked_take(dvdt, upstroke_idx, up_valid, fill=jnp.nan)
    avg_up = jnp.nanmean(up_vals)
    target = avg_up * thresh_frac
    prev_up = jnp.concatenate([jnp.asarray([0]), upstroke_idx[:-1]])

    def threshold_for(k: Array) -> Array:
        u = upstroke_idx[k]
        p = prev_up[k]
        valid = up_valid[k] & (u > p)
        m = valid & (xr <= u) & (xr > p)
        cand = jnp.where(m & (dvdt[xr] <= target), xr, -1)
        best = jnp.max(cand)
        return jnp.where(best >= 0, best, p)

    threshold_idx = jax.vmap(threshold_for)(jnp.arange(max_spikes))
    threshold_valid = up_valid
    return threshold_idx, threshold_valid


def _check_thresholds_and_peaks(
    t: Array,
    v: Array,
    threshold_idx: Array,
    threshold_valid: Array,
    peak_idx: Array,
    peak_valid: Array,
    end_idx: Array,
    max_interval: float = 0.005,
) -> tuple[Array, Array, Array, Array, Array]:
    max_spikes = threshold_idx.shape[0]
    t_len = t.shape[0]
    x = jnp.arange(t_len)

    peak_t = _masked_take(t, peak_idx, peak_valid, fill=jnp.inf)
    threshold_t = _masked_take(t, threshold_idx, threshold_valid, fill=-jnp.inf)
    pair_ok = threshold_valid & peak_valid & ((peak_t - threshold_t) < max_interval)
    threshold_idx, threshold_valid = _compact(threshold_idx, pair_ok)
    peak_idx, peak_valid = _compact(peak_idx, pair_ok)

    valid_pair = peak_valid & threshold_valid
    n_valid = jnp.sum(valid_pair)
    last_idx = jnp.maximum(n_valid - 1, 0)
    p_last = peak_idx[last_idx]
    th_last = threshold_idx[last_idx]
    returned = jnp.any(
        (x >= p_last)
        & (x <= end_idx)
        & (v <= (v[jnp.maximum(th_last, 0)] + 1.0))
    )
    clipped_last = (n_valid > 0) & (~returned)
    clipped = jnp.zeros((max_spikes,), dtype=bool).at[last_idx].set(clipped_last)

    return threshold_idx, threshold_valid, peak_idx, peak_valid, clipped


def _find_trough_indexes(
    v: Array,
    peak_idx: Array,
    peak_valid: Array,
    threshold_idx: Array,
    threshold_valid: Array,
    clipped: Array,
    end_idx: Array,
) -> tuple[Array, Array]:
    max_spikes = peak_idx.shape[0]
    t_len = v.shape[0]
    x = jnp.arange(t_len)
    next_thr = _next_valid_or_default(threshold_idx, threshold_valid, end_idx)

    def trough_for(k: Array) -> Array:
        p = peak_idx[k]
        n = next_thr[k]
        valid = peak_valid[k] & (n > p) & (~clipped[k])
        m = valid & (x >= p) & (x < n)
        vals = jnp.where(m, v, jnp.inf)
        return jnp.where(valid, jnp.argmin(vals), -1)

    trough_idx = jax.vmap(trough_for)(jnp.arange(max_spikes))
    trough_valid = trough_idx >= 0
    return trough_idx, trough_valid


def _find_downstroke_indexes(
    dvdt: Array,
    peak_idx: Array,
    spike_valid: Array,
    trough_idx: Array,
    trough_valid: Array,
) -> tuple[Array, Array]:
    max_spikes = peak_idx.shape[0]
    t_len = dvdt.shape[0]
    x = jnp.arange(t_len)

    def downstroke_for(k: Array) -> Array:
        p = peak_idx[k]
        tr = trough_idx[k]
        valid = spike_valid[k] & trough_valid[k] & (tr > p)
        m = valid & (x >= p) & (x < tr)
        vals = jnp.where(m, dvdt, jnp.inf)
        return jnp.where(valid, jnp.argmin(vals), -1)

    downstroke_idx = jax.vmap(downstroke_for)(jnp.arange(max_spikes))
    downstroke_valid = downstroke_idx >= 0
    return downstroke_idx, downstroke_valid


def detect_events(
    t: Array,
    v: Array,
    i: Array,
    t0: float | Array,
    t1: float | Array,
    *,
    dvdt: Optional[Array] = None,
    max_spikes: int = 256,
    dvdt_filter: float | None = 10.0,
    threshold_cutoff: float = 20.0,
    peak_cutoff: float = -30.0,
    trough_cutoff: float = 2.0,
) -> Events:
    if t.ndim != 1 or v.ndim != 1 or i.ndim != 1:
        raise ValueError("detect_events expects 1D t, v, i")
    if not (t.shape[0] == v.shape[0] == i.shape[0]):
        raise ValueError("detect_events expects aligned t, v, i arrays")
    if max_spikes <= 0:
        raise ValueError("max_spikes must be > 0")

    T = t.shape[0]
    x = jnp.arange(T)
    start_idx = _first_index_at_or_after(t, t0)
    end_idx = _first_index_at_or_after(t, t1)

    # 1) dv/dt and putative spike detection.
    dvdt = (
        compute_dvdt(t, v, filter=dvdt_filter)
        if dvdt is None
        else dvdt
    )

    spike_idx, spike_valid, crossing_overflow = _detect_putative_spikes(
        dvdt,
        start_idx,
        end_idx,
        max_spikes,
        threshold_cutoff,
    )

    # 2) Peak detection and putative spike filtering.
    peak_idx, peak_valid = _find_peak_indexes(v, spike_idx, spike_valid, end_idx)
    spike_idx, spike_valid, peak_idx, peak_valid = _filter_putative_spikes(
        v,
        spike_idx,
        spike_valid,
        peak_idx,
        peak_valid,
        trough_cutoff,
        peak_cutoff,
    )

    # 3) Upstroke detection and threshold refinement.
    upstroke_idx, up_valid = _find_upstroke_indexes(
        dvdt,
        spike_idx,
        spike_valid,
        peak_idx,
        peak_valid,
    )
    threshold_idx, threshold_valid = _refine_threshold_indexes(
        dvdt,
        upstroke_idx,
        up_valid,
        0.05,
    )
    threshold_valid = threshold_valid & spike_valid

    # 4) Threshold/peak consistency checks and clipped final-spike handling.
    threshold_idx, threshold_valid, peak_idx, peak_valid, clipped = (
        _check_thresholds_and_peaks(
            t,
            v,
            threshold_idx,
            threshold_valid,
            peak_idx,
            peak_valid,
            end_idx,
            0.005,
        )
    )

    # Recompute upstrokes from finalized threshold/peak pairs.
    upstroke_idx, upstroke_valid = _find_upstroke_indexes(
        dvdt,
        threshold_idx,
        threshold_valid,
        peak_idx,
        peak_valid,
    )

    # 5) Trough detection.
    trough_idx, trough_valid = _find_trough_indexes(
        v,
        peak_idx,
        peak_valid,
        threshold_idx,
        threshold_valid,
        clipped,
        end_idx,
    )

    downstroke_idx, downstroke_valid = _find_downstroke_indexes(
        dvdt,
        peak_idx,
        threshold_valid,
        trough_idx,
        trough_valid,
    )

    threshold_idx = jnp.where(threshold_valid, threshold_idx, -1)
    peak_idx = jnp.where(peak_valid, peak_idx, -1)
    upstroke_idx = jnp.where(upstroke_valid, upstroke_idx, -1)
    trough_idx = jnp.where(trough_valid, trough_idx, -1)
    downstroke_idx = jnp.where(downstroke_valid, downstroke_idx, -1)

    threshold_mask = _indices_to_mask(T, threshold_idx)
    peak_mask = _indices_to_mask(T, peak_idx)
    trough_mask = _indices_to_mask(T, trough_idx)

    # assign spike labels by latest threshold
    th = jnp.where(threshold_valid, threshold_idx, -1)
    eligible = (th[None, :] <= x[:, None]) & threshold_valid[None, :]
    th_vals = jnp.where(eligible, th[None, :], -1)
    k_latest = jnp.argmax(th_vals, axis=1)
    has = jnp.any(eligible, axis=1)
    spike_labels = jnp.where(has, k_latest, -1)

    in_window_t = (x >= start_idx) & (x <= end_idx)
    threshold_mask = threshold_mask & in_window_t
    peak_mask = peak_mask & in_window_t
    trough_mask = trough_mask & in_window_t
    spike_labels = jnp.where(in_window_t, spike_labels, -1)

    n_spikes = jnp.maximum(jnp.max(spike_labels) + 1, 0)
    n_thresholds = jnp.sum(threshold_mask)
    n_peaks = jnp.sum(peak_mask)
    n_troughs = jnp.sum(trough_mask)
    overflow = (
        (n_spikes > max_spikes)
        | (n_thresholds > max_spikes)
        | (n_peaks > max_spikes)
        | (n_troughs > max_spikes)
        | crossing_overflow
    )

    _raise_if_overflow(overflow, max_spikes)

    return Events(
        spikes=spike_labels,
        thresholds=threshold_mask,
        peaks=peak_mask,
        troughs=trough_mask,
        n_spikes=n_spikes,
        overflow=overflow,
        start_idx=start_idx,
        end_idx=end_idx,
        spike_valid=threshold_valid,
        threshold_idxs=threshold_idx,
        peak_idxs=peak_idx,
        trough_idxs=trough_idx,
        upstroke_idxs=upstroke_idx,
        downstroke_idxs=downstroke_idx,
        trough_valid=trough_valid,
        downstroke_valid=downstroke_valid,
        clipped=clipped,
    )
