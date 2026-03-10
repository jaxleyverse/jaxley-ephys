from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from typing import Optional


@jax.tree_util.register_dataclass
@dataclass
class Events:
    spikes: Array
    thresholds: Array
    peaks: Array
    troughs: Array

    n_spikes: Array
    overflow: Array

    threshold_idxs: Array
    peak_idxs: Array
    trough_idxs: Array

    threshold_valid: Array
    peak_valid: Array
    trough_valid: Array

    @classmethod
    def from_masks(
        cls,
        spikes: Array,
        thresholds: Array,
        peaks: Array,
        troughs: Array,
        *,
        max_spikes: int = 128,
    ) -> Events:
        spikes = spikes.astype(jnp.int32)
        thresholds = thresholds.astype(jnp.bool_)
        peaks = peaks.astype(jnp.bool_)
        troughs = troughs.astype(jnp.bool_)

        n_spikes = jnp.maximum(jnp.max(spikes) + 1, 0).astype(jnp.int32)
        n_thresholds = jnp.sum(thresholds)
        n_peaks = jnp.sum(peaks)
        n_troughs = jnp.sum(troughs)

        overflow = (
            (n_spikes > max_spikes)
            | (n_thresholds > max_spikes)
            | (n_peaks > max_spikes)
            | (n_troughs > max_spikes)
        )

        threshold_idxs = jnp.where(thresholds, size=max_spikes, fill_value=-1)[0]
        peak_idxs = jnp.where(peaks, size=max_spikes, fill_value=-1)[0]
        trough_idxs = jnp.where(troughs, size=max_spikes, fill_value=-1)[0]

        threshold_valid = threshold_idxs != -1
        peak_valid = peak_idxs != -1
        trough_valid = trough_idxs != -1

        return cls(
            spikes=spikes,
            thresholds=thresholds,
            peaks=peaks,
            troughs=troughs,
            n_spikes=n_spikes,
            overflow=overflow,
            threshold_idxs=threshold_idxs.astype(jnp.int32),
            peak_idxs=peak_idxs.astype(jnp.int32),
            trough_idxs=trough_idxs.astype(jnp.int32),
            threshold_valid=threshold_valid,
            peak_valid=peak_valid,
            trough_valid=trough_valid,
        )


def compute_dvdt(t: Array, v: Array, *, filter: float | None = None) -> Array:
    if t.ndim != 1 or v.ndim != 1 or t.shape[0] != v.shape[0]:
        raise ValueError("compute_dvdt expects 1D aligned t and v")
    if t.shape[0] < 2:
        return jnp.full(t.shape, jnp.nan)

    v_used = v
    if filter is not None:
        kernel = jnp.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=jnp.float32)
        kernel = kernel / jnp.sum(kernel)
        v_used = jnp.convolve(v.astype(jnp.float32), kernel, mode="same")

    dv = jnp.diff(v_used)
    dt = jnp.diff(t)
    short = jnp.where(dt != 0.0, 1e-3 * dv / dt, jnp.nan)
    out = jnp.full(t.shape, jnp.nan)
    return out.at[1:].set(short)


def _first_index_at_or_after(t: Array, t0: float | Array) -> Array:
    mask = t >= t0
    idx = jnp.argmax(mask)
    return jnp.where(jnp.any(mask), idx, t.shape[0] - 1).astype(jnp.int32)


def _compact(values: Array, valid: Array) -> tuple[Array, Array]:
    n = values.shape[0]
    order = jnp.argsort(jnp.where(valid, jnp.arange(n), n), stable=True)
    sorted_vals = values[order]
    n_valid = jnp.sum(valid)
    sorted_valid = jnp.arange(n) < n_valid
    sorted_vals = jnp.where(sorted_valid, sorted_vals, -1)
    return sorted_vals.astype(jnp.int32), sorted_valid


def _masked_take(values: Array, idx: Array, valid: Array, fill: float) -> Array:
    safe_idx = jnp.where(valid, idx, 0)
    return jnp.where(valid, values[safe_idx], fill)


def _indices_to_mask(length: int, idx: Array) -> Array:
    x = jnp.arange(length)
    return jnp.any(x[:, None] == idx[None, :], axis=1)


def _next_valid_or_default(values: Array, valid: Array, default: Array) -> Array:
    shifted_values = jnp.concatenate(
        [values[1:], jnp.asarray([default], dtype=values.dtype)]
    )
    shifted_valid = jnp.concatenate([valid[1:], jnp.asarray([False])])
    return jnp.where(shifted_valid, shifted_values, default)


def detect_events(
    t: Array,
    v: Array,
    i: Array,
    t0: float | Array,
    t1: float | Array,
    *,
    dvdt: Optional[Array] = None,
    max_spikes: int = 100,
    dvdt_filter: float | None = 10.0,
    threshold_cutoff: float = 20.0,
    peak_cutoff: float = -30.0,
    trough_cutoff: float = 2.0,
    **kwargs,
) -> Events:
    del kwargs

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

    dvdt = compute_dvdt(t, v, filter=dvdt_filter) if dvdt is None else dvdt

    ge = dvdt >= threshold_cutoff
    crossing_pos = jnp.arange(T - 1, dtype=jnp.int32)
    crossing_mask = (~ge[:-1]) & ge[1:]
    crossing_in_window = (
        crossing_mask
        & (crossing_pos >= start_idx)
        & (crossing_pos <= end_idx)
        & (start_idx < end_idx)
    )
    crossing_overflow = jnp.sum(crossing_in_window) > max_spikes

    crossings = jnp.where((~ge[:-1]) & ge[1:], size=max_spikes, fill_value=-1)[0]
    in_window = (
        (crossings >= start_idx) & (crossings <= end_idx) & (start_idx < end_idx)
    )
    spike_idx = jnp.where(in_window, crossings, -1).astype(jnp.int32)
    spike_valid = spike_idx >= 0

    next_spike = _next_valid_or_default(spike_idx, spike_valid, end_idx)

    def peak_for(k: Array) -> Array:
        s = spike_idx[k]
        n = next_spike[k]
        valid = spike_valid[k] & (n > s)
        m = valid & (x >= s) & (x < n)
        vals = jnp.where(m, v, -jnp.inf)
        return jnp.where(valid, jnp.argmax(vals), -1)

    peak_idx = jax.vmap(peak_for)(jnp.arange(max_spikes)).astype(jnp.int32)
    peak_valid = peak_idx >= 0

    peak_v = _masked_take(v, peak_idx, peak_valid, fill=-jnp.inf)
    spike_v = _masked_take(v, spike_idx, spike_valid, fill=jnp.inf)
    keep = spike_valid & peak_valid
    keep = keep & (peak_v >= peak_cutoff)
    keep = keep & ((peak_v - spike_v) >= trough_cutoff)
    spike_idx, spike_valid = _compact(spike_idx, keep)
    peak_idx, peak_valid = _compact(peak_idx, keep)

    def upstroke_for(k: Array) -> Array:
        s = spike_idx[k]
        p = peak_idx[k]
        valid = spike_valid[k] & peak_valid[k] & (p > s)
        m = valid & (x >= s) & (x < p)
        vals = jnp.where(m, dvdt, -jnp.inf)
        return jnp.where(valid, jnp.argmax(vals), -1)

    upstroke_idx = jax.vmap(upstroke_for)(jnp.arange(max_spikes)).astype(jnp.int32)
    up_valid = upstroke_idx >= 0
    up_vals = _masked_take(dvdt, upstroke_idx, up_valid, fill=jnp.nan)
    avg_up = jnp.nanmean(up_vals)
    target = avg_up * 0.05

    prev_up = jnp.concatenate([jnp.asarray([0], dtype=jnp.int32), upstroke_idx[:-1]])
    xr = x[::-1]

    def threshold_for(k: Array) -> Array:
        u = upstroke_idx[k]
        p = prev_up[k]
        valid = up_valid[k] & (u > p)
        m = valid & (xr <= u) & (xr > p)
        cand = jnp.where(m & (dvdt[xr] <= target), xr, -1)
        best = jnp.max(cand)
        return jnp.where(best >= 0, best, p)

    threshold_idx = jax.vmap(threshold_for)(jnp.arange(max_spikes)).astype(jnp.int32)
    threshold_valid = spike_valid

    peak_t = _masked_take(t, peak_idx, peak_valid, fill=jnp.inf)
    threshold_t = _masked_take(t, threshold_idx, threshold_valid, fill=-jnp.inf)
    pair_ok = threshold_valid & peak_valid & ((peak_t - threshold_t) < 0.005)
    threshold_idx, threshold_valid = _compact(threshold_idx, pair_ok)
    peak_idx, peak_valid = _compact(peak_idx, pair_ok)

    next_thr = _next_valid_or_default(threshold_idx, threshold_valid, end_idx)

    def trough_for(k: Array) -> Array:
        p = peak_idx[k]
        n = next_thr[k]
        valid = peak_valid[k] & (n > p)
        m = valid & (x >= p) & (x < n)
        vals = jnp.where(m, v, jnp.inf)
        return jnp.where(valid, jnp.argmin(vals), -1)

    trough_idx = jax.vmap(trough_for)(jnp.arange(max_spikes)).astype(jnp.int32)
    trough_valid = trough_idx >= 0

    threshold_idx = jnp.where(threshold_valid, threshold_idx, -1)
    peak_idx = jnp.where(peak_valid, peak_idx, -1)
    trough_idx = jnp.where(trough_valid, trough_idx, -1)

    threshold_mask = _indices_to_mask(T, threshold_idx)
    peak_mask = _indices_to_mask(T, peak_idx)
    trough_mask = _indices_to_mask(T, trough_idx)

    # assign spike labels by latest threshold
    th = jnp.where(threshold_valid, threshold_idx, -1)
    eligible = (th[None, :] <= x[:, None]) & threshold_valid[None, :]
    th_vals = jnp.where(eligible, th[None, :], -1)
    k_latest = jnp.argmax(th_vals, axis=1).astype(jnp.int32)
    has = jnp.any(eligible, axis=1)
    spike_labels = jnp.where(has, k_latest, -1)

    in_window_t = (x >= start_idx) & (x <= end_idx)
    threshold_mask = threshold_mask & in_window_t
    peak_mask = peak_mask & in_window_t
    trough_mask = trough_mask & in_window_t
    spike_labels = jnp.where(in_window_t, spike_labels, -1)

    events = Events.from_masks(
        spikes=spike_labels,
        thresholds=threshold_mask,
        peaks=peak_mask,
        troughs=trough_mask,
        max_spikes=max_spikes,
    )
    events.overflow = events.overflow | crossing_overflow
    return events
