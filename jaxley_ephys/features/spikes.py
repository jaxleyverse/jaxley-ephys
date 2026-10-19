from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from jaxley_ephys.events import Events, compute_dvdt, detect_events

ISI_TYPE_MISSING = -1
ISI_TYPE_DIRECT = 0
ISI_TYPE_DETOUR = 1


def _project(values: Array, idx: Array, valid: Array) -> Array:
    safe_idx = jnp.where(valid, idx, 0)
    return jnp.where(valid, values[safe_idx], jnp.nan)


def _valid_from_index(idx: Array) -> Array:
    return idx >= 0


def _trough_valid(ev: Events) -> Array:
    return ev.trough_idxs >= 0


def _downstroke_valid(ev: Events) -> Array:
    return ev.downstroke_idxs >= 0


def _fast_trough_index_from_events(ev: Events, dvdt: Array) -> tuple[Array, Array]:
    t_len = dvdt.shape[0]
    x = jnp.arange(t_len)
    end_idx = jnp.max(jnp.where(ev.window, x, 0))
    usable = ev.spike_valid & (~ev.clipped)
    order = jnp.arange(ev.threshold_idxs.shape[0])
    next_thr = jnp.min(
        jnp.where(
            (order[None, :] > order[:, None]) & usable[None, :],
            ev.threshold_idxs[None, :],
            end_idx,
        ),
        axis=1,
    )

    def one_fast(k: Array) -> Array:
        peak = ev.peak_idxs[k]
        nxt = next_thr[k]
        valid = usable[k] & (peak >= 0) & (nxt > peak)
        # Allen's trough analysis finds its own downstroke over the entire
        # peak-to-next-unclipped-threshold interval. The event downstroke only
        # searches as far as the minimum-voltage trough and is not interchangeable.
        down = jnp.argmin(jnp.where(valid & (x >= peak) & (x < nxt), dvdt, jnp.inf))
        target = 0.01 * dvdt[jnp.maximum(down, 0)]
        m = valid & (x >= down) & (x < nxt)
        cand = jnp.where(m & (dvdt >= target), x, t_len + 1)
        term = jnp.min(cand)
        return jnp.where(term < t_len, term, -1)

    idx = jnp.asarray([one_fast(k) for k in range(ev.threshold_idxs.shape[0])])
    valid = idx >= 0
    return idx, valid


def _width_feature(
    t: Array,
    v: Array,
    ev: Events,
    trough_idx: Array,
    trough_valid: Array,
) -> Array:
    t_len = t.shape[0]
    x = jnp.arange(t_len)

    def one_width(k: Array) -> Array:
        valid = ev.spike_valid[k] & trough_valid[k] & (~ev.clipped[k])
        spk = ev.threshold_idxs[k]
        pk = ev.peak_idxs[k]
        tr = trough_idx[k]

        tr_safe = jnp.maximum(tr, 0)
        spk_safe = jnp.maximum(spk, 0)
        pk_safe = jnp.maximum(pk, 0)

        height = v[pk_safe] - v[tr_safe]
        width_level = v[tr_safe] + height / 2.0
        thresh_to_peak_level = v[spk_safe] + (v[pk_safe] - v[spk_safe]) / 2.0
        width_level = jnp.where(
            width_level < v[spk_safe], thresh_to_peak_level, width_level
        )

        left_mask = valid & (x <= pk_safe) & (x > spk_safe) & (v <= width_level)
        start_idx = jnp.max(jnp.where(left_mask, x, -1))

        right_mask = valid & (x >= pk_safe) & (x < tr_safe) & (v <= width_level)
        end_idx = jnp.min(jnp.where(right_mask, x, t_len + 1))

        ok = valid & (start_idx >= 0) & (end_idx < t_len)
        start_safe = jnp.where(start_idx >= 0, start_idx, 0)
        end_safe = jnp.where(end_idx < t_len, end_idx, 0)
        return jnp.where(ok, t[end_safe] - t[start_safe], jnp.nan)

    return jnp.asarray([one_width(k) for k in range(ev.threshold_idxs.shape[0])])


def threshold_index(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i, dvdt
    return jnp.where(ev.spike_valid, ev.threshold_idxs, -1)


def peak_index(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i, dvdt
    return jnp.where(ev.spike_valid, ev.peak_idxs, -1)


def trough_index(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i, dvdt
    return jnp.where(_trough_valid(ev), ev.trough_idxs, -1)


def upstroke_index(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i, dvdt
    valid = _valid_from_index(ev.upstroke_idxs)
    return jnp.where(valid, ev.upstroke_idxs, -1)


def downstroke_index(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i, dvdt
    return jnp.where(_downstroke_valid(ev), ev.downstroke_idxs, -1)


def clipped(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i, dvdt
    return jnp.where(ev.spike_valid, ev.clipped.astype(float), jnp.nan)


def threshold_t(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del v, i, dvdt
    return _project(t, ev.threshold_idxs, ev.spike_valid)


def threshold_v(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, i, dvdt
    return _project(v, ev.threshold_idxs, ev.spike_valid)


def threshold_i(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, dvdt
    return _project(i, ev.threshold_idxs, ev.spike_valid)


def peak_t(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del v, i, dvdt
    return _project(t, ev.peak_idxs, ev.spike_valid)


def peak_v(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, i, dvdt
    return _project(v, ev.peak_idxs, ev.spike_valid)


def peak_i(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, dvdt
    return _project(i, ev.peak_idxs, ev.spike_valid)


def trough_t(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del v, i, dvdt
    return _project(t, ev.trough_idxs, _trough_valid(ev))


def trough_v(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, i, dvdt
    return _project(v, ev.trough_idxs, _trough_valid(ev))


def trough_i(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, dvdt
    return _project(i, ev.trough_idxs, _trough_valid(ev))


def upstroke(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i
    valid = _valid_from_index(ev.upstroke_idxs)
    return _project(dvdt, ev.upstroke_idxs, valid)


def upstroke_t(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del v, i, dvdt
    valid = _valid_from_index(ev.upstroke_idxs)
    return _project(t, ev.upstroke_idxs, valid)


def upstroke_v(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, i, dvdt
    valid = _valid_from_index(ev.upstroke_idxs)
    return _project(v, ev.upstroke_idxs, valid)


def downstroke(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i
    return _project(dvdt, ev.downstroke_idxs, _downstroke_valid(ev))


def downstroke_t(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del v, i, dvdt
    return _project(t, ev.downstroke_idxs, _downstroke_valid(ev))


def downstroke_v(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, i, dvdt
    return _project(v, ev.downstroke_idxs, _downstroke_valid(ev))


def isi_type(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i, dvdt
    return jnp.where(ev.spike_valid & (~ev.clipped), ISI_TYPE_DIRECT, ISI_TYPE_MISSING)


def fast_trough_index(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i
    idx, _ = _fast_trough_index_from_events(ev, dvdt)
    return idx


def adp_index(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i, dvdt
    return jnp.full(ev.threshold_idxs.shape, -1)


def slow_trough_index(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i, dvdt
    return jnp.full(ev.threshold_idxs.shape, -1)


def _all_nan_like_spikes(ev: Events) -> Array:
    return jnp.full(ev.threshold_idxs.shape, jnp.nan)


def fast_trough_t(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del v, i
    idx, _ = _fast_trough_index_from_events(ev, dvdt)
    valid = idx >= 0
    return _project(t, idx, valid)


def fast_trough_v(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, i
    idx, _ = _fast_trough_index_from_events(ev, dvdt)
    valid = idx >= 0
    return _project(v, idx, valid)


def fast_trough_i(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v
    idx, _ = _fast_trough_index_from_events(ev, dvdt)
    valid = idx >= 0
    return _project(i, idx, valid)


def adp_t(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i, dvdt
    return _all_nan_like_spikes(ev)


def adp_v(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i, dvdt
    return _all_nan_like_spikes(ev)


def adp_i(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i, dvdt
    return _all_nan_like_spikes(ev)


def slow_trough_t(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i, dvdt
    return _all_nan_like_spikes(ev)


def slow_trough_v(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i, dvdt
    return _all_nan_like_spikes(ev)


def slow_trough_i(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del t, v, i, dvdt
    return _all_nan_like_spikes(ev)


def width(t: Array, v: Array, i: Array, ev: Events, dvdt: Array) -> Array:
    del i
    fast_idx, fast_valid = _fast_trough_index_from_events(ev, dvdt)
    return _width_feature(t, v, ev, fast_idx, fast_valid)


def upstroke_downstroke_ratio(
    t: Array,
    v: Array,
    i: Array,
    ev: Events,
    dvdt: Array,
) -> Array:
    del t, v, i
    up = upstroke(dvdt, dvdt, dvdt, ev, dvdt)
    down = downstroke(dvdt, dvdt, dvdt, ev, dvdt)
    valid = jnp.isfinite(up) & jnp.isfinite(down) & (down != 0)
    return jnp.where(valid, up / -down, jnp.nan)


SPIKE_FEATURE_FUNCTIONS: dict[
    str, Callable[[Array, Array, Array, Events, Array], Array]
] = {
    "threshold_index": threshold_index,
    "clipped": clipped,
    "threshold_t": threshold_t,
    "threshold_v": threshold_v,
    "threshold_i": threshold_i,
    "peak_index": peak_index,
    "peak_t": peak_t,
    "peak_v": peak_v,
    "peak_i": peak_i,
    "trough_index": trough_index,
    "trough_t": trough_t,
    "trough_v": trough_v,
    "trough_i": trough_i,
    "upstroke_index": upstroke_index,
    "upstroke": upstroke,
    "upstroke_t": upstroke_t,
    "upstroke_v": upstroke_v,
    "downstroke_index": downstroke_index,
    "downstroke": downstroke,
    "downstroke_t": downstroke_t,
    "downstroke_v": downstroke_v,
    "isi_type": isi_type,
    "fast_trough_index": fast_trough_index,
    "fast_trough_t": fast_trough_t,
    "fast_trough_v": fast_trough_v,
    "fast_trough_i": fast_trough_i,
    "adp_index": adp_index,
    "adp_t": adp_t,
    "adp_v": adp_v,
    "adp_i": adp_i,
    "slow_trough_index": slow_trough_index,
    "slow_trough_t": slow_trough_t,
    "slow_trough_v": slow_trough_v,
    "slow_trough_i": slow_trough_i,
    "width": width,
    "upstroke_downstroke_ratio": upstroke_downstroke_ratio,
}


@partial(jax.jit, static_argnames=("filter",))
def _spike_derivative(t: Array, v: Array, *, filter: float | None) -> Array:
    return compute_dvdt(t, v, filter=filter)


@partial(jax.jit, static_argnames=("max_spikes", "feature_names"))
def _compute_spike_features(
    t: Array,
    v: Array,
    i: Array,
    start: Array,
    end: Array,
    dvdt: Array,
    *,
    max_spikes: int,
    feature_names: tuple[str, ...],
) -> tuple[Array, dict[str, Array]]:
    ev = detect_events(
        t,
        v,
        i,
        start,
        end,
        max_spikes=max_spikes,
        dvdt=dvdt,
    )

    out: dict[str, Array] = {}
    for key in feature_names:
        if key not in SPIKE_FEATURE_FUNCTIONS:
            raise KeyError(f"unknown spike feature '{key}'")
        fn = SPIKE_FEATURE_FUNCTIONS[key]
        value = fn(t, v, i, ev, dvdt)
        if value.shape != (max_spikes,):
            raise ValueError(
                f"spike feature '{key}' returned shape {value.shape}, "
                f"expected {(max_spikes,)}"
            )
        out[key] = value
    return ev.overflow, out


def compute_spike_features(
    t: Array,
    v: Array,
    i: Array,
    start: Array,
    end: Array,
    *,
    max_spikes: int | None = 256,
    dvdt_filter: float | None = 10.0,
    feature_names: tuple[str, ...] | None = None,
) -> dict[str, Array]:
    """Extract selected spike features from aligned ``t/v/i[T]`` arrays.

    Time is in seconds, voltage in mV. Continuous features are NaN-padded;
    indices use -1. Filtering/detection are shared across features and compiled
    internally. Eager calls raise on capacity overflow rather than truncate.

    ``max_spikes=None`` sizes the output from all putative derivative crossings
    (at least 8 slots, rounded up to a power of two). This convenience mode is
    eager-only; use an explicit static capacity inside ``jit``/``vmap``.
    """
    dvdt = _spike_derivative(t, v, filter=dvdt_filter)
    if max_spikes is None:
        above = dvdt >= 20.0
        count = int(jnp.sum(~above[:-1] & above[1:]))
        max_spikes = max(8, 1 << max(0, count - 1).bit_length())
    names = (
        tuple(SPIKE_FEATURE_FUNCTIONS)
        if feature_names is None
        else tuple(feature_names)
    )
    overflow, out = _compute_spike_features(
        t, v, i, start, end, dvdt, max_spikes=max_spikes, feature_names=names
    )
    if not isinstance(overflow, jax.core.Tracer) and bool(overflow):
        raise ValueError(
            f"detect_events overflowed max_spikes={max_spikes}; "
            "increase max_spikes or use max_spikes=None"
        )
    return out
