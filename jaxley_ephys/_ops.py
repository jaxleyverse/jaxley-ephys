"""O(T) array primitives used by spike detection and feature extraction.

Spike-level quantities live in fixed-size buffers of length K (``max_spikes``):
valid entries are packed at the front, missing entries are -1 (indices) or NaN
(values). Allen searches a separate time window per spike (e.g. from a spike's
peak to the next threshold). Here such windows are disjoint and ordered, so
each sample is labelled with the window containing it ("segments") and all
spikes are handled in one pass over the trace: O(T + K) instead of O(T * K).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array, lax


def take(values: Array, idx: Array, fill: float = jnp.nan) -> Array:
    """``values[idx]`` where ``idx >= 0``, ``fill`` elsewhere."""
    valid = idx >= 0
    return jnp.where(valid, values[jnp.where(valid, idx, 0)], fill)


def compact(keep: Array, *arrays: Array) -> tuple[Array, ...]:
    """Stably move entries with ``keep`` to the front and fill the rest with -1."""
    order = jnp.argsort(~keep, stable=True)
    valid = jnp.arange(keep.shape[0]) < jnp.sum(keep)
    return tuple(jnp.where(valid, a[order], -1) for a in arrays)


def shift_left(values: Array, fill: float | bool) -> Array:
    """``values[k + 1]`` with ``fill`` for the last entry."""
    return jnp.concatenate([values[1:], jnp.asarray([fill], dtype=values.dtype)])


def shift_right(values: Array, fill: float | bool) -> Array:
    """``values[k - 1]`` with ``fill`` for the first entry."""
    return jnp.concatenate([jnp.asarray([fill], dtype=values.dtype), values[:-1]])


def cummax(x: Array, reverse: bool = False) -> Array:
    """Running maximum of an integer array.

    On CPU an unrolled sequential scan is ~5x faster than ``lax.cummax`` (a
    parallel prefix scan, which is the better choice on GPU/TPU).
    """
    if jax.default_backend() != "cpu":
        return lax.cummax(x, reverse=reverse)
    init = jnp.full(x.shape[1:], jnp.iinfo(x.dtype).min, x.dtype)
    return lax.scan(
        lambda c, a: (jnp.maximum(c, a),) * 2, init, x, reverse=reverse, unroll=8
    )[1]


def next_true(cond: Array) -> Array:
    """For every sample, the first index >= it where ``cond`` holds (T if none)."""
    size = cond.shape[0]
    return -cummax(jnp.where(cond, -jnp.arange(size), -size), reverse=True)


def prev_true(cond: Array) -> Array:
    """For every sample, the last index <= it where ``cond`` holds (-1 if none)."""
    return cummax(jnp.where(cond, jnp.arange(cond.shape[0]), -1))


def segments(lo: Array, hi: Array, size: int) -> Array:
    """Label samples with the index k of the window ``[lo[k], hi[k])`` containing them.

    Windows must be disjoint and ordered by k. Empty or invalid windows
    (``lo < 0``) are ignored. Samples outside all windows get -1.
    """
    k = jnp.arange(lo.shape[0])
    valid = (lo >= 0) & (lo < hi)
    starts = jnp.full(size, -1).at[jnp.where(valid, lo, size)].max(k, mode="drop")
    seg = cummax(starts)
    inside = (seg >= 0) & (jnp.arange(size) < hi[jnp.maximum(seg, 0)])
    return jnp.where(inside, seg, -1)


def _segment_ids(seg: Array, num: int) -> Array:
    return jnp.where(seg >= 0, seg, num)  # out-of-range ids are dropped by jax.ops


def segment_first(cond: Array, seg: Array, num: int) -> Array:
    """First index in each segment where ``cond`` holds (-1 if none)."""
    size = cond.shape[0]
    first = jax.ops.segment_min(
        jnp.where(cond, jnp.arange(size), size), _segment_ids(seg, num), num
    )
    return jnp.where(first < size, first, -1)


def segment_last(cond: Array, seg: Array, num: int) -> Array:
    """Last index in each segment where ``cond`` holds (-1 if none)."""
    last = jax.ops.segment_max(
        jnp.where(cond, jnp.arange(cond.shape[0]), -1), _segment_ids(seg, num), num
    )
    return jnp.maximum(last, -1)


def segment_argmax(values: Array, seg: Array, num: int) -> Array:
    """Index of the first maximum of ``values`` in each segment (-1 if empty).

    NaN values (dV/dt where ``dt == 0``) are ignored.
    """
    values = jnp.where(jnp.isnan(values), -jnp.inf, values)
    ids = _segment_ids(seg, num)
    best = jax.ops.segment_max(values, ids, num)
    is_best = values == jnp.concatenate([best, jnp.asarray([jnp.nan])])[ids]
    return segment_first(is_best, seg, num)


def segment_argmin(values: Array, seg: Array, num: int) -> Array:
    """Index of the first minimum of ``values`` in each segment (-1 if empty)."""
    return segment_argmax(-values, seg, num)


def window_argmax(values: Array, lo: Array, hi: Array) -> Array:
    """Index of the first maximum of ``values[lo:hi]`` (-1 if empty)."""
    x = jnp.arange(values.shape[0])
    inside = (x >= lo) & (x < hi)
    idx = jnp.argmax(jnp.where(inside, values, -jnp.inf))
    return jnp.where(jnp.any(inside), idx, -1)


def masked_mean(values: Array, mask: Array) -> Array:
    """Mean of ``values`` where ``mask`` (NaN if empty)."""
    n = jnp.sum(mask)
    total = jnp.sum(jnp.where(mask, values, 0.0))
    return jnp.where(n > 0, total / jnp.maximum(n, 1), jnp.nan)
