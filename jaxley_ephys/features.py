from __future__ import annotations

from collections.abc import Sequence

from jax import Array


def compute_spike_features(
    t: Array,
    v: Array,
    i: Array,
    start: float,
    end: float,
) -> dict[str, Array]:
    raise NotImplementedError("compute_spike_features is not implemented yet")


def compute_sweep_features(
    t: Array,
    v: Array,
    i: Array,
    start: float,
    end: float,
) -> dict[str, Array]:
    raise NotImplementedError("compute_sweep_features is not implemented yet")


def compute_sweepset_features(
    t_set: Sequence[Array],
    v_set: Sequence[Array],
    i_set: Sequence[Array],
    start_set: Sequence[float],
    end_set: Sequence[float],
) -> dict[str, dict[str, Array]]:
    raise NotImplementedError("compute_sweepset_features is not implemented yet")
