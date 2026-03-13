from __future__ import annotations

from jax import Array


def compute_sweepset_features(
    t_set,
    v_set,
    i_set,
    start_set,
    end_set,
    *,
    max_spikes: int = 256,
    dvdt_filter: float | None = 10.0,
) -> dict[str, dict[str, Array]]:
    del t_set, v_set, i_set, start_set, end_set, max_spikes, dvdt_filter
    raise NotImplementedError(
        "compute_sweepset_features is temporarily disabled and will be reworked"
    )
