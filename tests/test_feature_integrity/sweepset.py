from __future__ import annotations

import pytest

from jaxley_ephys.features import compute_sweepset_features
from tests.utils.simulate_testdata import simulate_sweepset


def test_jax_sweepset_features_are_temporarily_disabled():
    t_set, v_set, i_set = simulate_sweepset(
        num_sweeps=4,
        dt=1e-4,
        t_max=0.12,
        i_range=(0.25, 0.9),
    )

    start_set = [0.02] * int(t_set.shape[0])
    end_set = [0.1] * int(t_set.shape[0])

    with pytest.raises(NotImplementedError, match="temporarily disabled"):
        compute_sweepset_features(t_set, v_set, i_set, start_set, end_set)
