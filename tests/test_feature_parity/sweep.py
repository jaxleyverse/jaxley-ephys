from __future__ import annotations

import pytest

from jaxley_ephys.base import compute_dvdt, detect_events
from jaxley_ephys.features import compute_sweep_features
from tests.utils.jax_feature_contract import SWEEP_PARITY_FEATURE_NAMES, assert_feature_equal


def _long_spiking_sweep(specimen_case):
    sweep = specimen_case.representative_sweep("long_spiking")
    sweep.process_spikes()
    return sweep


@pytest.mark.parametrize("feature_name", SWEEP_PARITY_FEATURE_NAMES)
def test_jax_sweep_feature_matches_vendored_allen(specimen_case, feature_name: str):
    sweep = _long_spiking_sweep(specimen_case)
    expected = sweep.sweep_feature(feature_name, allow_missing=True)
    ev = detect_events(
        sweep.t,
        sweep.v,
        sweep.i,
        float(sweep.start),
        float(sweep.end),
        max_spikes=256,
    )
    dvdt = compute_dvdt(sweep.t, sweep.v)

    computed_map = compute_sweep_features(
        sweep.t,
        sweep.v,
        sweep.i,
        ev,
        dvdt,
    )

    assert feature_name in computed_map
    assert_feature_equal(expected, computed_map[feature_name])
