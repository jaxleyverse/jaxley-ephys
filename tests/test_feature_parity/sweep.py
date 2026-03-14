from __future__ import annotations

import pytest

from jaxley_ephys.base import compute_dvdt, detect_events
from jaxley_ephys.features import compute_sweep_features
from tests.utils.jax_feature_contract import SWEEP_PARITY_FEATURE_NAMES, assert_feature_equal


SWEEP_EXTRA_PARITY_FEATURE_NAMES = ("peak_deflect", "peak_deflect_index")
SWEEP_DIRECT_PARITY_FEATURE_NAMES = (
    SWEEP_PARITY_FEATURE_NAMES + SWEEP_EXTRA_PARITY_FEATURE_NAMES
)


def _long_spiking_sweep(specimen_sweepset_extractors):
    ext = specimen_sweepset_extractors.get("long_square")
    if ext is None:
        pytest.skip("no long-square sweeps available")
    spiking = [sweep for sweep in ext.sweeps() if sweep.sweep_feature("avg_rate") > 0]
    if not spiking:
        pytest.skip("no spiking long-square sweep available")
    return spiking[0]


@pytest.mark.parametrize("feature_name", SWEEP_DIRECT_PARITY_FEATURE_NAMES)
def test_jax_sweep_feature_matches_vendored_allen(
    specimen_sweepset_extractors,
    feature_name: str,
):
    sweep = _long_spiking_sweep(specimen_sweepset_extractors)
    if feature_name == "peak_deflect":
        expected = sweep.sweep_feature("peak_deflect", allow_missing=True)[0]
    elif feature_name == "peak_deflect_index":
        expected = sweep.sweep_feature("peak_deflect", allow_missing=True)[1]
    else:
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
