from __future__ import annotations

import pytest

from jaxley_ephys.features import (
    compute_spike_features,
    compute_sweep_features,
    compute_sweepset_features,
)
from tests.utils.jax_feature_contract import (
    SPIKE_FEATURE_NAMES,
    SWEEP_FEATURE_NAMES,
    assert_feature_equal,
)


def _long_spiking_sweep(specimen_case):
    sweep = specimen_case.representative_sweep("long_spiking")
    sweep.process_spikes()
    return sweep


def _long_square_sweepset(specimen_case):
    records = specimen_case.records(kind="long_square")
    sweepset = specimen_case.make_sweep_set(records)
    sweepset.process_spikes()
    return sweepset, records


@pytest.mark.parametrize("feature_name", SPIKE_FEATURE_NAMES)
def test_jax_spike_feature_matches_vendored_allen(specimen_case, feature_name: str):
    sweep = _long_spiking_sweep(specimen_case)
    expected = sweep.spike_feature(feature_name, include_clipped=True)

    computed_map = compute_spike_features(
        sweep.t,
        sweep.v,
        sweep.i,
        float(sweep.start),
        float(sweep.end),
    )

    assert feature_name in computed_map
    assert_feature_equal(expected, computed_map[feature_name])


@pytest.mark.parametrize("feature_name", SWEEP_FEATURE_NAMES)
def test_jax_sweep_feature_matches_vendored_allen(specimen_case, feature_name: str):
    sweep = _long_spiking_sweep(specimen_case)
    expected = sweep.sweep_feature(feature_name, allow_missing=True)

    computed_map = compute_sweep_features(
        sweep.t,
        sweep.v,
        sweep.i,
        float(sweep.start),
        float(sweep.end),
    )

    assert feature_name in computed_map
    assert_feature_equal(expected, computed_map[feature_name])


@pytest.mark.parametrize("feature_name", SWEEP_FEATURE_NAMES)
def test_jax_sweepset_sweep_feature_matches_vendored_allen(
    specimen_case, feature_name: str
):
    sweepset, records = _long_square_sweepset(specimen_case)
    expected = sweepset.sweep_features(feature_name, allow_missing=True)

    t_set = []
    v_set = []
    i_set = []
    start_set = []
    end_set = []
    for record in records:
        t, v, i, start, end = specimen_case.trace_for(record.sweep_number)
        t_set.append(t)
        v_set.append(v)
        i_set.append(i)
        start_set.append(start)
        end_set.append(end)

    computed = compute_sweepset_features(t_set, v_set, i_set, start_set, end_set)
    assert "sweep_features" in computed
    assert feature_name in computed["sweep_features"]
    assert_feature_equal(expected, computed["sweep_features"][feature_name])


@pytest.mark.parametrize("feature_name", SPIKE_FEATURE_NAMES)
def test_jax_sweepset_spike_feature_average_matches_vendored_allen(
    specimen_case, feature_name: str
):
    sweepset, records = _long_square_sweepset(specimen_case)
    expected = sweepset.spike_feature_averages(feature_name)

    t_set = []
    v_set = []
    i_set = []
    start_set = []
    end_set = []
    for record in records:
        t, v, i, start, end = specimen_case.trace_for(record.sweep_number)
        t_set.append(t)
        v_set.append(v)
        i_set.append(i)
        start_set.append(start)
        end_set.append(end)

    computed = compute_sweepset_features(t_set, v_set, i_set, start_set, end_set)
    assert "spike_feature_averages" in computed
    assert feature_name in computed["spike_feature_averages"]
    assert_feature_equal(expected, computed["spike_feature_averages"][feature_name])
