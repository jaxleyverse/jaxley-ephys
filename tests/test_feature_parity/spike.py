from __future__ import annotations

import numpy as np
import pytest

from jaxley_ephys.features import ISI_TYPE_DETOUR, ISI_TYPE_DIRECT, ISI_TYPE_MISSING, compute_spike_features
from tests.utils.jax_feature_contract import SPIKE_FEATURE_NAMES, assert_feature_equal


def _long_spiking_sweep(specimen_sweepset_extractors):
    ext = specimen_sweepset_extractors.get("long_square")
    if ext is None:
        pytest.skip("no long-square sweeps available")
    spiking = [sweep for sweep in ext.sweeps() if sweep.sweep_feature("avg_rate") > 0]
    if not spiking:
        pytest.skip("no spiking long-square sweep available")
    return spiking[0]


def _pad_expected_spike_feature(feature_name: str, expected, max_spikes: int):
    if feature_name.endswith("_index"):
        out = np.full(max_spikes, -1, dtype=int)
        arr = np.asarray(expected)
        finite = np.isfinite(arr)
        vals = arr[finite].astype(int)
        out[: len(vals)] = vals[:max_spikes]
        return out

    if feature_name == "isi_type":
        mapping = {
            "direct": ISI_TYPE_DIRECT,
            "detour": ISI_TYPE_DETOUR,
        }
        out = np.full(max_spikes, ISI_TYPE_MISSING, dtype=int)
        arr = np.asarray(expected, dtype=object)
        vals = []
        for value in arr.tolist():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                vals.append(ISI_TYPE_MISSING)
            else:
                vals.append(mapping.get(str(value), ISI_TYPE_MISSING))
        out[: len(vals)] = np.asarray(vals[:max_spikes], dtype=int)
        return out

    out = np.full(max_spikes, np.nan, dtype=float)
    vals = np.asarray(expected, dtype=float)
    out[: len(vals)] = vals[:max_spikes]
    return out


@pytest.mark.parametrize("feature_name", SPIKE_FEATURE_NAMES)
def test_jax_spike_feature_matches_vendored_allen(
    specimen_sweepset_extractors,
    feature_name: str,
):
    sweep = _long_spiking_sweep(specimen_sweepset_extractors)
    expected = sweep.spike_feature(feature_name, include_clipped=True)

    max_spikes = 256
    computed_map = compute_spike_features(
        sweep.t,
        sweep.v,
        sweep.i,
        float(sweep.start),
        float(sweep.end),
        max_spikes=max_spikes,
    )

    assert feature_name in computed_map
    expected_padded = _pad_expected_spike_feature(feature_name, expected, max_spikes)
    assert_feature_equal(expected_padded, computed_map[feature_name])
