from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from jaxley_ephys.events import compute_dvdt, detect_events
from jaxley_ephys.features import (
    ISI_TYPE_DETOUR,
    ISI_TYPE_DIRECT,
    ISI_TYPE_MISSING,
    SPIKE_FEATURE_FUNCTIONS,
)
from tests.utils.jax_feature_contract import SPIKE_FEATURE_NAMES, assert_feature_equal


def _long_spiking_sweep(specimen_sweepset_extractors):
    ext = specimen_sweepset_extractors.get("long_square")
    if ext is None:
        return None
    spiking = [sweep for sweep in ext.sweeps() if sweep.sweep_feature("avg_rate") > 0]
    if not spiking:
        return None
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


@pytest.fixture(scope="module")
def long_spiking_sweep_primitives(specimen_sweepset_extractors):
    sweep = _long_spiking_sweep(specimen_sweepset_extractors)
    if sweep is None:
        pytest.skip("No long-spiking sweep found for spike parity checks.")

    max_spikes = 256
    dvdt_filter = 10.0
    ev = detect_events(
        sweep.t,
        sweep.v,
        sweep.i,
        jnp.asarray(float(sweep.start)),
        jnp.asarray(float(sweep.end)),
        max_spikes=max_spikes,
        dvdt_filter=dvdt_filter,
    )
    dvdt = compute_dvdt(sweep.t, sweep.v, filter=dvdt_filter)
    return sweep, ev, dvdt, max_spikes


@pytest.mark.parametrize("feature_name", SPIKE_FEATURE_NAMES)
def test_jax_spike_feature_matches_vendored_allen(
    long_spiking_sweep_primitives,
    feature_name: str,
):
    sweep, ev, dvdt, max_spikes = long_spiking_sweep_primitives
    expected = sweep.spike_feature(feature_name, include_clipped=True)

    computed = SPIKE_FEATURE_FUNCTIONS[feature_name](
        sweep.t,
        sweep.v,
        sweep.i,
        ev,
        dvdt,
    )

    assert computed.shape == (max_spikes,)
    # TODO: move if here:
    # if feature_name == "isi_type":
    #     expected = _map_isi_type(expected, max_spikes)
    # if feature_name.endswith("_index") or feature_name == "isi_type":
    #   #     expected =_pad_max_spikes(expected, max_spikes)
    expected_padded = _pad_expected_spike_feature(feature_name, expected, max_spikes)
    assert_feature_equal(expected_padded, computed)
