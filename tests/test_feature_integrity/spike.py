from __future__ import annotations

import jax
import pytest

from jaxley_ephys.base import compute_dvdt, detect_events
from jaxley_ephys.features import SPIKE_FEATURE_FUNCTIONS, compute_spike_features
from tests.utils.jax_feature_contract import SPIKE_FEATURE_NAMES
from tests.utils.simulate_testdata import simulate_sweep, simulate_sweepset


@pytest.fixture(scope="module")
def synthetic_spike_primitives():
    t, v, i = simulate_sweep(dt=1e-4, t_max=0.12, stim_delay=0.02, stim_dur=0.08)
    max_spikes = 256
    ev = detect_events(t, v, i, 0.02, 0.1, max_spikes=max_spikes)
    dvdt = compute_dvdt(t, v)
    return t, v, i, ev, dvdt, max_spikes


def test_jax_spike_features_compute_on_synthetic_trace(synthetic_spike_primitives):
    t, v, i, ev, dvdt, max_spikes = synthetic_spike_primitives
    computed = {
        feature_name: SPIKE_FEATURE_FUNCTIONS[feature_name](t, v, i, ev, dvdt)
        for feature_name in SPIKE_FEATURE_NAMES
    }

    for feature_name in SPIKE_FEATURE_NAMES:
        assert feature_name in computed
        assert computed[feature_name].shape == (max_spikes,)


def test_compute_spike_features_jittable():
    t, v, i = simulate_sweep(dt=1e-4, t_max=0.12, stim_amp=0.7)
    fn = jax.jit(
        lambda tt, vv, ii: compute_spike_features(
            tt,
            vv,
            ii,
            0.02,
            0.1,
            max_spikes=16,
        )
    )
    out = fn(t, v, i)
    assert out["threshold_index"].shape == (16,)
    assert out["width"].shape == (16,)


def test_compute_spike_features_vmappable():
    t_batch, v_batch, i_batch = simulate_sweepset(
        num_sweeps=3,
        dt=1e-4,
        t_max=0.12,
        i_range=(0.2, 0.9),
    )

    fn = jax.vmap(
        lambda tt, vv, ii: compute_spike_features(
            tt,
            vv,
            ii,
            0.02,
            0.1,
            max_spikes=16,
        ),
        in_axes=(0, 0, 0),
    )
    out = fn(t_batch, v_batch, i_batch)
    assert out["threshold_index"].shape == (3, 16)
    assert out["width"].shape == (3, 16)


def test_spike_feature_registry_complete():
    for feature_name in SPIKE_FEATURE_NAMES:
        assert feature_name in SPIKE_FEATURE_FUNCTIONS
