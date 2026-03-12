from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxley_ephys.base import detect_events
from jaxley_ephys.features import (
    compute_spike_features,
    compute_sweep_features,
    compute_sweepset_features,
)
from tests.utils.jax_feature_contract import SPIKE_FEATURE_NAMES, SWEEP_FEATURE_NAMES
from tests.utils.simulate_testdata import simulate_sweep, simulate_sweepset


def test_jax_spike_features_compute_on_synthetic_trace():
    t, v, i = simulate_sweep(dt=1e-4, t_max=0.12, stim_delay=0.02, stim_dur=0.08)

    computed = compute_spike_features(t, v, i, start=0.02, end=0.1)

    for feature_name in SPIKE_FEATURE_NAMES:
        assert feature_name in computed


def test_jax_sweep_features_compute_on_synthetic_trace():
    t, v, i = simulate_sweep(dt=1e-4, t_max=0.12, stim_delay=0.02, stim_dur=0.08)

    computed = compute_sweep_features(t, v, i, start=0.02, end=0.1)

    for feature_name in SWEEP_FEATURE_NAMES:
        assert feature_name in computed


def test_jax_sweepset_features_compute_on_synthetic_sweepset():
    t_set, v_set, i_set = simulate_sweepset(
        num_sweeps=4,
        dt=1e-4,
        t_max=0.12,
        i_range=(0.25, 0.9),
    )

    start_set = [0.02] * int(t_set.shape[0])
    end_set = [0.1] * int(t_set.shape[0])

    computed = compute_sweepset_features(t_set, v_set, i_set, start_set, end_set)

    assert "sweep_features" in computed
    assert "spike_feature_averages" in computed

    for feature_name in SWEEP_FEATURE_NAMES:
        assert feature_name in computed["sweep_features"]

    for feature_name in SPIKE_FEATURE_NAMES:
        assert feature_name in computed["spike_feature_averages"]


def test_detect_events_jittable_in_integrity_suite():
    t, v, i = simulate_sweep(dt=1e-4, t_max=0.12, stim_amp=0.7)

    fn = jax.jit(lambda tt, vv, ii: detect_events(tt, vv, ii, 0.02, 0.1, max_spikes=16))
    out = fn(t, v, i)

    assert out.spikes.shape == t.shape
    assert out.threshold_idxs.shape == (16,)
    assert out.peak_idxs.shape == (16,)
    assert out.trough_idxs.shape == (16,)


def test_detect_events_vmappable_in_integrity_suite():
    t_batch, v_batch, i_batch = simulate_sweepset(
        num_sweeps=3,
        dt=1e-4,
        t_max=0.12,
        i_range=(0.2, 0.9),
    )
    t0 = jnp.full((t_batch.shape[0],), 0.02)
    t1 = jnp.full((t_batch.shape[0],), 0.1)

    fn = jax.vmap(
        lambda tt, vv, ii, t0_, t1_: detect_events(
            tt,
            vv,
            ii,
            t0_,
            t1_,
            max_spikes=16,
        ),
        in_axes=(0, 0, 0, 0, 0),
    )
    out = fn(t_batch, v_batch, i_batch, t0, t1)

    assert out.spikes.shape == t_batch.shape
    assert out.threshold_idxs.shape == (3, 16)
    assert out.peak_idxs.shape == (3, 16)
    assert out.trough_idxs.shape == (3, 16)
