from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxley_ephys.base import compute_dvdt, detect_events
from jaxley_ephys.features.long_square.sweep import SWEEP_FEATURE_FUNCTIONS, sag
from jaxley_ephys.features.ramp.sweep import median_isi as ramp_median_isi
from jaxley_ephys.features.short_square.sweep import median_isi as short_square_median_isi
from jaxley_ephys.features import compute_sweep_features
from tests.utils.jax_feature_contract import SWEEP_FEATURE_NAMES
from tests.utils.simulate_testdata import simulate_sweep, simulate_sweepset


@pytest.fixture(scope="module")
def synthetic_sweep_primitives():
    t, v, i = simulate_sweep(dt=1e-4, t_max=0.12, stim_delay=0.02, stim_dur=0.08)
    ev = detect_events(t, v, i, 0.02, 0.1, max_spikes=256)
    dvdt = compute_dvdt(t, v)
    return t, v, i, ev, dvdt


def test_jax_sweep_features_compute_on_synthetic_trace(synthetic_sweep_primitives):
    t, v, i, ev, dvdt = synthetic_sweep_primitives

    computed = compute_sweep_features(t, v, i, ev, dvdt)

    for feature_name in SWEEP_FEATURE_NAMES:
        assert feature_name in computed


def test_sweep_feature_registry_signature(synthetic_sweep_primitives):
    t, v, i, ev, dvdt = synthetic_sweep_primitives

    for feature_name in SWEEP_FEATURE_NAMES:
        assert feature_name in SWEEP_FEATURE_FUNCTIONS

    for fn in SWEEP_FEATURE_FUNCTIONS.values():
        value = fn(t, v, i, ev, dvdt)
        assert value.shape == ()


def test_compute_sweep_features_jittable(synthetic_sweep_primitives):
    t, v, i, ev, dvdt = synthetic_sweep_primitives
    fn = jax.jit(compute_sweep_features)
    out = fn(t, v, i, ev, dvdt)
    for feature_name in SWEEP_FEATURE_NAMES:
        assert feature_name in out

def test_sag_vmapped_hyperpolarizing_vs_depolarizing():
    t_batch, v_batch, i_batch = simulate_sweepset(
        num_sweeps=2,
        dt=1e-4,
        t_max=0.12,
        stim_delay=0.02,
        stim_dur=0.08,
        i_range=(0.6, -0.6),
    )

    ev_batch = jax.vmap(
        lambda tt, vv, ii: detect_events(tt, vv, ii, 0.02, 0.1, max_spikes=256),
        in_axes=(0, 0, 0),
    )(t_batch, v_batch, i_batch)
    dvdt_batch = jax.vmap(compute_dvdt, in_axes=(0, 0))(t_batch, v_batch)

    sag_values = jax.vmap(sag, in_axes=(0, 0, 0, 0, 0))(
        t_batch,
        v_batch,
        i_batch,
        ev_batch,
        dvdt_batch,
    )

    assert jnp.isnan(sag_values[0])
    assert jnp.isfinite(sag_values[1])


def test_stim_amp_returns_nan_for_ramp_like_stimulus():
    t, v, i = simulate_sweep(dt=1e-4, t_max=0.12, stim_delay=0.02, stim_dur=0.08, stim_amp=0.6)
    ramp_mask = (t >= 0.02) & (t < 0.1)
    i = jnp.where(ramp_mask, jnp.linspace(0.0, 0.6, t.shape[0]), i)

    ev = detect_events(t, v, i, 0.02, 0.1, max_spikes=256)
    dvdt = compute_dvdt(t, v)
    out = compute_sweep_features(t, v, i, ev, dvdt)

    assert jnp.isnan(out["stim_amp"])


def test_stimulus_sweep_modules_reexport_common_features():
    assert ramp_median_isi is SWEEP_FEATURE_FUNCTIONS["median_isi"]
    assert short_square_median_isi is SWEEP_FEATURE_FUNCTIONS["median_isi"]
