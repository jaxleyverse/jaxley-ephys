"""The public extraction call owns compilation, shared dV/dt and safe capacity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxley_ephys.features import spikes


@pytest.fixture
def recording():
    jax.config.update("jax_enable_x64", True)
    t = np.arange(1601) * 0.000025
    v = np.interp(
        t,
        np.array([0, 10, 11, 11.3, 12, 14, 25, 26, 26.3, 27, 29, 40]) / 1000,
        [-65, -65, -10, 35, -75, -65, -65, -10, 35, -75, -65, -65],
    )
    return jnp.asarray(t), jnp.asarray(v), jnp.zeros_like(t)


def test_auto_capacity_matches_explicit_extraction(recording):
    t, v, i = recording
    names = ("peak_t", "threshold_v", "fast_trough_v", "width")
    fixed = spikes.compute_spike_features(
        t, v, i, t[0], t[-1], max_spikes=8, feature_names=names
    )
    automatic = spikes.compute_spike_features(
        t, v, i, t[0], t[-1], max_spikes=None, feature_names=names
    )
    assert np.count_nonzero(np.isfinite(automatic["peak_t"])) == 2
    for name in names:
        np.testing.assert_allclose(automatic[name], fixed[name], equal_nan=True)


def test_explicit_capacity_remains_jittable(recording):
    t, v, i = recording

    @jax.jit
    def extract(t, v, i):
        return spikes.compute_spike_features(
            t, v, i, t[0], t[-1], max_spikes=8, feature_names=("peak_t",)
        )["peak_t"]

    assert np.count_nonzero(np.isfinite(extract(t, v, i))) == 2


def test_public_call_raises_instead_of_returning_truncated_features(recording):
    t, v, i = recording
    with pytest.raises(ValueError, match="overflowed max_spikes=1"):
        spikes.compute_spike_features(
            t, v, i, t[0], t[-1], max_spikes=1, feature_names=("peak_t",)
        )


def test_automatic_capacity_counts_all_putative_crossings_once(monkeypatch):
    derivative = jnp.asarray(np.r_[np.tile([-1.0, 25.0], 70), np.zeros(60)])
    calls = []

    def dvdt(*args, **kwargs):
        calls.append("derivative")
        return derivative

    def extract(*args, max_spikes, **kwargs):
        calls.append(max_spikes)
        return False, {"peak_t": jnp.full(max_spikes, jnp.nan)}

    monkeypatch.setattr(spikes, "_spike_derivative", dvdt)
    monkeypatch.setattr(spikes, "_compute_spike_features", extract)
    t = jnp.arange(200) / 1000
    spikes.compute_spike_features(
        t,
        jnp.zeros(200),
        jnp.zeros(200),
        t[0],
        t[-1],
        max_spikes=None,
        feature_names=("peak_t",),
    )
    assert calls == ["derivative", 128]
