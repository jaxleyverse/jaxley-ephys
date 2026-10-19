"""The public detection call owns compilation and a safe spike capacity."""

import jax
import numpy as np
import pytest

from jaxley_ephys import detect_spikes, spike_features

DT = 0.000025  # s
# One spike-like waveform: (time in ms, voltage in mV), 4 ms long
SPIKE = (np.array([0, 1, 1.3, 2, 4]), np.array([-65, -10, 35, -75, -65]))


def _spike_train(n_spikes: int) -> tuple[np.ndarray, np.ndarray]:
    """Spikes every 15 ms from 10 ms on, at -65 mV in between."""
    t = np.arange(round((10 + 15 * n_spikes) / 1000 / DT) + 1) * DT
    times = np.concatenate([10 + 15 * k + SPIKE[0] for k in range(n_spikes)])
    return t, np.interp(t, times / 1000, np.tile(SPIKE[1], n_spikes))


@pytest.fixture
def recording():
    return _spike_train(2)


def test_auto_capacity_matches_explicit_capacity(recording):
    names = ("peak_t", "threshold_v", "fast_trough_v", "width")
    fixed = spike_features(detect_spikes(*recording, max_spikes=8))
    automatic = spike_features(detect_spikes(*recording, max_spikes=None))
    assert np.count_nonzero(np.isfinite(automatic["peak_t"])) == 2
    for name in names:
        np.testing.assert_allclose(automatic[name][:8], fixed[name], equal_nan=True)
        assert np.all(np.isnan(automatic[name][8:]))


def test_explicit_capacity_remains_jittable(recording):
    @jax.jit
    def peak_times(t, v):
        return spike_features(detect_spikes(t, v, max_spikes=8))["peak_t"]

    assert np.count_nonzero(np.isfinite(peak_times(*recording))) == 2


def test_auto_capacity_needs_concrete_inputs(recording):
    with pytest.raises(ValueError, match="max_spikes=None"):
        jax.jit(lambda t, v: detect_spikes(t, v, max_spikes=None).peak)(*recording)


def test_public_call_raises_instead_of_returning_truncated_features(recording):
    with pytest.raises(ValueError, match="max_spikes=1"):
        detect_spikes(*recording, max_spikes=1)


def test_auto_capacity_grows_until_all_spikes_fit():
    sp = detect_spikes(*_spike_train(300), max_spikes=None)
    assert int(sp.n_spikes) == 300
    assert not bool(sp.overflow)
