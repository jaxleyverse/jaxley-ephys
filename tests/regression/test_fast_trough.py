"""Allen parity when a late voltage fall dominates post-spike dV/dt."""

import numpy as np

from allen_ephys.ephys_extractor import EphysSweepFeatureExtractor
from jaxley_ephys import detect_spikes, spike_features


def test_late_voltage_drop_uses_allen_trough_analysis_interval():
    t = np.arange(3201) * 0.000025
    v = np.interp(
        t,
        np.array([0, 10, 11, 11.3, 12, 14, 30, 30.025, 80]) / 1000,
        [-65, -65, -10, 35, -75, -55, -55, -70, -70],
    )
    oracle = EphysSweepFeatureExtractor(t=t, v=v, start=t[0], end=t[-1], filter=10)
    oracle.process_spikes()
    assert len(oracle.spikes()) == 1

    features = spike_features(detect_spikes(t, v, start=t[0], end=t[-1], max_spikes=8))
    index, voltage = features["fast_trough_index"], features["fast_trough_v"]
    assert index[0] == oracle.spike_feature("fast_trough_index")[0]
    np.testing.assert_allclose(voltage[0], oracle.spike_feature("fast_trough_v")[0])
    np.testing.assert_allclose(features["width"][0], oracle.spike_feature("width")[0])
    assert np.all(index[1:] == -1)
    assert np.all(np.isnan(voltage[1:]))
