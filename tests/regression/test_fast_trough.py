"""Allen parity when a late voltage fall dominates post-spike dV/dt."""

import jax
import jax.numpy as jnp
import numpy as np
from allen_ephys.ephys_extractor import EphysSweepFeatureExtractor
from jaxley_ephys import compute_dvdt, detect_events
from jaxley_ephys.features import spikes


def test_late_voltage_drop_uses_allen_trough_analysis_interval():
    jax.config.update("jax_enable_x64", True)
    t = np.arange(3201) * 0.000025
    v = np.interp(
        t,
        np.array([0, 10, 11, 11.3, 12, 14, 30, 30.025, 80]) / 1000,
        [-65, -65, -10, 35, -75, -55, -55, -70, -70],
    )
    oracle = EphysSweepFeatureExtractor(t=t, v=v, start=t[0], end=t[-1], filter=10)
    oracle.process_spikes()
    assert len(oracle.spikes()) == 1

    @jax.jit
    def extract(t, v):
        current = jnp.zeros_like(v)
        dvdt = compute_dvdt(t, v, filter=10)
        ev = detect_events(t, v, current, t[0], t[-1], dvdt=dvdt, max_spikes=8)
        return (
            spikes.fast_trough_index(t, v, current, ev, dvdt),
            spikes.fast_trough_v(t, v, current, ev, dvdt),
            spikes.width(t, v, current, ev, dvdt),
        )

    index, voltage, width = jax.device_get(extract(jnp.asarray(t), jnp.asarray(v)))
    assert index[0] == oracle.spike_feature("fast_trough_index")[0]
    np.testing.assert_allclose(voltage[0], oracle.spike_feature("fast_trough_v")[0])
    np.testing.assert_allclose(width[0], oracle.spike_feature("width")[0])
    assert np.all(index[1:] == -1)
    assert np.all(np.isnan(voltage[1:]))
