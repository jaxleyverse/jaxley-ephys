from __future__ import annotations

import numpy as np
from scipy import signal

from jaxley_ephys.bessel import (
    allen_bessel_sosfiltfilt,
    bessel4_lowpass_sos,
    sosfiltfilt,
)


def _synthetic_trace(n: int = 4096) -> np.ndarray:
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 24.0, n, dtype=np.float64)
    return np.sin(x) + 0.2 * np.sin(13.0 * x) + 0.05 * rng.standard_normal(n)


def test_bessel4_sos_frequency_response_matches_scipy():
    w = np.linspace(0.0, np.pi, 1024)
    for wn in (0.05, 0.1, 0.2, 0.4):
        sos_jax = np.asarray(bessel4_lowpass_sos(wn), dtype=np.float64)
        sos_scipy = signal.bessel(4, wn, "low", output="sos")

        _, h_jax = signal.sosfreqz(sos_jax, worN=w)
        _, h_scipy = signal.sosfreqz(sos_scipy, worN=w)
        np.testing.assert_allclose(h_jax, h_scipy, rtol=0.0, atol=1e-5)


def test_sosfiltfilt_matches_scipy_on_synthetic_signal():
    x = _synthetic_trace()
    for wn in (0.05, 0.1, 0.2):
        y_jax = np.asarray(sosfiltfilt(bessel4_lowpass_sos(wn), x))
        y_scipy = signal.sosfiltfilt(signal.bessel(4, wn, "low", output="sos"), x)
        np.testing.assert_allclose(y_jax, y_scipy, rtol=0.0, atol=1e-5)


def test_allen_bessel_helper_matches_scipy_filtfilt():
    dt = 5e-6  # 200 kHz sampling, valid for 10 kHz low-pass
    t = np.arange(6000, dtype=np.float64) * dt
    v = np.sin(2 * np.pi * 1000 * t) + 0.2 * np.sin(2 * np.pi * 25000 * t)

    y_jax = np.asarray(allen_bessel_sosfiltfilt(v, t, cutoff_khz=10.0))
    wn = (10e3) / ((1.0 / dt) / 2.0)
    b, a = signal.bessel(4, wn, "low")
    y_scipy = signal.filtfilt(b, a, v, axis=0)

    np.testing.assert_allclose(y_jax, y_scipy, rtol=0.0, atol=2e-5)
