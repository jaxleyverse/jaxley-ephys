"""``calculate_dvdt`` and its Bessel filter vs SciPy and the vendored Allen code."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from allen_ephys import ephys_features as allen
from jaxley_ephys import calculate_dvdt
from jaxley_ephys.dvdt import bessel4_sos

DT = 2e-5  # s (50 kHz)


def _trace(n: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """A spike-like bump on a slow oscillation plus noise (s, mV)."""
    rng = np.random.default_rng(7)
    t = np.arange(n) * DT
    v = (
        -65.0
        + 40.0 * np.exp(-(((t - 0.03) / 5e-4) ** 2))
        + 5.0 * np.sin(2 * np.pi * 50 * t)
        + 0.2 * rng.standard_normal(n)
    )
    return t, v


@pytest.mark.parametrize("wn", [0.05, 0.2, 0.4, 0.8])
def test_bessel_sos_matches_scipy(wn):
    w = np.linspace(0, np.pi, 512)
    _, h = signal.sosfreqz(np.asarray(bessel4_sos(wn)), worN=w)
    _, h_scipy = signal.sosfreqz(signal.bessel(4, wn, "low", output="sos"), worN=w)
    np.testing.assert_allclose(h, h_scipy, rtol=0, atol=1e-10)


@pytest.mark.parametrize("cutoff", [None, 1.0, 10.0])
def test_dvdt_matches_allen(cutoff):
    t, v = _trace()
    dvdt = np.asarray(calculate_dvdt(v, t, cutoff))
    np.testing.assert_allclose(
        dvdt[:-1], allen.calculate_dvdt(v, t, cutoff), rtol=1e-9, atol=1e-9
    )
    assert np.isnan(dvdt[-1])


def test_window_matches_allen_on_the_window():
    t, v = _trace()
    lo, hi = 1000, 3000
    dvdt = np.asarray(calculate_dvdt(v, t, 10.0, window=(lo, hi)))
    expected = allen.calculate_dvdt(v[lo : hi + 1], t[lo : hi + 1], 10.0)
    np.testing.assert_allclose(dvdt[lo:hi], expected, rtol=1e-9, atol=1e-9)


def test_variable_dt_is_not_filtered():
    t, v = _trace()
    t[2500:] += DT / 2
    dvdt = np.asarray(calculate_dvdt(v, t, 10.0))
    np.testing.assert_allclose(dvdt[:-1], 1e-3 * np.diff(v) / np.diff(t), rtol=1e-12)
    np.testing.assert_allclose(dvdt[:-1], allen.calculate_dvdt(v, t, 10.0), rtol=1e-12)


def test_zero_dt_gives_nan_instead_of_dropping_the_sample():
    t, v = _trace()
    t[2500] = t[2499]
    dvdt = np.asarray(calculate_dvdt(v, t, 10.0))
    assert np.isnan(dvdt[2499])
    expected = allen.calculate_dvdt(v, t, 10.0)  # one sample shorter
    np.testing.assert_allclose(np.delete(dvdt[:-1], 2499), expected, rtol=1e-12)


def test_cutoff_above_nyquist_gives_nan():
    t, v = _trace()
    with pytest.raises(ValueError):
        allen.calculate_dvdt(v, t, 30.0)  # Nyquist frequency: 25 kHz
    assert np.all(np.isnan(calculate_dvdt(v, t, 30.0)))


def test_float32_inputs_give_the_float64_result():
    t, v = _trace()
    t32, v32 = t.astype(np.float32), v.astype(np.float32)
    np.testing.assert_allclose(
        calculate_dvdt(v32, t32, 10.0),
        calculate_dvdt(v32.astype(float), t32.astype(float), 10.0),
        rtol=0,
        atol=0,
    )
