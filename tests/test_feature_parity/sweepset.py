from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from jaxley_ephys.features.long_square.sweepset import compute_sweepset_features as compute_long
from jaxley_ephys.features.ramp.sweepset import compute_sweepset_features as compute_ramp
from jaxley_ephys.features.short_square.sweepset import compute_sweepset_features as compute_short
from tests.utils.jax_feature_contract import (
    LONG_SQUARE_SWEEPSET_FEATURE_NAMES,
    RAMP_SWEEPSET_FEATURE_NAMES,
    SHORT_SQUARE_SWEEPSET_FEATURE_NAMES,
    SWEEPSET_COMMON_FEATURE_NAMES,
    assert_feature_equal,
)


SAG_TARGET_MV = -100.0
SUBTHRESH_MIN_AMP = -100.0
SUBTHRESH_MAX_AMP = 0.0


def _sweep_column(ext, feature_name: str):
    if feature_name == "peak_deflect":
        pairs = ext.sweep_features("peak_deflect", allow_missing=True)
        return np.asarray([float(v[0]) for v in pairs], dtype=float)
    if feature_name == "peak_deflect_index":
        pairs = ext.sweep_features("peak_deflect", allow_missing=True)
        return np.asarray([float(v[1]) for v in pairs], dtype=float)
    return np.asarray(ext.sweep_features(feature_name, allow_missing=True), dtype=float)


def _vectors_from_ext(ext):
    return {
        name: jnp.asarray(_sweep_column(ext, name), dtype=float)
        for name in SWEEPSET_COMMON_FEATURE_NAMES
    }


def _nanmean_or_nan(values, mask=None):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if mask is not None:
        finite = finite & mask
    if not np.any(finite):
        return float("nan")
    return float(np.nanmean(arr[finite]))


def _spiking_mask(avg_rate):
    rate = np.asarray(avg_rate, dtype=float)
    return np.isfinite(rate) & (rate > 0.0)


def _subthreshold_mask(avg_rate):
    rate = np.asarray(avg_rate, dtype=float)
    return np.isfinite(rate) & (rate == 0.0)


def _common_amp(avg_rate, stim_amp):
    rate = np.asarray(avg_rate, dtype=float)
    amp = np.asarray(stim_amp, dtype=float)
    mask = _spiking_mask(rate) & np.isfinite(amp)
    vals = amp[mask]
    if vals.size == 0:
        return float("nan")
    uniq, counts = np.unique(vals, return_counts=True)
    max_count = counts.max()
    return float(np.min(uniq[counts == max_count]))


def _expected_long(feature_name: str, columns: dict[str, np.ndarray]) -> float:
    avg_rate = columns["avg_rate"]
    stim_amp = columns["stim_amp"]
    peak_deflect = columns["peak_deflect"]
    v_baseline = columns["v_baseline"]
    tau = columns["tau"]
    sag = columns["sag"]

    if feature_name == "rheobase_index":
        spiking = _spiking_mask(avg_rate) & np.isfinite(stim_amp)
        if not np.any(spiking):
            return -1.0
        min_amp = np.min(stim_amp[spiking])
        candidates = np.where(spiking & np.isclose(stim_amp, min_amp))[0]
        if len(candidates) == 1:
            return float(candidates[0])
        rates = avg_rate[candidates]
        return float(candidates[np.nanargmax(rates)])

    if feature_name == "rheobase_i":
        idx = int(_expected_long("rheobase_index", columns))
        if idx < 0:
            return float("nan")
        return float(stim_amp[idx])

    if feature_name == "fi_fit_slope":
        spiking = _spiking_mask(avg_rate) & np.isfinite(stim_amp)
        x = stim_amp[spiking]
        y = avg_rate[spiking]
        if len(x) < 2:
            return float("nan")
        A = np.vstack([x, np.ones_like(x)]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(m)

    if feature_name in {"sag", "vm_for_sag"}:
        sub = _subthreshold_mask(avg_rate) & np.isfinite(peak_deflect)
        if not np.any(sub):
            return float("nan")
        idx = np.argmin(np.where(sub, np.abs(peak_deflect - SAG_TARGET_MV), np.inf))
        if feature_name == "sag":
            return float(sag[idx])
        return float(peak_deflect[idx])

    if feature_name == "input_resistance":
        mask = (
            _subthreshold_mask(avg_rate)
            & np.isfinite(stim_amp)
            & np.isfinite(peak_deflect)
            & (stim_amp > SUBTHRESH_MIN_AMP)
            & (stim_amp < SUBTHRESH_MAX_AMP)
        )
        if not np.any(mask):
            return float("nan")
        v = peak_deflect[mask]
        i = stim_amp[mask]
        if len(v) == 1:
            first_idx = np.where(mask)[0][0]
            v = np.append(v, v_baseline[first_idx])
            i = np.append(i, 0.0)
        A = np.vstack([i, np.ones_like(i)]).T
        m, _ = np.linalg.lstsq(A, v, rcond=None)[0]
        return float(m * 1e3)

    if feature_name == "tau":
        mask = (
            _subthreshold_mask(avg_rate)
            & np.isfinite(stim_amp)
            & (stim_amp > SUBTHRESH_MIN_AMP)
            & (stim_amp < SUBTHRESH_MAX_AMP)
        )
        return _nanmean_or_nan(tau, mask)

    return _nanmean_or_nan(columns[feature_name])


def _expected_short(feature_name: str, columns: dict[str, np.ndarray]) -> float:
    avg_rate = columns["avg_rate"]
    stim_amp = columns["stim_amp"]
    common = _common_amp(avg_rate, stim_amp)
    if feature_name == "stimulus_amplitude":
        return common
    mask = _spiking_mask(avg_rate) & np.isclose(stim_amp, common, rtol=0.0, atol=1e-9)
    if feature_name == "stim_amp":
        return common
    return _nanmean_or_nan(columns[feature_name], mask)


def _expected_ramp(feature_name: str, columns: dict[str, np.ndarray]) -> float:
    avg_rate = columns["avg_rate"]
    mask = _spiking_mask(avg_rate)
    return _nanmean_or_nan(columns[feature_name], mask)


@pytest.mark.parametrize("feature_name", LONG_SQUARE_SWEEPSET_FEATURE_NAMES)
def test_jax_long_square_sweepset_feature_matches_policy(
    specimen_sweepset_extractors,
    feature_name: str,
):
    ext = specimen_sweepset_extractors.get("long_square")
    if ext is None or not ext.sweeps():
        pytest.skip("no long-square sweeps available")

    vectors = _vectors_from_ext(ext)
    columns = {k: np.asarray(vectors[k], dtype=float) for k in vectors}
    expected = _expected_long(feature_name, columns)
    computed = compute_long(vectors)
    assert feature_name in computed
    assert_feature_equal(expected, computed[feature_name], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("feature_name", SHORT_SQUARE_SWEEPSET_FEATURE_NAMES)
def test_jax_short_square_sweepset_feature_matches_policy(
    specimen_sweepset_extractors,
    feature_name: str,
):
    ext = specimen_sweepset_extractors.get("short_square")
    if ext is None or not ext.sweeps():
        pytest.skip("no short-square sweeps available")

    vectors = _vectors_from_ext(ext)
    columns = {k: np.asarray(vectors[k], dtype=float) for k in vectors}
    expected = _expected_short(feature_name, columns)
    computed = compute_short(vectors)
    assert feature_name in computed
    assert_feature_equal(expected, computed[feature_name], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("feature_name", RAMP_SWEEPSET_FEATURE_NAMES)
def test_jax_ramp_sweepset_feature_matches_policy(
    specimen_sweepset_extractors,
    feature_name: str,
):
    ext = specimen_sweepset_extractors.get("ramp")
    if ext is None or not ext.sweeps():
        pytest.skip("no ramp sweeps available")

    vectors = _vectors_from_ext(ext)
    columns = {k: np.asarray(vectors[k], dtype=float) for k in vectors}
    expected = _expected_ramp(feature_name, columns)
    computed = compute_ramp(vectors)
    assert feature_name in computed
    assert_feature_equal(expected, computed[feature_name], atol=1e-4, rtol=1e-4)
