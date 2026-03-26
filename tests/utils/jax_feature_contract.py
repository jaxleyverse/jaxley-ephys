from __future__ import annotations

import numpy as np


SPIKE_FEATURE_NAMES = (
    "threshold_index",
    "clipped",
    "threshold_t",
    "threshold_v",
    "threshold_i",
    "peak_index",
    "peak_t",
    "peak_v",
    "peak_i",
    "trough_index",
    "trough_t",
    "trough_v",
    "trough_i",
    "upstroke_index",
    "upstroke",
    "upstroke_t",
    "upstroke_v",
    "downstroke_index",
    "downstroke",
    "downstroke_t",
    "downstroke_v",
    "isi_type",
    "fast_trough_index",
    "fast_trough_t",
    "fast_trough_v",
    "fast_trough_i",
    "adp_index",
    "adp_t",
    "adp_v",
    "adp_i",
    "slow_trough_index",
    "slow_trough_t",
    "slow_trough_v",
    "slow_trough_i",
    "width",
    "upstroke_downstroke_ratio",
)

SWEEP_PARITY_FEATURE_NAMES = (
    "adapt",
    "latency",
    "isi_cv",
    "mean_isi",
    "median_isi",
    "first_isi",
    "avg_rate",
)

SWEEP_HYPERPOL_FEATURE_NAMES = (
    "v_baseline",
    "tau",
    "sag",
    "peak_deflect",
    "peak_deflect_index",
    "stim_amp",
)

SWEEP_ALL_FEATURE_NAMES = SWEEP_PARITY_FEATURE_NAMES + SWEEP_HYPERPOL_FEATURE_NAMES
SWEEP_FEATURE_NAMES = SWEEP_ALL_FEATURE_NAMES
SWEEPSET_COMMON_FEATURE_NAMES = SWEEP_ALL_FEATURE_NAMES
LONG_SQUARE_SWEEPSET_EXTRA_FEATURE_NAMES = (
    "rheobase_index",
    "rheobase_i",
    "fi_fit_slope",
    "vm_for_sag",
    "input_resistance",
)
SHORT_SQUARE_SWEEPSET_EXTRA_FEATURE_NAMES = ("stimulus_amplitude",)

LONG_SQUARE_SWEEPSET_FEATURE_NAMES = (
    SWEEPSET_COMMON_FEATURE_NAMES + LONG_SQUARE_SWEEPSET_EXTRA_FEATURE_NAMES
)
RAMP_SWEEPSET_FEATURE_NAMES = SWEEPSET_COMMON_FEATURE_NAMES
SHORT_SQUARE_SWEEPSET_FEATURE_NAMES = (
    SWEEPSET_COMMON_FEATURE_NAMES + SHORT_SQUARE_SWEEPSET_EXTRA_FEATURE_NAMES
)

SWEEPSET_FEATURE_NAMES = SWEEPSET_COMMON_FEATURE_NAMES

def assert_feature_equal(
    expected,
    computed,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    expected_arr = np.asarray(expected)
    computed_arr = np.asarray(computed)

    assert computed_arr.shape == expected_arr.shape

    if expected_arr.dtype.kind in {"U", "S", "O"}:
        assert np.array_equal(computed_arr, expected_arr)
        return

    assert np.allclose(
        computed_arr.astype(float),
        expected_arr.astype(float),
        atol=atol,
        rtol=rtol,
        equal_nan=True,
    )
