"""Comparing JAX outputs with the vendored-Allen reference."""

from __future__ import annotations

import numpy as np

# Columns of Allen's spike table, in Allen's order.
SPIKE_COLUMNS = (
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
ISI_TYPES = {"direct": 0, "detour": 1}  # -1 where Allen has no type

# Agreement is to floating-point precision, except for tau: Allen's curve_fit
# stops at its default tolerance, up to 1.1e-3 (relative) short of the
# least-squares optimum. Known differences: allen_ephys/ALLEN_EPHYS_FIXES.md,
# section 6.
RTOL = {"tau": 2e-3}
DEFAULT_RTOL = 1e-9
ATOL = 1e-9


def allen_values(name: str, values) -> np.ndarray:
    """Allen spike-table column as floats (NaN where undefined)."""
    if name == "isi_type":
        return np.array(
            [ISI_TYPES.get(x, -1) if isinstance(x, str) else -1 for x in values], float
        )
    return np.asarray(values, dtype=float)


def jax_values(name: str, values) -> np.ndarray:
    """JAX spike feature as floats, with Allen's NaN for missing indices."""
    values = np.asarray(values, dtype=float)
    if name.endswith("_index"):
        values = np.where(values < 0, np.nan, values)
    return values


def mismatches(name: str, expected, actual) -> np.ndarray:
    """Boolean mask of entries that disagree (NaN agrees with NaN)."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    rtol = RTOL.get(name, DEFAULT_RTOL)
    close = np.isclose(actual, expected, rtol=rtol, atol=ATOL, equal_nan=True)
    return ~close
