"""float32 (``jax_enable_x64`` off) vs the vendored-Allen reference.

Exact parity needs float64; in float32 jaxley_ephys stays close (see
allen_ephys/ALLEN_EPHYS_FIXES.md, section 6). On the 8 cached specimens: same
spike counts; of about 14,000 spike indices a few move by one sample and 2 by
more (a trough at the window edge, a fast trough in a flat stretch of dV/dt);
features agree to about 2e-4, ``tau`` and ``sag`` to 1.5e-3. The inputs are the
float64 traces, which JAX casts to float32 as for any user without x64.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tests.utils.allen_reference import KINDS, SWEEP_FEATURES
from tests.utils.jax_runner import analyse_cell, analyse_sweep
from tests.utils.parity import SPIKE_COLUMNS, allen_values, jax_values

# tau also includes Allen's own optimizer tolerance (up to 1.1e-3); sag is a
# ratio with a small numerator.
RTOL = {"tau": 3e-3, "sag": 3e-3}
DEFAULT_RTOL = 5e-4
ATOL = 1e-6
MAX_INDEX_SHIFT = 1  # samples
MAX_FAR_FRACTION = 0.01  # of spike indices further than MAX_INDEX_SHIFT


@pytest.fixture(scope="module")
def float32_results(sweeps):
    """Sweep and cell results computed with x64 disabled."""
    jax.config.update("jax_enable_x64", False)
    try:
        per_sweep = {
            kind: [
                analyse_sweep(
                    s.t,
                    s.v,
                    s.i,
                    s.start,
                    s.end,
                    stimulus=kind,
                    dv_cutoff=s.dv_cutoff,
                    thresh_frac=s.thresh_frac,
                )
                for s in sweeps[kind]
            ]
            for kind in KINDS
        }
        as_tuples = {
            kind: [(s.t, s.v, s.i, s.start, s.end) for s in sweeps[kind]]
            for kind in KINDS
        }
        cell = analyse_cell(
            as_tuples["long_square"], as_tuples["short_square"], as_tuples["ramp"]
        )
    finally:
        jax.config.update("jax_enable_x64", True)
    return per_sweep, cell


def _close(name: str, expected, actual) -> np.ndarray:
    rtol = RTOL.get(name, DEFAULT_RTOL)
    return np.isclose(actual, expected, rtol=rtol, atol=ATOL, equal_nan=True)


def test_float32_spikes_match_allen(reference, float32_results):
    """Same spike counts; spike indices within one sample, except for at most
    MAX_FAR_FRACTION of them (pooled over the specimen's sweeps)."""
    problems, far, total = [], [], 0
    pairs = [
        (ref, res)
        for kind in KINDS
        for ref, res in zip(reference["sweeps"][kind], float32_results[0][kind])
    ]
    for ref, res in pairs:
        sweep = ref["sweep_number"]
        if res["n_spikes"] != ref["n_spikes"]:
            problems.append(
                f"sweep {sweep}: {res['n_spikes']} vs {ref['n_spikes']} spikes"
            )
            continue
        for column in SPIKE_COLUMNS:
            if not column.endswith("_index") or not ref["n_spikes"]:
                continue
            expected = allen_values(column, ref["spikes"][column])
            actual = jax_values(column, res["spikes"][column])
            total += len(expected)
            missing = np.isnan(expected) != np.isnan(actual)
            shift = np.where(
                np.isnan(expected) | np.isnan(actual), 0, expected - actual
            )
            for k in np.flatnonzero(missing | (np.abs(shift) > MAX_INDEX_SHIFT)):
                far.append(
                    f"sweep {sweep}, {column}[{k}]: {actual[k]} vs {expected[k]}"
                )
    if len(far) > MAX_FAR_FRACTION * total:
        problems += far
    assert not problems, problems[:5]


@pytest.mark.parametrize("kind", KINDS)
def test_float32_sweep_features_match_allen(reference, float32_results, kind):
    problems = [
        f"sweep {ref['sweep_number']} {name}: {res['features'][name]} vs "
        f"{ref['features'][name]}"
        for ref, res in zip(reference["sweeps"][kind], float32_results[0][kind])
        for name in SWEEP_FEATURES
        if not _close(name, ref["features"][name], res["features"][name])
    ]
    assert not problems, problems[:5]


def test_float32_cell_features_match_allen(reference, float32_results):
    cell = float32_results[1]["cell"]
    problems = [
        f"{name}: {float(cell[name])} vs {expected}"
        for name, expected in reference["cell"]["record"].items()
        if not _close(name, expected, float(cell[name]))
    ]
    assert not problems, problems
