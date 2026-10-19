"""Allen's own spike-detection test traces (``tests/testdata/spike_test_*.txt``).

Allen is run on the fly (default parameters, whole trace as analysis window).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from allen_ephys.ephys_extractor import EphysSweepFeatureExtractor
from tests.utils.jax_runner import analyse_sweep
from tests.utils.parity import allen_values, jax_values, mismatches

TESTDATA = Path(__file__).parent / "testdata"


def _allen_and_jax(name: str):
    data = np.loadtxt(TESTDATA / f"{name}.txt")
    t, v = data[:, 0], data[:, 1]
    allen = EphysSweepFeatureExtractor(t, v)
    allen.process_spikes()
    jax = analyse_sweep(t, v, np.zeros_like(v), t[0], t[-1])
    return t, v, allen._spikes_df, jax


@pytest.mark.xfail(
    reason="current implementation: thresholds 2 samples early", strict=False
)
@pytest.mark.parametrize("name", ["spike_test_pair", "spike_test_high_init_dvdt"])
def test_fixture_spike_table_matches_allen(name):
    _, _, table, jax = _allen_and_jax(name)
    assert jax["n_spikes"] == len(table)
    for column in table.columns:
        expected = allen_values(column, table[column])
        actual = jax_values(column, jax["spikes"][column])
        assert not mismatches(column, expected, actual).any(), column


@pytest.mark.xfail(
    reason="current implementation smooths dV/dt for variable dt instead of not filtering",
    strict=False,
)
def test_variable_dt_fixture_keeps_dvdt_aligned():
    """The trace repeats one time stamp (dt == 0). Allen drops that dV/dt sample,
    which shifts every later dV/dt index by one relative to ``v``; jaxley_ephys
    keeps dV/dt aligned (NaN at dt == 0), a documented deviation
    (allen_ephys/ALLEN_EPHYS_FIXES.md, section 6)."""
    t, v, table, jax = _allen_and_jax("spike_test_var_dt")
    assert np.sum(np.diff(t) == 0) == 1
    assert jax["n_spikes"] == len(table)
    for column in ("peak_index", "trough_index"):
        assert np.array_equal(jax_values(column, jax["spikes"][column]), table[column])
    for column in (
        "threshold_index",
        "upstroke_index",
        "downstroke_index",
        "fast_trough_index",
    ):
        assert np.array_equal(
            jax_values(column, jax["spikes"][column]), table[column] + 1
        )
