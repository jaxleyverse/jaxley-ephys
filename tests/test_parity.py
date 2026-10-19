"""JAX spike and sweep features vs the vendored-Allen reference, on every sweep.

Each test covers one feature for all sweeps of one stimulus type of a specimen
and reports the first disagreeing sweeps. Intended differences (e.g. the tau
tolerance) are documented in allen_ephys/ALLEN_EPHYS_FIXES.md, section 6.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.utils.allen_reference import KINDS, SWEEP_FEATURES
from tests.utils.parity import SPIKE_COLUMNS, allen_values, jax_values, mismatches


def _report(kind: str, name: str, problems: list[str]) -> str:
    shown = "\n  ".join(problems[:5])
    return f"{kind} {name}: {len(problems)} sweep(s) differ from Allen, e.g.\n  {shown}"


@pytest.mark.parametrize("kind", KINDS)
def test_spike_counts(reference, results, kind):
    problems = [
        f"sweep {ref['sweep_number']}: allen {ref['n_spikes']}, jax {res['n_spikes']}"
        + (f" ({res['error']})" if res["error"] else "")
        for ref, res in zip(reference["sweeps"][kind], results[kind])
        if ref["n_spikes"] != res["n_spikes"]
    ]
    assert not problems, _report(kind, "spike count", problems)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("column", SPIKE_COLUMNS)
def test_spike_feature(reference, results, kind, column):
    problems = []
    for ref, res in zip(reference["sweeps"][kind], results[kind]):
        if ref["n_spikes"] == 0 or ref["n_spikes"] != res["n_spikes"]:
            continue  # spike counts are checked separately
        expected = allen_values(column, ref["spikes"][column])
        actual = jax_values(column, res["spikes"][column])
        bad = mismatches(column, expected, actual)
        if bad.any():
            k = int(np.argmax(bad))
            problems.append(
                f"sweep {ref['sweep_number']}, spike {k}: allen {expected[k]}, jax {actual[k]}"
            )
    assert not problems, _report(kind, column, problems)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("feature", SWEEP_FEATURES)
def test_sweep_feature(reference, results, kind, feature):
    problems = []
    for ref, res in zip(reference["sweeps"][kind], results[kind]):
        if res["error"]:
            continue  # reported by test_spike_counts
        expected, actual = ref["features"][feature], res["features"][feature]
        if mismatches(feature, [expected], [actual])[0]:
            problems.append(
                f"sweep {ref['sweep_number']}: allen {expected}, jax {actual}"
            )
    assert not problems, _report(kind, feature, problems)
