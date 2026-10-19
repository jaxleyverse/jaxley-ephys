"""JAX spike and sweep features vs the vendored-Allen reference, on every sweep.

Each test covers one feature for all sweeps of one stimulus type of a specimen
and reports the first disagreeing sweeps. ``KNOWN_GAPS`` lists what the current
implementation does not match yet (non-strict xfails); it should shrink to empty.
Intended differences (e.g. tau tolerance, dt == 0 handling) are not gaps; they
are documented in allen_ephys/ALLEN_EPHYS_FIXES.md, section 6.
It is calibrated on the default specimen plus the 2026-10-19 review of all cached
specimens. With the current implementation, ``--num_specimens 8`` needs a lot of
memory: its O(T * max_spikes) intermediates exhaust RAM on the 200 kHz sweeps.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.utils.allen_reference import KINDS, SWEEP_FEATURES
from tests.utils.parity import SPIKE_COLUMNS, allen_values, jax_values, mismatches

_ADP = "ADP not implemented (stubbed to -1/NaN)"
_SLOW = "slow trough not implemented (stubbed to -1/NaN)"
_THRESH = "thresh_frac is fixed at 0.05; Allen uses its adjusted value (>= 0.1) for short squares"
_ISI_MS = "median_isi/first_isi are returned in s; the reference uses ms"
# (kind or None for all kinds, feature) -> reason
KNOWN_GAPS = {
    (None, "isi_type"): "isi_type not implemented (always 'direct')",
    **{(None, f"adp_{s}"): _ADP for s in ("index", "t", "v", "i")},
    **{(None, f"slow_trough_{s}"): _SLOW for s in ("index", "t", "v", "i")},
    **{("short_square", f"threshold_{s}"): _THRESH for s in ("index", "t", "v", "i")},
    ("short_square", "latency"): _THRESH,
    (None, "latency"): "latency uses t[start_idx] instead of start",
    (None, "avg_rate"): "duration is t[end_idx] - t[start_idx] instead of end - start",
    (None, "median_isi"): _ISI_MS,
    (None, "first_isi"): _ISI_MS,
    (None, "tau"): "tau is a 256-point grid search, not a least-squares fit",
    (
        "ramp",
        "stim_amp",
    ): "stim_amp heuristic returns a value for some ramps (Allen: NaN)",
}


def _mark_known_gap(request, kind: str, name: str) -> None:
    reason = KNOWN_GAPS.get((kind, name)) or KNOWN_GAPS.get((None, name))
    if reason:
        request.applymarker(pytest.mark.xfail(reason=reason, strict=False))


def _report(kind: str, name: str, problems: list[str]) -> str:
    shown = "\n  ".join(problems[:5])
    return f"{kind} {name}: {len(problems)} sweep(s) differ from Allen, e.g.\n  {shown}"


@pytest.mark.parametrize("kind", KINDS)
def test_spike_counts(request, reference, results, kind):
    _mark_known_gap(request, kind, "spike_count")
    problems = [
        f"sweep {ref['sweep_number']}: allen {ref['n_spikes']}, jax {res['n_spikes']}"
        + (f" ({res['error']})" if res["error"] else "")
        for ref, res in zip(reference["sweeps"][kind], results[kind])
        if ref["n_spikes"] != res["n_spikes"]
    ]
    assert not problems, _report(kind, "spike count", problems)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("column", SPIKE_COLUMNS)
def test_spike_feature(request, reference, results, kind, column):
    _mark_known_gap(request, kind, column)
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
def test_sweep_feature(request, reference, results, kind, feature):
    _mark_known_gap(request, kind, feature)
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
