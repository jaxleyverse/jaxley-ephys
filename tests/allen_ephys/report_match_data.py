#!/usr/bin/env python3
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.conftest import build_specimen_cases
from tests.utils.match_data_utils import (
    DEFAULT_TOLERANCE,
    FEATURE_TOLERANCES,
    expected_db_feature_dict,
    flatten_for_db,
)


def main() -> None:
    cases = build_specimen_cases()

    status_totals: Counter[str] = Counter()
    key_status: dict[str, Counter[str]] = {}
    key_absdiff: dict[str, float] = {}
    key_samples: Counter[str] = Counter()

    for specimen_id, case in sorted(cases.items()):
        expected = expected_db_feature_dict(case.meta)
        computed = flatten_for_db(case.compute_cell_features())

        specimen_counts: Counter[str] = Counter()
        mismatches: list[tuple[str, float]] = []
        for key, exp in expected.items():
            comp = computed.get(key, float("nan"))
            if np.isnan(exp) and np.isnan(comp):
                status = "match"
            elif np.isnan(exp):
                status = "expected_missing"
            elif np.isnan(comp):
                status = "computed_missing"
            else:
                tol = FEATURE_TOLERANCES.get(key, DEFAULT_TOLERANCE)
                if np.isclose(exp, comp, atol=tol.atol, rtol=tol.rtol):
                    status = "match"
                else:
                    status = "mismatch"
                    mismatches.append((key, abs(comp - exp)))

            specimen_counts[status] += 1
            key_status.setdefault(key, Counter())
            key_status[key][status] += 1

        for key, abs_diff in mismatches:
            key_absdiff[key] = key_absdiff.get(key, 0.0) + abs_diff
            key_samples[key] += 1

        status_totals.update(specimen_counts)
        print(f"specimen={specimen_id} counts={dict(specimen_counts)}")

    print("\nstatus_totals", dict(status_totals))

    print("\nall-match features")
    all_match = sorted(
        key for key, counts in key_status.items() if set(counts.keys()) == {"match"}
    )
    for key in all_match:
        print(f"- {key}")

    print("\nmismatch features")
    for key in sorted(key_status):
        counts = key_status[key]
        if counts.get("mismatch", 0) > 0:
            n = sum(counts.values())
            mean_abs = key_absdiff.get(key, 0.0) / max(key_samples.get(key, 0), 1)
            print(
                f"- {key}: {counts.get('mismatch', 0)} / {n}, mean_abs_diff={mean_abs:.6g}"
            )


if __name__ == "__main__":
    main()
