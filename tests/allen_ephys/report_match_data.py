#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.allen_ephys.test_parity import (
    _computed_cell_feature_dict,
    _expected_cell_feature_dict,
    _feature_tolerance,
)
from tests.utils.allen_dataset import cached_specimen_ids, load_specimen_case


def main() -> None:
    specimen_ids = cached_specimen_ids()
    if not specimen_ids:
        raise RuntimeError(
            "No cached specimens found in tests/testdata/allen_cache. "
            "Run: uv run python tests/load_testdata.py"
        )

    totals: Counter[str] = Counter()
    mismatch_counts: Counter[str] = Counter()
    mismatch_abs_diff: dict[str, float] = {}

    for specimen_id in specimen_ids:
        specimen_case = load_specimen_case(specimen_id)
        expected = _expected_cell_feature_dict(specimen_case)
        computed = _computed_cell_feature_dict(specimen_case)

        specimen_totals: Counter[str] = Counter()
        for feature_name, expected_value in sorted(expected.items()):
            computed_value = computed.get(feature_name, float("nan"))

            if np.isnan(expected_value):
                specimen_totals["expected_nan"] += 1
                continue
            if np.isnan(computed_value):
                specimen_totals["computed_nan"] += 1
                mismatch_counts[feature_name] += 1
                continue

            atol, rtol = _feature_tolerance(feature_name)
            abs_diff = abs(computed_value - expected_value)
            if abs_diff <= atol + rtol * abs(expected_value):
                specimen_totals["match"] += 1
            else:
                specimen_totals["mismatch"] += 1
                mismatch_counts[feature_name] += 1
                mismatch_abs_diff[feature_name] = (
                    mismatch_abs_diff.get(feature_name, 0.0) + abs_diff
                )

        totals.update(specimen_totals)
        print(f"specimen={specimen_id} totals={dict(specimen_totals)}")

    print("\noverall", dict(totals))
    if not mismatch_counts:
        print("\nNo mismatches.")
        return

    print("\nmismatched features")
    for feature_name in sorted(mismatch_counts):
        count = mismatch_counts[feature_name]
        mean_abs = mismatch_abs_diff.get(feature_name, 0.0) / count
        print(f"- {feature_name}: count={count} mean_abs_diff={mean_abs:.6g}")


if __name__ == "__main__":
    main()
