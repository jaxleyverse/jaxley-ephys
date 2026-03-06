#!/usr/bin/env python3
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.conftest import build_specimen_cases
from tests.utils.match_data_utils import compare_with_database


def main() -> None:
    cases = build_specimen_cases()

    status_totals: Counter[str] = Counter()
    key_status: dict[str, Counter[str]] = {}
    key_absdiff: dict[str, float] = {}
    key_samples: Counter[str] = Counter()

    for specimen_id, case in sorted(cases.items()):
        entries = compare_with_database(case)
        specimen_counts = Counter(entry.status for entry in entries.values())
        status_totals.update(specimen_counts)
        print(f"specimen={specimen_id} counts={dict(specimen_counts)}")

        for key, entry in entries.items():
            key_status.setdefault(key, Counter())
            key_status[key][entry.status] += 1
            if entry.abs_diff is not None:
                key_absdiff[key] = key_absdiff.get(key, 0.0) + entry.abs_diff
                key_samples[key] += 1

    print("\nstatus_totals", dict(status_totals))

    print("\nall-match features")
    all_match = sorted(
        key for key, counts in key_status.items() if set(counts.keys()) == {"match"}
    )
    for key in all_match:
        print(f"- {key}")

    print("\nunit-suspect features")
    for key in sorted(key_status):
        counts = key_status[key]
        if counts.get("unit_suspect", 0) > 0:
            print(f"- {key}: {counts.get('unit_suspect', 0)} / {sum(counts.values())}")

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
