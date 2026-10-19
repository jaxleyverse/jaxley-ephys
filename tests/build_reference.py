#!/usr/bin/env python3
"""Build the vendored-Allen reference used by the parity tests.

Run after ``tests/load_testdata.py``. References that are already up to date are
skipped unless ``--force`` is given. See ``tests/utils/allen_reference.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.utils.allen_dataset import cached_specimen_ids
from tests.utils.allen_reference import is_current, write_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specimen-ids",
        nargs="+",
        type=int,
        default=None,
        help="Specimens to process (default: all cached specimens).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild up-to-date files."
    )
    args = parser.parse_args()

    specimen_ids = args.specimen_ids or cached_specimen_ids()
    if not specimen_ids:
        raise SystemExit(
            "No cached specimens found. Run: uv run python tests/load_testdata.py"
        )
    for specimen_id in specimen_ids:
        if not args.force and is_current(specimen_id):
            print(f"[skip] specimen {specimen_id} is up to date")
            continue
        start = time.perf_counter()
        path = write_reference(specimen_id)
        print(
            f"[ok] specimen {specimen_id} -> {path} ({time.perf_counter() - start:.0f}s)"
        )


if __name__ == "__main__":
    main()
