#!/usr/bin/env python3
"""Download and cache Allen NWB fixtures for deterministic tests.

This script stores only AllenSDK-native cache artifacts under `tests/testdata/allen_cache`:

- `cell_types_manifest.json`
- `cells.json`
- `ephys_features.csv`
- `specimen_<id>/ephys_sweeps.json`
- `specimen_<id>/ephys.nwb`

No custom NPZ/JSON projections are generated.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tests.utils.allen_dataset import install_allen_compat_shims


DEFAULT_CELL_IDS = [
    525011903,
    528642047,
    537256313,
    519832676,
    469801138,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and cache Allen NWB fixtures for selected specimen IDs."
    )
    parser.add_argument(
        "--cell-ids",
        nargs="+",
        type=int,
        default=DEFAULT_CELL_IDS,
        help="Specimen IDs to cache.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("tests/testdata/allen_cache"),
        help="Root directory for AllenSDK cache artifacts.",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=None,
        help="Optional manifest file path. Default: <cache-dir>/cell_types_manifest.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete specimen cache directories before re-fetching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = args.manifest_file or (cache_dir / "cell_types_manifest.json")
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    install_allen_compat_shims()
    from allensdk.core.cell_types_cache import CellTypesCache  # type: ignore[import-not-found]

    ctc = CellTypesCache(manifest_file=str(manifest_file))

    # Ensure global metadata tables are present in cache.
    ctc.get_cells(require_reconstruction=False)
    ctc.get_ephys_features()

    specimen_ids = [int(specimen_id) for specimen_id in args.cell_ids]
    for specimen_id in specimen_ids:
        specimen_dir = cache_dir / f"specimen_{specimen_id}"
        if args.force and specimen_dir.exists():
            shutil.rmtree(specimen_dir)

        print(f"[fetch] specimen {specimen_id}")
        ctc.get_ephys_sweeps(specimen_id)
        ctc.get_ephys_data(specimen_id)

        sweeps_path = specimen_dir / "ephys_sweeps.json"
        nwb_path = specimen_dir / "ephys.nwb"
        if not sweeps_path.exists() or not nwb_path.exists():
            raise RuntimeError(
                f"Missing expected cache files for specimen {specimen_id}: "
                f"{sweeps_path} and {nwb_path}"
            )
        print(f"[ok] cached {specimen_id}")

    print("Done.")


if __name__ == "__main__":
    main()
