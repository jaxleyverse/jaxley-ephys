#!/usr/bin/env python3
"""Download and cache reproducible Allen ephys testcases.

This script fetches mixed mouse/human specimens from the Allen Cell Types
Database, then stores sweep traces and metadata locally for deterministic tests.

Output per cell (test artifacts):
- tests/data/cell_<specimen_id>_data.npz
- tests/data/cell_<specimen_id>_meta.json

AllenSDK raw/download cache (re-downloadable):
- tests/data/allen_cache/

NPZ schema (one row per selected sweep):
- specimen_id: scalar int
- sweep_numbers: int[N]
- sweep_ids: int[N]
- stimulus_names: object[N] of str
- stimulus_start_times: float[N]
- stimulus_durations: float[N]
- stimulus_absolute_amplitudes: float[N]
- sampling_rates: float[N]
- index_ranges: int[N, 2]
- t: object[N], each element is float[T_i] in seconds
- v: object[N], each element is float[T_i] in mV
- i: object[N], each element is float[T_i] in pA

Notes:
- AllenSDK currently imports `numpy.VisibleDeprecationWarning`, which is not
  available in newer NumPy versions. A local compatibility shim is applied.
- AllenSDK is used here in tests/dev tooling only (not runtime package code).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CELL_IDS = [
    # Mixed panel: 4 human + 4 mouse specimens
    525011903,
    528642047,
    537256313,
    519832676,
    565871768,
    469801138,
    605889373,
    485909730,
]


def _install_numpy_compat_shim() -> None:
    """Install compatibility shim for AllenSDK on newer NumPy versions."""
    if not hasattr(np, "VisibleDeprecationWarning"):
        np.VisibleDeprecationWarning = Warning  # type: ignore[attr-defined]


def _install_h5py_compat_shim() -> None:
    """Restore deprecated h5py Dataset.value used by older AllenSDK code."""
    import h5py  # type: ignore[import-not-found]

    dataset_type = h5py.Dataset
    if not hasattr(dataset_type, "value"):
        dataset_type.value = property(lambda self: self[()])  # type: ignore[attr-defined]


def _is_target_stimulus(stimulus_name: str | None) -> bool:
    if not stimulus_name:
        return False
    name = stimulus_name.lower()
    return "long square" in name or "short square" in name or "ramp" in name


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _build_ephys_feature_lookup(cell_types_cache: Any) -> dict[int, dict[str, Any]]:
    rows = cell_types_cache.get_ephys_features()
    by_specimen: dict[int, dict[str, Any]] = {}
    for row in rows:
        specimen_id = int(row["specimen_id"])
        by_specimen[specimen_id] = row
    return by_specimen


def _save_cell_cache(
    cell_types_cache: Any,
    cell: dict[str, Any],
    feature_lookup: dict[int, dict[str, Any]],
    output_dir: Path,
    overwrite: bool,
) -> None:
    specimen_id = int(cell["id"])
    npz_path = output_dir / f"cell_{specimen_id}_data.npz"
    meta_path = output_dir / f"cell_{specimen_id}_meta.json"

    if not overwrite and npz_path.exists() and meta_path.exists():
        print(f"[skip] specimen {specimen_id}: cache already exists")
        return

    print(f"[fetch] specimen {specimen_id} ({cell.get('species')})")
    sweeps = cell_types_cache.get_ephys_sweeps(specimen_id)
    target_sweeps = [s for s in sweeps if _is_target_stimulus(s.get("stimulus_name"))]

    if not target_sweeps:
        raise RuntimeError(
            f"Specimen {specimen_id} has no long-square/ramp/short-square sweeps"
        )

    ephys_data = cell_types_cache.get_ephys_data(specimen_id)

    sweep_numbers: list[int] = []
    sweep_ids: list[int] = []
    stimulus_names: list[str] = []
    stimulus_start_times: list[float] = []
    stimulus_durations: list[float] = []
    stimulus_absolute_amplitudes: list[float] = []
    sampling_rates: list[float] = []
    index_ranges: list[tuple[int, int]] = []

    t_series: list[np.ndarray] = []
    v_series: list[np.ndarray] = []
    i_series: list[np.ndarray] = []

    for sweep_meta in sorted(target_sweeps, key=lambda s: int(s["sweep_number"])):
        sweep_number = int(sweep_meta["sweep_number"])
        sweep_id = int(sweep_meta["id"])

        sweep_data = ephys_data.get_sweep(sweep_number)
        response = np.asarray(sweep_data["response"], dtype=np.float64) * 1e3
        stimulus = np.asarray(sweep_data["stimulus"], dtype=np.float64) * 1e12
        sampling_rate = float(sweep_data["sampling_rate"])
        t = np.arange(response.size, dtype=np.float64) / sampling_rate
        idx_range = sweep_data["index_range"]

        sweep_numbers.append(sweep_number)
        sweep_ids.append(sweep_id)
        stimulus_names.append(str(sweep_meta.get("stimulus_name", "")))
        stimulus_start_times.append(_safe_float(sweep_meta.get("stimulus_start_time")))
        stimulus_durations.append(_safe_float(sweep_meta.get("stimulus_duration")))
        stimulus_absolute_amplitudes.append(
            _safe_float(sweep_meta.get("stimulus_absolute_amplitude"))
        )
        sampling_rates.append(sampling_rate)
        index_ranges.append((int(idx_range[0]), int(idx_range[1])))

        t_series.append(t)
        v_series.append(response)
        i_series.append(stimulus)

    np.savez_compressed(
        npz_path,
        specimen_id=np.int64(specimen_id),
        sweep_numbers=np.asarray(sweep_numbers, dtype=np.int64),
        sweep_ids=np.asarray(sweep_ids, dtype=np.int64),
        stimulus_names=np.asarray(stimulus_names, dtype=object),
        stimulus_start_times=np.asarray(stimulus_start_times, dtype=np.float64),
        stimulus_durations=np.asarray(stimulus_durations, dtype=np.float64),
        stimulus_absolute_amplitudes=np.asarray(
            stimulus_absolute_amplitudes, dtype=np.float64
        ),
        sampling_rates=np.asarray(sampling_rates, dtype=np.float64),
        index_ranges=np.asarray(index_ranges, dtype=np.int64),
        t=np.asarray(t_series, dtype=object),
        v=np.asarray(v_series, dtype=object),
        i=np.asarray(i_series, dtype=object),
    )

    meta = {
        "specimen_id": specimen_id,
        "cell_metadata": cell,
        "ephys_features": feature_lookup.get(specimen_id),
        "selected_sweeps": target_sweeps,
        "n_selected_sweeps": len(target_sweeps),
        "n_total_sweeps": len(sweeps),
        "stimulus_types_present": sorted(
            {str(s.get("stimulus_name", "")) for s in target_sweeps}
        ),
    }

    meta_json = _json_sanitize(meta)
    meta_path.write_text(
        json.dumps(meta_json, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[ok] wrote {npz_path.name} and {meta_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and cache mixed Allen mouse/human electrophysiology testcases."
        )
    )
    parser.add_argument(
        "--cell-ids",
        nargs="+",
        type=int,
        default=DEFAULT_CELL_IDS,
        help="Specimen IDs to cache (default: curated mixed panel).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/data"),
        help="Output directory for .npz/.json cache files.",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=None,
        help=(
            "AllenSDK manifest path. Defaults to "
            "<allen-cache-dir>/cell_types_manifest.json."
        ),
    )
    parser.add_argument(
        "--allen-cache-dir",
        type=Path,
        default=Path("tests/data/allen_cache"),
        help=(
            "Directory for AllenSDK raw cache files (cells.json, ephys_features.csv, "
            "specimen_* folders)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate cache files even when they already exist.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deprecated alias for --force.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    force = bool(args.force or args.overwrite)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    allen_cache_dir = args.allen_cache_dir
    allen_cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = args.manifest_file
    if manifest_file is None:
        manifest_file = allen_cache_dir / "cell_types_manifest.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    cell_ids = [int(cid) for cid in args.cell_ids]

    pending_ids = []
    for specimen_id in cell_ids:
        npz_path = output_dir / f"cell_{specimen_id}_data.npz"
        meta_path = output_dir / f"cell_{specimen_id}_meta.json"
        if force or not (npz_path.exists() and meta_path.exists()):
            pending_ids.append(specimen_id)

    print(
        f"Requested {len(cell_ids)} specimens in {output_dir} "
        f"(manifest: {manifest_file})"
    )
    if not pending_ids:
        print("All requested cache files already exist. Nothing to do.")
        return

    print(f"Need to fetch/regenerate {len(pending_ids)} specimen(s).")

    _install_numpy_compat_shim()
    _install_h5py_compat_shim()
    from allensdk.core.cell_types_cache import CellTypesCache  # type: ignore[import-not-found]

    ctc = CellTypesCache(manifest_file=str(manifest_file))
    all_cells = ctc.get_cells(require_reconstruction=False)
    cells_by_id = {int(c["id"]): c for c in all_cells}
    features_by_specimen = _build_ephys_feature_lookup(ctc)

    for specimen_id in pending_ids:
        if specimen_id not in cells_by_id:
            raise ValueError(f"Specimen ID {specimen_id} not found in Allen cell list")
        _save_cell_cache(
            cell_types_cache=ctc,
            cell=cells_by_id[specimen_id],
            feature_lookup=features_by_specimen,
            output_dir=output_dir,
            overwrite=force,
        )

    print("Done.")


if __name__ == "__main__":
    main()
