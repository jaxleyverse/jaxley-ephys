from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.allen_ephys.ephys_extractor import (
    EphysSweepFeatureExtractor,
    EphysSweepSetFeatureExtractor,
)


def cached_paths(
    specimen_id: int, data_dir: Path | str = Path("tests/data")
) -> tuple[Path, Path]:
    root = Path(data_dir)
    return (
        root / f"cell_{specimen_id}_data.npz",
        root / f"cell_{specimen_id}_meta.json",
    )


def load_cached_specimen(
    specimen_id: int, data_dir: Path | str = Path("tests/data")
) -> tuple[dict, np.lib.npyio.NpzFile]:
    data_path, meta_path = cached_paths(specimen_id, data_dir)
    if not data_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Missing cached Allen testcase files. Run: "
            f"uv run python tests/utils/load_allen_testcases.py --cell-ids {specimen_id}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arr = np.load(data_path, allow_pickle=True)
    return meta, arr


def sweep_from_index(arr: np.lib.npyio.NpzFile, idx: int) -> EphysSweepFeatureExtractor:
    t = np.asarray(arr["t"][idx], dtype=float)
    v = np.asarray(arr["v"][idx], dtype=float)
    i = np.asarray(arr["i"][idx], dtype=float)
    start = arr["stimulus_start_times"][idx]
    dur = arr["stimulus_durations"][idx]
    start_t = float(start) if np.isfinite(start) else None
    end_t = float(start + dur) if (np.isfinite(start) and np.isfinite(dur)) else None
    sweep_number = int(arr["sweep_numbers"][idx])
    return EphysSweepFeatureExtractor(
        t, v, i, start=start_t, end=end_t, id=sweep_number
    )


def group_indices_by_stimulus(arr: np.lib.npyio.NpzFile) -> dict[str, list[int]]:
    groups = {"long_square": [], "short_square": [], "ramp": []}
    for idx in range(len(arr["sweep_numbers"])):
        stim_name = str(arr["stimulus_names"][idx])
        if "Long Square" in stim_name:
            groups["long_square"].append(idx)
        if "Short Square" in stim_name:
            groups["short_square"].append(idx)
        if "Ramp" in stim_name:
            groups["ramp"].append(idx)
    return groups


def make_sweep_set(
    arr: np.lib.npyio.NpzFile, indices: list[int]
) -> EphysSweepSetFeatureExtractor:
    t_set = []
    v_set = []
    i_set = []
    start = []
    end = []
    id_set = []
    for idx in indices:
        t = np.asarray(arr["t"][idx], dtype=float)
        v = np.asarray(arr["v"][idx], dtype=float)
        i = np.asarray(arr["i"][idx], dtype=float)
        st = arr["stimulus_start_times"][idx]
        dur = arr["stimulus_durations"][idx]
        start_t = float(st) if np.isfinite(st) else None
        end_t = float(st + dur) if (np.isfinite(st) and np.isfinite(dur)) else None
        t_set.append(t)
        v_set.append(v)
        i_set.append(i)
        start.append(start_t)
        end.append(end_t)
        id_set.append(int(arr["sweep_numbers"][idx]))
    return EphysSweepSetFeatureExtractor(
        t_set, v_set, i_set, start=start, end=end, id_set=id_set
    )
