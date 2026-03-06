from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from allen_ephys.ephys_extractor import (
    EphysCellFeatureExtractor,
    EphysSweepFeatureExtractor,
    EphysSweepSetFeatureExtractor,
    _short_step_stim_amp,
    _step_stim_amp,
)


CANONICAL_STIMULUS_NAMES = {
    "long_square": "Long Square",
    "short_square": "Short Square",
    "ramp": "Ramp",
}


def cached_specimen_ids(data_dir: Path = Path("tests/data")) -> list[int]:
    pattern = re.compile(r"cell_(\d+)_meta\.json$")
    specimen_ids: list[int] = []

    for meta_path in sorted(data_dir.glob("cell_*_meta.json")):
        match = pattern.match(meta_path.name)
        if not match:
            continue

        specimen_id = int(match.group(1))
        data_path = data_dir / f"cell_{specimen_id}_data.npz"
        if data_path.exists():
            specimen_ids.append(specimen_id)

    return specimen_ids


def _load_cached_specimen(specimen_id: int, data_dir: Path = Path("tests/data")):
    data_path = data_dir / f"cell_{specimen_id}_data.npz"
    meta_path = data_dir / f"cell_{specimen_id}_meta.json"
    if not data_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Missing cached Allen testcase files. Run: "
            f"uv run python tests/load_testdata.py --cell-ids {specimen_id}"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(data_path, allow_pickle=True) as npz:
        arr = {name: npz[name] for name in npz.files}
    return meta, arr


def _stimulus_groups(arr: dict[str, np.ndarray]) -> dict[str, list[int]]:
    groups = {"long_square": [], "short_square": [], "ramp": []}
    for idx in range(len(arr["sweep_numbers"])):
        stim_name = str(arr["stimulus_names"][idx])
        if stim_name == CANONICAL_STIMULUS_NAMES["long_square"]:
            groups["long_square"].append(idx)
        if stim_name == CANONICAL_STIMULUS_NAMES["short_square"]:
            groups["short_square"].append(idx)
        if stim_name == CANONICAL_STIMULUS_NAMES["ramp"]:
            groups["ramp"].append(idx)
    return groups


@dataclass
class SpecimenCase:
    specimen_id: int
    meta: dict[str, Any]
    arr: dict[str, np.ndarray]
    groups: dict[str, list[int]]
    sweep_meta: dict[int, dict[str, Any]]
    _representative_indices: dict[str, int] | None = None
    _cell_features: dict[str, Any] | None = None

    def _pick_index(self, indices: list[int], predicate) -> int | None:
        for idx in indices:
            sweep_number = int(self.arr["sweep_numbers"][idx])
            if predicate(self.sweep_meta[sweep_number]):
                return idx
        return None

    @property
    def representative_indices(self) -> dict[str, int]:
        if self._representative_indices is None:
            long_idx_spiking = self._pick_index(
                self.groups["long_square"],
                lambda sm: int(sm.get("num_spikes") or 0) > 2,
            )
            long_idx_no_spike = self._pick_index(
                self.groups["long_square"],
                lambda sm: int(sm.get("num_spikes") or 0) == 0,
            )
            long_idx_hyperpol = self._pick_index(
                self.groups["long_square"],
                lambda sm: float(sm.get("stimulus_absolute_amplitude") or np.nan) < 0,
            )
            ramp_idx = self._pick_index(self.groups["ramp"], lambda sm: True)
            short_idx = self._pick_index(self.groups["short_square"], lambda sm: True)

            if None in (
                long_idx_spiking,
                long_idx_no_spike,
                long_idx_hyperpol,
                ramp_idx,
                short_idx,
            ):
                raise ValueError(
                    f"Could not find representative sweeps for specimen {self.specimen_id}"
                )
            assert long_idx_spiking is not None
            assert long_idx_no_spike is not None
            assert long_idx_hyperpol is not None
            assert ramp_idx is not None
            assert short_idx is not None

            # Prefer a hyperpolarizing sweep where tau can be computed warning-free.
            for idx in self.groups["long_square"]:
                sweep_number = int(self.arr["sweep_numbers"][idx])
                sweep_meta = self.sweep_meta[sweep_number]
                amp = float(sweep_meta.get("stimulus_absolute_amplitude") or np.nan)
                if amp >= 0:
                    continue

                sweep = self.make_sweep(idx)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    value = sweep.sweep_feature("tau", allow_missing=True)

                if not caught and value is not None:
                    long_idx_hyperpol = idx
                    break

            self._representative_indices = {
                "long_spiking": int(long_idx_spiking),
                "long_no_spike": int(long_idx_no_spike),
                "long_hyperpol": int(long_idx_hyperpol),
                "ramp": int(ramp_idx),
                "short_square": int(short_idx),
            }
        return self._representative_indices

    def make_sweep(self, idx: int) -> EphysSweepFeatureExtractor:
        t = np.asarray(self.arr["t"][idx], dtype=float)
        v = np.asarray(self.arr["v"][idx], dtype=float)
        i = np.asarray(self.arr["i"][idx], dtype=float)
        start = self.arr["stimulus_start_times"][idx]
        dur = self.arr["stimulus_durations"][idx]
        start_t = float(start) if np.isfinite(start) else None
        end_t = (
            float(start + dur) if (np.isfinite(start) and np.isfinite(dur)) else None
        )
        sweep_number = int(self.arr["sweep_numbers"][idx])
        return EphysSweepFeatureExtractor(
            t, v, i, start=start_t, end=end_t, id=sweep_number
        )

    def representative_sweep(self, kind: str) -> EphysSweepFeatureExtractor:
        return self.make_sweep(self.representative_indices[kind])

    def _make_sweep_set(self, indices: list[int]) -> EphysSweepSetFeatureExtractor:
        t_set = []
        v_set = []
        i_set = []
        start = []
        end = []
        id_set = []

        for idx in indices:
            t = np.asarray(self.arr["t"][idx], dtype=float)
            v = np.asarray(self.arr["v"][idx], dtype=float)
            i = np.asarray(self.arr["i"][idx], dtype=float)
            st = self.arr["stimulus_start_times"][idx]
            dur = self.arr["stimulus_durations"][idx]
            start_t = float(st) if np.isfinite(st) else None
            end_t = float(st + dur) if (np.isfinite(st) and np.isfinite(dur)) else None

            t_set.append(t)
            v_set.append(v)
            i_set.append(i)
            start.append(start_t)
            end.append(end_t)
            id_set.append(int(self.arr["sweep_numbers"][idx]))

        return EphysSweepSetFeatureExtractor(
            t_set,
            v_set,
            i_set,
            start=start,
            end=end,
            id_set=id_set,
        )

    def compute_cell_features(self) -> dict[str, Any]:
        if self._cell_features is None:
            ramps_ext = self._make_sweep_set(self.groups["ramp"])
            short_ext = self._make_sweep_set(self.groups["short_square"])
            long_ext = self._make_sweep_set(self.groups["long_square"])

            for sweep in ramps_ext.sweeps() or []:
                sweep.set_stimulus_amplitude_calculator(_step_stim_amp)
            for sweep in long_ext.sweeps() or []:
                sweep.set_stimulus_amplitude_calculator(_step_stim_amp)
            for sweep in short_ext.sweeps() or []:
                sweep.set_stimulus_amplitude_calculator(_short_step_stim_amp)

            cell = EphysCellFeatureExtractor(ramps_ext, short_ext, long_ext)
            cell.process()
            features = cell.as_dict()

            self._cell_features = features
        assert self._cell_features is not None
        return self._cell_features


CACHED_SPECIMEN_IDS = cached_specimen_ids()


def build_specimen_cases() -> dict[int, SpecimenCase]:
    out: dict[int, SpecimenCase] = {}
    for specimen_id in CACHED_SPECIMEN_IDS:
        meta, arr = _load_cached_specimen(specimen_id)
        groups = _stimulus_groups(arr)
        if not all(groups.values()):
            raise ValueError(
                f"Missing stimulus group in cached data for {specimen_id}: {groups}"
            )
        sweep_meta = {
            int(item["sweep_number"]): item for item in meta.get("selected_sweeps", [])
        }
        out[specimen_id] = SpecimenCase(
            specimen_id=specimen_id,
            meta=meta,
            arr=arr,
            groups=groups,
            sweep_meta=sweep_meta,
        )
    return out


@pytest.fixture(scope="session")
def specimen_cases() -> dict[int, SpecimenCase]:
    if not CACHED_SPECIMEN_IDS:
        pytest.fail(
            "No cached specimens found in tests/data. Run: "
            "uv run python tests/load_testdata.py"
        )

    try:
        return build_specimen_cases()
    except ValueError as exc:
        pytest.fail(str(exc))


@pytest.fixture(params=CACHED_SPECIMEN_IDS, scope="module")
def specimen_case(request, specimen_cases: dict[int, SpecimenCase]) -> SpecimenCase:
    return specimen_cases[int(request.param)]
