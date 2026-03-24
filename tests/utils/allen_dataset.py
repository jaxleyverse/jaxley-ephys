from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


CANONICAL_STIMULUS_NAMES = {
    "long_square": "Long Square",
    "short_square": "Short Square",
    "ramp": "Ramp",
}
STIMULUS_KIND_BY_NAME = {v: k for k, v in CANONICAL_STIMULUS_NAMES.items()}


CURATED_SPECIMEN_IDS = [
    519832676,
    525011903,
    537256313,
    528642047,
    469801138,
]


def install_allen_compat_shims() -> None:
    if not hasattr(np, "VisibleDeprecationWarning"):
        np.VisibleDeprecationWarning = Warning  # type: ignore[attr-defined]

    import h5py  # type: ignore[import-not-found]

    if not hasattr(h5py.Dataset, "value"):
        h5py.Dataset.value = property(lambda self: self[()])  # type: ignore[attr-defined]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(cast):
        return None
    return cast


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class SweepRecord:
    specimen_id: int
    sweep_number: int
    sweep_id: int
    stimulus_name: str
    stimulus_start_time: float | None
    stimulus_duration: float | None
    stimulus_absolute_amplitude: float | None
    num_spikes: int | None
    peak_deflection: float | None

    @property
    def kind(self) -> str | None:
        return STIMULUS_KIND_BY_NAME.get(self.stimulus_name)

    @property
    def has_spikes(self) -> bool:
        return (self.num_spikes or 0) > 0

    @property
    def is_hyperpol(self) -> bool:
        amp = self.stimulus_absolute_amplitude
        return amp is not None and amp < 0


@dataclass
class SpecimenCase:
    specimen_id: int
    data_root: Path
    sweep_records: list[SweepRecord]
    feature_row: dict[str, str]

    _trace_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]
    _cell_features_cache: dict[str, Any] | None = None

    @property
    def specimen_dir(self) -> Path:
        return self.data_root / f"specimen_{self.specimen_id}"

    @property
    def nwb_path(self) -> Path:
        return self.specimen_dir / "ephys.nwb"

    def _dataset(self):
        install_allen_compat_shims()
        from allensdk.core.nwb_data_set import NwbDataSet  # type: ignore[import-not-found]

        return NwbDataSet(str(self.nwb_path))

    def records(
        self,
        *,
        kind: str | None = None,
        has_spikes: bool | None = None,
        is_hyperpol: bool | None = None,
    ) -> list[SweepRecord]:
        out = []
        for rec in self.sweep_records:
            if kind is not None and rec.kind != kind:
                continue
            if has_spikes is not None and rec.has_spikes != has_spikes:
                continue
            if is_hyperpol is not None and rec.is_hyperpol != is_hyperpol:
                continue
            out.append(rec)
        return out

    def require_record(
        self,
        *,
        kind: str,
        has_spikes: bool | None = None,
        is_hyperpol: bool | None = None,
    ) -> SweepRecord:
        records = self.records(
            kind=kind, has_spikes=has_spikes, is_hyperpol=is_hyperpol
        )
        if not records:
            raise ValueError(
                f"specimen {self.specimen_id}: no sweep for "
                f"kind={kind}, has_spikes={has_spikes}, is_hyperpol={is_hyperpol}"
            )
        return records[0]

    def representative_sweep(self, kind: str):
        if kind == "long_spiking":
            record = self.require_record(kind="long_square", has_spikes=True)
        elif kind == "long_no_spike":
            record = self.require_record(kind="long_square", has_spikes=False)
        elif kind == "long_hyperpol":
            record = self.require_record(kind="long_square", is_hyperpol=True)
        elif kind == "short_square":
            short_spiking = self.records(kind="short_square", has_spikes=True)
            record = (
                short_spiking[0]
                if short_spiking
                else self.require_record(kind="short_square")
            )
        elif kind == "ramp":
            ramp_spiking = self.records(kind="ramp", has_spikes=True)
            record = (
                ramp_spiking[0] if ramp_spiking else self.require_record(kind="ramp")
            )
        else:
            raise ValueError(f"Unknown representative kind: {kind}")
        return self.make_sweep(record)

    def trace_for(
        self, sweep_number: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        if sweep_number in self._trace_cache:
            return self._trace_cache[sweep_number]

        rec = next(r for r in self.sweep_records if r.sweep_number == sweep_number)
        data = self._dataset().get_sweep(sweep_number)

        v = np.asarray(data["response"], dtype=float) * 1e3
        i = np.asarray(data["stimulus"], dtype=float) * 1e12
        dt = 1.0 / float(data["sampling_rate"])
        t = np.arange(v.size, dtype=float) * dt

        idx0, idx1 = data["index_range"]
        start = rec.stimulus_start_time
        duration = rec.stimulus_duration
        if start is None:
            start = float(idx0) * dt
        if duration is None:
            end = float(idx1) * dt
        else:
            end = start + duration

        out = (t, v, i, start, end)
        self._trace_cache[sweep_number] = out
        return out

    def make_sweep(self, record: SweepRecord):
        from allen_ephys.ephys_extractor import EphysSweepFeatureExtractor

        t, v, i, start, end = self.trace_for(record.sweep_number)
        return EphysSweepFeatureExtractor(
            t,
            v,
            i,
            start=start,
            end=end,
            id=record.sweep_number,
        )

    def make_sweep_set(self, records: list[SweepRecord]):
        from allen_ephys.ephys_extractor import EphysSweepSetFeatureExtractor

        t_set = []
        v_set = []
        i_set = []
        start = []
        end = []
        ids = []

        for rec in records:
            t, v, i, st, en = self.trace_for(rec.sweep_number)
            t_set.append(t)
            v_set.append(v)
            i_set.append(i)
            start.append(st)
            end.append(en)
            ids.append(rec.sweep_number)

        return EphysSweepSetFeatureExtractor(
            t_set,
            v_set,
            i_set,
            start=start,
            end=end,
            id_set=ids,
        )

    def build_cell_extractor_for_nwb(self):
        from allen_ephys.ephys_extractor import cell_extractor_for_nwb

        dataset = self._dataset()
        ramps = [r.sweep_number for r in self.records(kind="ramp")]
        short = [r.sweep_number for r in self.records(kind="short_square")]
        long = [r.sweep_number for r in self.records(kind="long_square")]
        return cell_extractor_for_nwb(dataset, ramps, short, long)

    def compute_cell_features(self) -> dict[str, Any]:
        if self._cell_features_cache is None:
            cell = self.build_cell_extractor_for_nwb()
            cell.process()
            self._cell_features_cache = cell.as_dict()
        return self._cell_features_cache


def cached_specimen_ids(data_root: Path = Path("tests/testdata/allen_cache")) -> list[int]:
    discovered = []
    for path in sorted(data_root.glob("specimen_*")):
        if not path.is_dir():
            continue
        try:
            specimen_id = int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if (path / "ephys.nwb").exists() and (path / "ephys_sweeps.json").exists():
            discovered.append(specimen_id)

    curated = [sid for sid in CURATED_SPECIMEN_IDS if sid in discovered]
    extras = [sid for sid in discovered if sid not in curated]
    return curated + extras


@lru_cache(maxsize=1)
def _feature_rows(
    data_root: str = "tests/testdata/allen_cache",
) -> dict[int, dict[str, str]]:
    features_path = Path(data_root) / "ephys_features.csv"
    out: dict[int, dict[str, str]] = {}
    with features_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            specimen_raw = row.get("specimen_id")
            specimen_id = _safe_int(specimen_raw)
            if specimen_id is None:
                continue
            out[specimen_id] = row
    return out


def _load_sweep_records(specimen_id: int, data_root: Path) -> list[SweepRecord]:
    sweeps_path = data_root / f"specimen_{specimen_id}" / "ephys_sweeps.json"
    rows = json.loads(sweeps_path.read_text(encoding="utf-8"))

    out: list[SweepRecord] = []
    for row in rows:
        stimulus_name = str(row.get("stimulus_name") or "")
        if stimulus_name not in STIMULUS_KIND_BY_NAME:
            continue

        sweep_number = _safe_int(row.get("sweep_number"))
        sweep_id = _safe_int(row.get("id"))
        if sweep_number is None or sweep_id is None:
            continue

        out.append(
            SweepRecord(
                specimen_id=specimen_id,
                sweep_number=sweep_number,
                sweep_id=sweep_id,
                stimulus_name=stimulus_name,
                stimulus_start_time=_safe_float(row.get("stimulus_start_time")),
                stimulus_duration=_safe_float(row.get("stimulus_duration")),
                stimulus_absolute_amplitude=_safe_float(
                    row.get("stimulus_absolute_amplitude")
                ),
                num_spikes=_safe_int(row.get("num_spikes")),
                peak_deflection=_safe_float(row.get("peak_deflection")),
            )
        )

    out.sort(key=lambda r: r.sweep_number)
    return out


@lru_cache(maxsize=None)
def load_specimen_case(
    specimen_id: int,
    data_root: str = "tests/testdata/allen_cache",
) -> SpecimenCase:
    root = Path(data_root)
    records = _load_sweep_records(specimen_id, root)
    if not records:
        raise ValueError(f"No canonical sweeps found for specimen {specimen_id}")

    feature_rows = _feature_rows(data_root)
    if specimen_id not in feature_rows:
        raise ValueError(
            f"No ephys feature row found for specimen {specimen_id} in {root / 'ephys_features.csv'}"
        )

    return SpecimenCase(
        specimen_id=specimen_id,
        data_root=root,
        sweep_records=records,
        feature_row=feature_rows[specimen_id],
        _trace_cache={},
    )
