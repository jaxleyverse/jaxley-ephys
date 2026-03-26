from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pandas import DataFrame

from allen_ephys.ephys_extractor import EphysSweepSetFeatureExtractor
from tests.utils.jax_feature_contract import SWEEP_FEATURE_NAMES


CACHE_JSON = "ephys_features.json"
CACHE_NPZ = "ephys_features.npz"

EXTRACTOR_KEYS = (
    "long_square",
    "long_square_spiking",
    "long_square_subthreshold",
    "long_square_subthreshold_membrane_property",
    "short_square",
    "ramp",
    "ramp_all",
)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _cache_paths(specimen_case) -> tuple[Path, Path]:
    base = specimen_case.specimen_dir
    return base / CACHE_JSON, base / CACHE_NPZ


def _raw_sweep_map(cell) -> dict[int, object]:
    sweep_by_id: dict[int, object] = {}
    for ext in (cell._long_squares_ext, cell._short_squares_ext, cell._ramps_ext):
        for sweep in ext.sweeps():
            if sweep.id is None:
                raise ValueError("cannot cache features for sweep without id")
            sweep_by_id[int(sweep.id)] = sweep
    return sweep_by_id


def _extractor_sweep_ids(sweep_extractors: dict[str, object]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for key in EXTRACTOR_KEYS:
        ext = sweep_extractors[key]
        out[key] = [int(sweep.id) for sweep in ext.sweeps()]
    return out


def write_cached_allen_features(
    specimen_case,
    cell_features: dict[str, Any],
    sweep_extractors: dict[str, object],
) -> None:
    json_path, npz_path = _cache_paths(specimen_case)

    unique_sweeps: dict[int, object] = {}
    for ext in sweep_extractors.values():
        for sweep in ext.sweeps():
            unique_sweeps[int(sweep.id)] = sweep

    npz_arrays: dict[str, np.ndarray] = {}
    sweeps_payload: dict[str, dict[str, Any]] = {}

    for sweep_id, sweep in unique_sweeps.items():
        for feature_name in SWEEP_FEATURE_NAMES:
            if feature_name == "peak_deflect_index":
                continue
            sweep.sweep_feature(feature_name, allow_missing=True)

        sweep_feature_names = sorted(sweep._sweep_features.keys())
        tuple_feature_names: list[str] = []

        for name in sweep_feature_names:
            value = sweep._sweep_features[name]
            if isinstance(value, tuple):
                tuple_feature_names.append(name)
                npz_arrays[f"sweep__{sweep_id}__{name}__0"] = np.asarray(value[0])
                npz_arrays[f"sweep__{sweep_id}__{name}__1"] = np.asarray(value[1])
            else:
                npz_arrays[f"sweep__{sweep_id}__{name}"] = np.asarray(value)

        spikes_df = getattr(sweep, "_spikes_df", DataFrame())
        spike_columns = spikes_df.columns.tolist()
        spike_records = spikes_df.to_dict("records")

        sweeps_payload[str(sweep_id)] = {
            "sweep_feature_names": sweep_feature_names,
            "tuple_sweep_feature_names": tuple_feature_names,
            "spike_columns": spike_columns,
            "spike_records": _to_jsonable(spike_records),
            "affected_by_clipping": list(getattr(sweep, "_affected_by_clipping", [])),
        }

    payload = {
        "cell_features": _to_jsonable(cell_features),
        "extractor_sweep_ids": _extractor_sweep_ids(sweep_extractors),
        "sweeps": sweeps_payload,
    }

    json_path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    np.savez(npz_path, **npz_arrays)


def load_cached_allen_features(specimen_case, cell) -> dict[str, Any]:
    json_path, npz_path = _cache_paths(specimen_case)
    if not json_path.exists() or not npz_path.exists():
        raise FileNotFoundError(f"missing cached features: {json_path} / {npz_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    npz = np.load(npz_path)
    sweep_by_id = _raw_sweep_map(cell)

    for sweep_id_str, sweep_payload in payload["sweeps"].items():
        sweep_id = int(sweep_id_str)
        if sweep_id not in sweep_by_id:
            raise KeyError(f"cached sweep id {sweep_id} not found in extractor")

        sweep = sweep_by_id[sweep_id]
        names = sweep_payload["sweep_feature_names"]
        tuple_names = set(sweep_payload["tuple_sweep_feature_names"])

        sweep_features: dict[str, Any] = {}
        for name in names:
            if name in tuple_names:
                v0 = np.asarray(npz[f"sweep__{sweep_id}__{name}__0"]).item()
                v1 = np.asarray(npz[f"sweep__{sweep_id}__{name}__1"]).item()
                sweep_features[name] = (v0, v1)
            else:
                sweep_features[name] = np.asarray(
                    npz[f"sweep__{sweep_id}__{name}"]
                ).item()

        sweep._sweep_features = sweep_features
        spike_columns = sweep_payload["spike_columns"]
        spike_records = sweep_payload["spike_records"]
        if spike_records:
            sweep._spikes_df = DataFrame.from_records(
                spike_records, columns=spike_columns
            )
        else:
            sweep._spikes_df = DataFrame(columns=spike_columns)
        sweep._affected_by_clipping = list(sweep_payload["affected_by_clipping"])

    ids = payload["extractor_sweep_ids"]

    def _build_ext(key: str):
        return EphysSweepSetFeatureExtractor.from_sweeps(
            [sweep_by_id[int(sid)] for sid in ids[key]]
        )

    cell._long_squares_ext = _build_ext("long_square")
    cell._spiking_long_squares_ext = _build_ext("long_square_spiking")
    cell._subthreshold_long_squares_ext = _build_ext("long_square_subthreshold")
    cell._subthreshold_membrane_property_ext = _build_ext(
        "long_square_subthreshold_membrane_property"
    )
    cell._short_squares_ext = _build_ext("short_square")
    cell._ramps_ext = _build_ext("ramp")
    cell._all_ramps_ext = _build_ext("ramp_all")

    return payload["cell_features"]
