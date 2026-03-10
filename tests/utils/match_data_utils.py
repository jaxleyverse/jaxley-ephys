from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


EXCLUDED_DB_KEYS = {
    "electrode_0_pa",
    "seal_gohm",
    "ri",
    "thumbnail_sweep_id",
    "id",
    "specimen_id",
    "rheobase_sweep_id",
    "has_burst",
    "has_delay",
    "has_pause",
}


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


DEFAULT_TOLERANCE = Tolerance(atol=1e-3, rtol=5e-2)
FEATURE_TOLERANCES: dict[str, Tolerance] = {
    "rheobase_sweep_number": Tolerance(atol=0.0, rtol=0.0),
    "input_resistance_mohm": Tolerance(atol=1.0, rtol=1e-1),
    "f_i_curve_slope": Tolerance(atol=1e-4, rtol=1e-1),
}


def _as_float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(cast):
        return float("nan")
    return cast


def _mean_first_spike_feature(sweeps: list[dict[str, Any]], name: str) -> float:
    values: list[float] = []
    for sweep in sweeps:
        avg_rate = _as_float_or_nan(sweep.get("avg_rate"))
        spikes = sweep.get("spikes", [])
        if np.isnan(avg_rate) or avg_rate <= 0 or not spikes:
            continue

        value = _as_float_or_nan(spikes[0].get(name))
        if np.isnan(value):
            continue

        values.append(value)

    if not values:
        return float("nan")
    return float(np.mean(values))


def flatten_for_db(cell_features: dict[str, Any]) -> dict[str, float]:
    """Flatten compute_cell_features() output to DB-style scalar keys."""
    long_squares = cell_features["long_squares"]
    hero = long_squares.get("hero_sweep") or {}

    hero_spikes = hero.get("spikes", [])
    hero_spike_0 = hero_spikes[0] if hero_spikes else {}

    rheobase_sweep = long_squares.get("rheobase_sweep", {})
    rheobase_spikes = rheobase_sweep.get("spikes", [])
    rheobase_spike_0 = rheobase_spikes[0] if rheobase_spikes else {}

    ramps_spiking = cell_features.get("ramps", {}).get("spiking_sweeps", [])
    short_common_sweeps = cell_features.get("short_squares", {}).get(
        "common_amp_sweeps", []
    )

    spike_feature_names = [
        "fast_trough_t",
        "fast_trough_v",
        "slow_trough_t",
        "slow_trough_v",
        "threshold_i",
        "threshold_t",
        "threshold_v",
        "peak_t",
        "peak_v",
        "trough_t",
        "trough_v",
        "upstroke_downstroke_ratio",
    ]

    ramps_spike_0 = {
        key: _mean_first_spike_feature(ramps_spiking, key)
        for key in spike_feature_names
    }
    short_spike_0 = {
        key: _mean_first_spike_feature(short_common_sweeps, key)
        for key in spike_feature_names
    }

    short_threshold_i = short_spike_0["threshold_i"]
    if np.isnan(short_threshold_i):
        short_threshold_i = _as_float_or_nan(
            cell_features.get("short_squares", {}).get("stimulus_amplitude")
        )

    out: dict[str, float] = {
        "adaptation": _as_float_or_nan(hero.get("adapt")),
        "avg_isi": _as_float_or_nan(hero.get("mean_isi")),
        "f_i_curve_slope": _as_float_or_nan(long_squares.get("fi_fit_slope")),
        "input_resistance_mohm": _as_float_or_nan(long_squares.get("input_resistance")),
        "latency": _as_float_or_nan(hero.get("latency")),
        "rheobase_sweep_number": _as_float_or_nan(
            long_squares.get("rheobase_sweep", {}).get("id")
        ),
        "sag": _as_float_or_nan(long_squares.get("sag")),
        "tau": _as_float_or_nan(long_squares.get("tau")),
        "vm_for_sag": _as_float_or_nan(long_squares.get("vm_for_sag")),
        "vrest": _as_float_or_nan(long_squares.get("v_baseline")),
        "threshold_i_long_square": _as_float_or_nan(long_squares.get("rheobase_i")),
        "threshold_i_short_square": short_threshold_i,
    }

    for name in spike_feature_names:
        long_value = _as_float_or_nan(rheobase_spike_0.get(name))
        if np.isnan(long_value):
            long_value = _as_float_or_nan(hero_spike_0.get(name))

        out[f"{name}_long_square"] = long_value
        out[f"{name}_ramp"] = ramps_spike_0[name]
        out[f"{name}_short_square"] = short_spike_0[name]

    return out


def expected_db_feature_dict(meta: dict[str, Any]) -> dict[str, float]:
    raw = meta["ephys_features"]
    out: dict[str, float] = {}
    for key, value in raw.items():
        if key in EXCLUDED_DB_KEYS:
            continue

        cast = _as_float_or_nan(value)
        if np.isnan(cast):
            continue

        out[key] = cast
    return out
