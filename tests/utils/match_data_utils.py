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


@dataclass(frozen=True)
class ComparisonEntry:
    key: str
    expected: float | None
    computed: float | None
    abs_diff: float | None
    rel_diff: float | None
    status: str
    unit_ratio: float | None


DEFAULT_TOLERANCE = Tolerance(atol=1e-3, rtol=5e-2)
FEATURE_TOLERANCES: dict[str, Tolerance] = {
    "rheobase_sweep_number": Tolerance(atol=0.0, rtol=0.0),
    "input_resistance_mohm": Tolerance(atol=1.0, rtol=1e-1),
    "f_i_curve_slope": Tolerance(atol=1e-4, rtol=1e-1),
}
UNIT_SUSPECT_RATIOS = (1e-3, 1e3, 1e-6, 1e6, 1e-9, 1e9, 1e-12, 1e12)
UNIT_RATIO_RTL = 5e-2


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(cast):
        return None
    return cast


def feature_tolerance(key: str) -> Tolerance:
    return FEATURE_TOLERANCES.get(key, DEFAULT_TOLERANCE)


def _unit_ratio(computed: float, expected: float) -> float | None:
    if expected == 0.0:
        return None
    ratio = computed / expected
    if ratio == 0.0:
        return None

    abs_ratio = abs(ratio)
    for canonical in UNIT_SUSPECT_RATIOS:
        if abs(abs_ratio - canonical) <= abs(canonical) * UNIT_RATIO_RTL:
            return canonical
    return None


def entry_status(
    key: str,
    expected: float | None,
    computed: float | None,
) -> tuple[str, float | None, float | None, float | None]:
    if expected is None:
        return "expected_missing", None, None, None
    if computed is None:
        return "computed_missing", None, None, None

    abs_diff = abs(computed - expected)
    denom = max(abs(expected), 1e-12)
    rel_diff = abs_diff / denom

    unit_ratio = _unit_ratio(computed, expected)
    if unit_ratio is not None:
        return "unit_suspect", abs_diff, rel_diff, unit_ratio

    tol = feature_tolerance(key)
    if abs_diff <= tol.atol + tol.rtol * abs(expected):
        return "match", abs_diff, rel_diff, None

    return "mismatch", abs_diff, rel_diff, None


def _flatten_comparable_features(
    cell_features: dict[str, Any],
) -> dict[str, float | None]:
    long_squares = cell_features["long_squares"]
    hero = long_squares.get("hero_sweep") or {}
    rheobase_i = safe_float(long_squares.get("rheobase_i"))

    hero_spikes = hero.get("spikes", [])
    hero_spike_0 = hero_spikes[0] if hero_spikes else {}
    rheobase_sweep = long_squares.get("rheobase_sweep", {})
    rheobase_spikes = rheobase_sweep.get("spikes", [])
    rheobase_spike_0 = rheobase_spikes[0] if rheobase_spikes else {}
    ramps_spiking = cell_features.get("ramps", {}).get("spiking_sweeps", [])
    short_common_sweeps = cell_features.get("short_squares", {}).get(
        "common_amp_sweeps", []
    )

    def mean_first_spike_feature(
        sweeps: list[dict[str, Any]], name: str
    ) -> float | None:
        values: list[float] = []
        for sweep in sweeps:
            avg_rate = safe_float(sweep.get("avg_rate"))
            spikes = sweep.get("spikes", [])
            if avg_rate is None or avg_rate <= 0 or not spikes:
                continue
            value = safe_float(spikes[0].get(name))
            if value is None:
                continue
            values.append(value)
        if not values:
            return None
        return float(np.mean(values))

    ramps_spike_0 = {
        key: mean_first_spike_feature(ramps_spiking, key)
        for key in (
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
        )
    }
    short_spike_0: dict[str, float | None] = {
        key: mean_first_spike_feature(short_common_sweeps, key)
        for key in (
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
        )
    }

    short_threshold_i = short_spike_0.get("threshold_i")
    if short_threshold_i is None:
        short_threshold_i = safe_float(
            cell_features.get("short_squares", {}).get("stimulus_amplitude")
        )

    output: dict[str, float | None] = {
        "adaptation": safe_float(hero.get("adapt")),
        "avg_isi": safe_float(hero.get("mean_isi")),
        "f_i_curve_slope": safe_float(long_squares.get("fi_fit_slope")),
        "input_resistance_mohm": safe_float(long_squares.get("input_resistance")),
        "latency": safe_float(hero.get("latency")),
        "rheobase_sweep_number": safe_float(
            long_squares.get("rheobase_sweep", {}).get("id")
        ),
        "sag": safe_float(long_squares.get("sag")),
        "tau": safe_float(long_squares.get("tau")),
        "vm_for_sag": safe_float(long_squares.get("vm_for_sag")),
        "vrest": safe_float(long_squares.get("v_baseline")),
    }

    spike_features = [
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
    for name in spike_features:
        long_value = safe_float(rheobase_spike_0.get(name))
        if long_value is None:
            long_value = safe_float(hero_spike_0.get(name))

        output[f"{name}_long_square"] = long_value
        output[f"{name}_ramp"] = safe_float(ramps_spike_0.get(name))
        output[f"{name}_short_square"] = safe_float(short_spike_0.get(name))

    output["threshold_i_short_square"] = short_threshold_i

    output["threshold_i_long_square"] = rheobase_i

    return output


def compare_with_database(specimen_case) -> dict[str, ComparisonEntry]:
    expected = specimen_case.meta["ephys_features"]
    computed = _flatten_comparable_features(specimen_case.compute_cell_features())

    entries: dict[str, ComparisonEntry] = {}
    for key, expected_value in expected.items():
        if key in EXCLUDED_DB_KEYS:
            continue

        expected_f = safe_float(expected_value)
        computed_f = safe_float(computed.get(key))
        status, abs_diff, rel_diff, unit_ratio = entry_status(
            key, expected_f, computed_f
        )

        entries[key] = ComparisonEntry(
            key=key,
            expected=expected_f,
            computed=computed_f,
            abs_diff=abs_diff,
            rel_diff=rel_diff,
            status=status,
            unit_ratio=unit_ratio,
        )

    return entries
