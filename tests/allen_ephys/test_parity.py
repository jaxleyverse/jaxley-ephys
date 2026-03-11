from __future__ import annotations

import math

import numpy as np
import pytest
from allen_ephys.ephys_extractor import EphysSweepFeatureExtractor


pytestmark = pytest.mark.filterwarnings("error")


STIMULUS_KINDS = ("long_square", "short_square", "ramp")
SPIKE_FEATURE_NAMES = (
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

# These keys are present in Allen DB tables but are not outputs of
# cell_extractor_for_nwb / EphysCellFeatureExtractor.
SKIPPED_CELL_FEATURES = {
    "electrode_0_pa": "electrode QC metadata",
    "seal_gohm": "seal QC metadata",
    "id": "database row id",
    "specimen_id": "database specimen id",
    "thumbnail_sweep_id": "database presentation metadata",
    "has_burst": "derived DB annotation not exported by extractor",
    "has_delay": "derived DB annotation not exported by extractor",
    "has_pause": "derived DB annotation not exported by extractor",
    "input_resistance_mohm": "no direct output from EphysCellFeatureExtractor",
}

# Only these sweep-level values have direct ground truth in ephys_sweeps.json.
SWEEP_PARITY_FEATURES = {
    "num_spikes": "spike count per sweep",
    "peak_deflection": "sweep peak deflection",
}

# Sweep features exposed by EphysSweepFeatureExtractor that are intentionally
# skipped because Allen DB does not expose per-sweep ground truth for them in
# ephys_sweeps.json.
SKIPPED_SWEEP_FEATURES = {
    "adapt": "no per-sweep DB target",
    "avg_rate": "no per-sweep DB target",
    "first_isi": "no per-sweep DB target",
    "isi_cv": "no per-sweep DB target",
    "latency": "no per-sweep DB target",
    "mean_isi": "no per-sweep DB target",
    "median_isi": "no per-sweep DB target",
}

DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3
FEATURE_TOLERANCES = {
    "input_resistance_mohm": (1.0, 1e-2),
    "f_i_curve_slope": (1e-4, 1e-2),
    "rheobase_sweep_number": (0.0, 0.0),
    "rheobase_sweep_id": (0.0, 0.0),
}


_INDEX_SPIKE_COUNT_CACHE: dict[tuple[int, int], int] = {}


def _as_float_or_nan(value) -> float:
    if value is None:
        return float("nan")
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(cast):
        return float("nan")
    return cast


def _feature_tolerance(feature_name: str) -> tuple[float, float]:
    return FEATURE_TOLERANCES.get(feature_name, (DEFAULT_ATOL, DEFAULT_RTOL))


def _assert_close(
    specimen_id: int,
    feature_name: str,
    expected: float,
    computed: float,
    *,
    atol: float,
    rtol: float,
) -> None:
    diff = abs(computed - expected)
    limit = atol + rtol * abs(expected)
    assert diff <= limit, (
        f"specimen={specimen_id} feature={feature_name} "
        f"expected={expected} computed={computed} abs_diff={diff} "
        f"atol={atol} rtol={rtol}"
    )


def _count_spikes_in_index_window(specimen_case, record) -> int:
    cache_key = (specimen_case.specimen_id, record.sweep_number)
    if cache_key in _INDEX_SPIKE_COUNT_CACHE:
        return _INDEX_SPIKE_COUNT_CACHE[cache_key]

    data = specimen_case._dataset().get_sweep(record.sweep_number)
    dt = 1.0 / float(data["sampling_rate"])
    index_start, index_end = data["index_range"]

    t, v, i, _, _ = specimen_case.trace_for(record.sweep_number)
    sweep = EphysSweepFeatureExtractor(
        t,
        v,
        i,
        start=float(index_start) * dt,
        end=float(index_end) * dt,
        id=record.sweep_number,
    )
    sweep.process_spikes()
    count = len(sweep.spikes())
    _INDEX_SPIKE_COUNT_CACHE[cache_key] = count
    return count


def _mean_first_spike_feature(sweeps: list[dict], feature_name: str) -> float:
    values = []
    for sweep in sweeps:
        spikes = sweep.get("spikes") or []
        if not spikes:
            continue
        value = _as_float_or_nan(spikes[0].get(feature_name))
        if math.isnan(value):
            continue
        values.append(value)
    if not values:
        return float("nan")
    return float(np.mean(values))


def _mean_first_spike_feature_with_fallback(
    sweeps: list[dict],
    feature_name: str,
    fallback_feature_name: str,
) -> float:
    values = []
    for sweep in sweeps:
        spikes = sweep.get("spikes") or []
        if not spikes:
            continue

        spike = spikes[0]
        value = _as_float_or_nan(spike.get(feature_name))
        if math.isnan(value):
            value = _as_float_or_nan(spike.get(fallback_feature_name))
        if math.isnan(value):
            continue
        values.append(value)

    if not values:
        return float("nan")
    return float(np.mean(values))


def _first_spike_feature(sweep_dict: dict, feature_name: str) -> float:
    spikes = sweep_dict.get("spikes") or []
    if not spikes:
        return float("nan")
    return _as_float_or_nan(spikes[0].get(feature_name))


def _short_square_threshold_i(short_squares: dict) -> float:
    sweeps = short_squares.get("common_amp_sweeps") or []
    stim_amp = _as_float_or_nan(short_squares.get("stimulus_amplitude"))

    values = []
    for sweep in sweeps:
        spikes = sweep.get("spikes") or []
        if not spikes:
            continue
        value = _as_float_or_nan(spikes[0].get("threshold_i"))
        if math.isnan(value):
            continue
        values.append(value)

    if not values:
        return stim_amp

    if len(sweeps) <= 2:
        return float(np.mean(values))

    nonzero = [value for value in values if not np.isclose(value, 0.0)]
    if nonzero:
        return float(np.mean(nonzero))

    return stim_amp


def _lookup_spiking_sweep_by_db_sweep_id(specimen_case, sweeps: list[dict], db_sweep_id: int):
    for sweep in sweeps:
        sweep_number_raw = sweep.get("id")
        if sweep_number_raw is None:
            continue
        for record in specimen_case.sweep_records:
            if (
                str(record.sweep_number) == str(sweep_number_raw)
                and record.sweep_id == db_sweep_id
            ):
                return sweep
    return None


def _computed_cell_feature_dict(specimen_case) -> dict[str, float]:
    """Return computed feature map from vendored Allen extractor outputs."""
    features = specimen_case.compute_cell_features()
    feature_row = specimen_case.feature_row
    long_squares = features["long_squares"]
    short_squares = features["short_squares"]
    ramps = features["ramps"]

    rheobase_sweep = long_squares.get("rheobase_sweep") or {}
    long_spiking_sweeps = long_squares.get("spiking_sweeps") or []

    thumbnail_sweep = None
    thumbnail_sweep_id = _as_float_or_nan(feature_row.get("thumbnail_sweep_id"))
    if not math.isnan(thumbnail_sweep_id):
        thumbnail_sweep = _lookup_spiking_sweep_by_db_sweep_id(
            specimen_case,
            long_spiking_sweeps,
            int(thumbnail_sweep_id),
        )
    if thumbnail_sweep is None:
        thumbnail_sweep = rheobase_sweep

    rheobase_sweep_number = _as_float_or_nan(rheobase_sweep.get("id"))
    rheobase_sweep_id = float("nan")
    if not math.isnan(rheobase_sweep_number):
        sweep_number = int(rheobase_sweep_number)
        for record in specimen_case.sweep_records:
            if record.sweep_number == sweep_number:
                rheobase_sweep_id = float(record.sweep_id)
                break

    out = {
        "adaptation": _as_float_or_nan(thumbnail_sweep.get("adapt")),
        "avg_isi": _as_float_or_nan(thumbnail_sweep.get("mean_isi")),
        "f_i_curve_slope": _as_float_or_nan(long_squares.get("fi_fit_slope")),
        "ri": _as_float_or_nan(long_squares.get("input_resistance")),
        "latency": _as_float_or_nan(thumbnail_sweep.get("latency")),
        "rheobase_sweep_number": rheobase_sweep_number,
        "rheobase_sweep_id": rheobase_sweep_id,
        "sag": _as_float_or_nan(long_squares.get("sag")),
        "tau": _as_float_or_nan(long_squares.get("tau")),
        "vm_for_sag": _as_float_or_nan(long_squares.get("vm_for_sag")),
        "vrest": _as_float_or_nan(long_squares.get("v_baseline")),
        "threshold_i_long_square": _as_float_or_nan(long_squares.get("rheobase_i")),
        "threshold_i_short_square": _short_square_threshold_i(short_squares),
        "threshold_i_ramp": _mean_first_spike_feature(
            ramps.get("spiking_sweeps") or [],
            "threshold_i",
        ),
    }

    for feature_name in SPIKE_FEATURE_NAMES:
        if feature_name == "threshold_i":
            out[f"{feature_name}_short_square"] = _short_square_threshold_i(
                short_squares
            )
            out[f"{feature_name}_ramp"] = _mean_first_spike_feature(
                ramps.get("spiking_sweeps") or [],
                feature_name,
            )
            continue

        long_value = _first_spike_feature(rheobase_sweep, feature_name)
        out[f"{feature_name}_long_square"] = long_value

        if feature_name in {"slow_trough_t", "slow_trough_v"}:
            out[f"{feature_name}_short_square"] = _mean_first_spike_feature_with_fallback(
                short_squares.get("common_amp_sweeps") or [],
                feature_name,
                fallback_feature_name=feature_name.replace("slow_", ""),
            )
        else:
            out[f"{feature_name}_short_square"] = _mean_first_spike_feature(
                short_squares.get("common_amp_sweeps") or [],
                feature_name,
            )

        out[f"{feature_name}_ramp"] = _mean_first_spike_feature(
            ramps.get("spiking_sweeps") or [],
            feature_name,
        )

    return out


def _expected_cell_feature_dict(specimen_case) -> dict[str, float]:
    """Return expected feature map from cached Allen DB row values."""
    out: dict[str, float] = {}
    for feature_name, value in specimen_case.feature_row.items():
        if feature_name in SKIPPED_CELL_FEATURES:
            continue
        out[feature_name] = _as_float_or_nan(value)
    return out


def _representative_record(specimen_case, kind: str):
    if kind == "long_square":
        records = specimen_case.records(kind="long_square", has_spikes=True)
        if records:
            return records[0]
        return specimen_case.require_record(kind="long_square")
    if kind == "short_square":
        records = specimen_case.records(kind="short_square", has_spikes=True)
        if records:
            return records[0]
        return specimen_case.require_record(kind="short_square")
    if kind == "ramp":
        records = specimen_case.records(kind="ramp", has_spikes=True)
        if records:
            return records[0]
        return specimen_case.require_record(kind="ramp")
    raise ValueError(f"Unknown kind: {kind}")


def _assert_sweep_parity(specimen_case, record, sweep) -> None:
    specimen_id = specimen_case.specimen_id

    if record.num_spikes is not None:
        computed_num_spikes = _count_spikes_in_index_window(specimen_case, record)
        assert computed_num_spikes == record.num_spikes, (
            f"specimen={specimen_id} sweep_number={record.sweep_number} "
            f"feature=num_spikes expected={record.num_spikes} "
            f"computed={computed_num_spikes}"
        )

    if record.peak_deflection is not None:
        computed_peak_deflect = sweep.sweep_feature("peak_deflect", allow_missing=True)
        assert isinstance(computed_peak_deflect, tuple), (
            f"specimen={specimen_id} sweep_number={record.sweep_number} "
            "feature=peak_deflection computed value missing"
        )
        computed_peak = float(computed_peak_deflect[0])
        expected_peak = float(record.peak_deflection)
        _assert_close(
            specimen_id,
            "peak_deflection",
            expected_peak,
            computed_peak,
            atol=1e-3,
            rtol=1e-4,
        )


def test_parity_documents_skipped_features() -> None:
    assert SKIPPED_CELL_FEATURES
    assert SWEEP_PARITY_FEATURES
    assert SKIPPED_SWEEP_FEATURES


def test_cell_feature_parity_against_allen_db(specimen_case) -> None:
    db_feature_values = _expected_cell_feature_dict(specimen_case)
    extracted_feature_values = _computed_cell_feature_dict(specimen_case)

    for feature_name in sorted(db_feature_values):
        assert feature_name in extracted_feature_values, (
            f"specimen={specimen_case.specimen_id} feature={feature_name} "
            "not produced by computed cell feature mapping"
        )

        db_value = db_feature_values[feature_name]
        extracted_value = extracted_feature_values[feature_name]
        if math.isnan(db_value):
            continue
        assert not math.isnan(extracted_value), (
            f"specimen={specimen_case.specimen_id} feature={feature_name} "
            f"expected={db_value} computed=nan"
        )

        atol, rtol = _feature_tolerance(feature_name)
        _assert_close(
            specimen_case.specimen_id,
            feature_name,
            db_value,
            extracted_value,
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize("kind", STIMULUS_KINDS)
def test_sweepset_feature_parity_against_allen_db(specimen_case, kind: str) -> None:
    records = specimen_case.records(kind=kind)
    if not records:
        pytest.skip(f"specimen {specimen_case.specimen_id} has no {kind} sweeps")

    sweepset = specimen_case.make_sweep_set(records)
    sweepset.process_spikes()

    record_by_sweep_number = {record.sweep_number: record for record in records}
    for sweep in sweepset.sweeps():
        record = record_by_sweep_number[int(sweep.id)]
        _assert_sweep_parity(specimen_case, record, sweep)


@pytest.mark.parametrize("kind", STIMULUS_KINDS)
def test_single_sweep_feature_parity_against_allen_db(specimen_case, kind: str) -> None:
    record = _representative_record(specimen_case, kind)
    sweep = specimen_case.make_sweep(record)
    sweep.process_spikes()
    _assert_sweep_parity(specimen_case, record, sweep)
