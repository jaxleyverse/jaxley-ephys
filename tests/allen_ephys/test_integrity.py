from __future__ import annotations

import math

import numpy as np
import pytest

from allen_ephys.ephys_extractor import (
    EphysCellFeatureExtractor,
    EphysSweepFeatureExtractor,
    EphysSweepSetFeatureExtractor,
)
from tests.utils.allen_reference import SWEEP_FEATURES as SWEEP_FEATURE_NAMES
from tests.utils.parity import SPIKE_COLUMNS as SPIKE_FEATURE_NAMES


STIMULUS_KINDS = ("long_square", "short_square", "ramp")
ALLEN_SWEEP_FEATURE_NAMES = tuple(
    name for name in SWEEP_FEATURE_NAMES if name != "peak_deflect_index"
)


def _compute_all_sweep_features(sweep: EphysSweepFeatureExtractor) -> None:
    for feature_name in sweep.sweep_feature_keys():
        sweep.sweep_feature(feature_name, allow_missing=True)


def _compute_all_spike_features(sweep: EphysSweepFeatureExtractor) -> None:
    for feature_name in sweep.spike_feature_keys():
        sweep.spike_feature(feature_name, include_clipped=True)


def _targeted_sweep_feature_value(sweep: EphysSweepFeatureExtractor, feature_name: str):
    if feature_name in {"peak_deflect", "peak_deflect_index"}:
        value = sweep.sweep_feature("peak_deflect", allow_missing=True)
        if not isinstance(value, tuple):
            raise TypeError("Allen peak_deflect feature is expected to be a tuple")
        if feature_name == "peak_deflect":
            return value[0]
        return value[1]
    return sweep.sweep_feature(feature_name, allow_missing=True)


def _require_records(specimen_case, kind: str):
    records = specimen_case.records(kind=kind)
    if not records:
        pytest.skip(f"specimen {specimen_case.specimen_id} has no {kind} sweeps")
    return records


def _require_spiking_record(specimen_case, kind: str):
    rows = specimen_case.records(kind=kind, has_spikes=True)
    if not rows:
        pytest.skip(f"specimen {specimen_case.specimen_id} has no spiking {kind} sweep")
    return rows[0]


def test_single_sweep_feature_extractor_construction(specimen_case):
    record = _require_spiking_record(specimen_case, "long_square")
    sweep = specimen_case.make_sweep(record)
    assert isinstance(sweep, EphysSweepFeatureExtractor)


@pytest.mark.parametrize("kind", STIMULUS_KINDS)
def test_single_sweep_features_compute_without_errors(specimen_case, kind: str):
    record = _require_spiking_record(specimen_case, kind)
    sweep = specimen_case.make_sweep(record)
    sweep.process_spikes()
    _compute_all_sweep_features(sweep)
    _compute_all_spike_features(sweep)


@pytest.mark.parametrize("kind", STIMULUS_KINDS)
def test_sweep_set_feature_extractor_construction(specimen_case, kind: str):
    records = _require_records(specimen_case, kind)
    ext = specimen_case.make_sweep_set(records)
    assert isinstance(ext, EphysSweepSetFeatureExtractor)


@pytest.mark.parametrize("kind", STIMULUS_KINDS)
def test_sweep_set_features_compute_without_errors(specimen_case, kind: str):
    records = _require_records(specimen_case, kind)
    ext = specimen_case.make_sweep_set(records)
    ext.process_spikes()

    sweeps = ext.sweeps()
    assert len(sweeps) == len(records)

    for sweep in sweeps:
        assert isinstance(sweep, EphysSweepFeatureExtractor)
        assert sweep.spikes() is not None
        _compute_all_sweep_features(sweep)
        _compute_all_spike_features(sweep)


def test_cell_extractor_for_nwb_computable(specimen_case):
    cell = specimen_case.build_cell_extractor_for_nwb()
    assert isinstance(cell, EphysCellFeatureExtractor)
    cell.process()
    out = cell.as_dict()
    assert "long_squares" in out
    assert "short_squares" in out
    assert "ramps" in out


def test_cell_features_computable(specimen_case):
    features = specimen_case.compute_cell_features()
    assert features["long_squares"]["rheobase_i"] is not None
    assert features["long_squares"]["fi_fit_slope"] is not None
    assert features["long_squares"]["input_resistance"] is not None
    assert features["long_squares"]["tau"] is not None
    assert features["short_squares"]["stimulus_amplitude"] is not None
    assert features["ramps"]["spiking_sweeps"] is not None


def test_manual_cell_extractor_computable(specimen_case):
    long_ext = specimen_case.make_sweep_set(specimen_case.records(kind="long_square"))
    short_ext = specimen_case.make_sweep_set(specimen_case.records(kind="short_square"))
    ramp_ext = specimen_case.make_sweep_set(specimen_case.records(kind="ramp"))
    cell = EphysCellFeatureExtractor(ramp_ext, short_ext, long_ext)
    cell.process()
    out = cell.as_dict()
    assert "long_squares" in out


def test_targeted_feature_reference_paths_stable(specimen_sweepset_extractors):
    long_ext = specimen_sweepset_extractors.get("long_square")
    if long_ext is not None and long_ext.sweeps():
        spiking = [s for s in long_ext.sweeps() if s.sweep_feature("avg_rate") > 0]
        if spiking:
            for feature_name in SPIKE_FEATURE_NAMES:
                spiking[0].spike_feature(feature_name, include_clipped=True)

    for ext in specimen_sweepset_extractors.values():
        sweeps = ext.sweeps()
        for sweep in sweeps:
            for feature_name in SWEEP_FEATURE_NAMES:
                _targeted_sweep_feature_value(sweep, feature_name)

        for feature_name in ALLEN_SWEEP_FEATURE_NAMES:
            ext.sweep_features(feature_name, allow_missing=True)

        peak_deflect = ext.sweep_features("peak_deflect", allow_missing=True)
        for value in peak_deflect:
            value_part, index_part = value
            float(value_part)
            int(index_part)

        spiking_sweeps = [sweep for sweep in sweeps if len(sweep.spikes()) > 0]
        if not spiking_sweeps:
            continue
        spiking_ext = EphysSweepSetFeatureExtractor.from_sweeps(spiking_sweeps)

        for feature_name in SPIKE_FEATURE_NAMES:
            if feature_name == "isi_type":
                continue
            supported = True
            for sweep in spiking_sweeps:
                if feature_name not in set(sweep.spike_feature_keys()):
                    supported = False
                    break
            if not supported:
                continue
            spiking_ext.spike_feature_averages(feature_name)


def test_stimulus_amplitude_calculator_policy(specimen_sweepset_extractors):
    long_ext = specimen_sweepset_extractors.get("long_square")
    if long_ext is not None and long_ext.sweeps():
        long_vals = [s.sweep_feature("stim_amp", allow_missing=True) for s in long_ext.sweeps()]
        assert any(np.isfinite(np.asarray(long_vals, dtype=float)))

    short_ext = specimen_sweepset_extractors.get("short_square")
    if short_ext is not None and short_ext.sweeps():
        short_vals = [s.sweep_feature("stim_amp", allow_missing=True) for s in short_ext.sweeps()]
        assert any(np.isfinite(np.asarray(short_vals, dtype=float)))

    ramp_ext = specimen_sweepset_extractors.get("ramp")
    if ramp_ext is not None and ramp_ext.sweeps():
        ramp_vals = [s.sweep_feature("stim_amp", allow_missing=True) for s in ramp_ext.sweeps()]
        assert all(math.isnan(float(v)) for v in ramp_vals)
