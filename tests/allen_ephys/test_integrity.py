from __future__ import annotations

import pytest

from allen_ephys.ephys_extractor import (
    EphysCellFeatureExtractor,
    EphysSweepFeatureExtractor,
    EphysSweepSetFeatureExtractor,
)


STIMULUS_KINDS = ("long_square", "short_square", "ramp")


def _compute_all_sweep_features(sweep: EphysSweepFeatureExtractor) -> None:
    for feature_name in sweep.sweep_feature_keys():
        sweep.sweep_feature(feature_name, allow_missing=True)


def _compute_all_spike_features(sweep: EphysSweepFeatureExtractor) -> None:
    for feature_name in sweep.spike_feature_keys():
        sweep.spike_feature(feature_name, include_clipped=True)


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
