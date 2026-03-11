from __future__ import annotations

import pytest

from allen_ephys.ephys_extractor import (
    EphysCellFeatureExtractor,
    EphysSweepFeatureExtractor,
    EphysSweepSetFeatureExtractor,
    cell_extractor_for_nwb,
)


BASE_SWEEP_FEATURES = [
    "avg_rate",
    "adapt",
    "latency",
    "isi_cv",
    "mean_isi",
    "median_isi",
    "first_isi",
    "v_baseline",
    "peak_deflect",
    "stim_amp",
]
HYPERPOL_FEATURES = ["tau", "sag"]


def _require_spiking_record(specimen_case, kind: str):
    rows = specimen_case.records(kind=kind, has_spikes=True)
    if not rows:
        pytest.skip(f"specimen {specimen_case.specimen_id} has no spiking {kind} sweep")
    return rows[0]


def test_single_sweep_feature_extractor_construction(specimen_case):
    record = _require_spiking_record(specimen_case, "long_square")
    sweep = specimen_case.make_sweep(record)
    assert isinstance(sweep, EphysSweepFeatureExtractor)


@pytest.mark.parametrize("kind", ["long_square", "short_square", "ramp"])
def test_process_spikes_spiking_sweeps(specimen_case, kind: str):
    record = _require_spiking_record(specimen_case, kind)
    sweep = specimen_case.make_sweep(record)
    sweep.process_spikes()
    assert sweep.spikes() is not None


@pytest.mark.parametrize("kind", ["long_square", "short_square", "ramp"])
@pytest.mark.parametrize("feature_name", BASE_SWEEP_FEATURES)
def test_base_sweep_features_on_spiking_sweeps(
    specimen_case, kind: str, feature_name: str
):
    record = _require_spiking_record(specimen_case, kind)
    sweep = specimen_case.make_sweep(record)
    value = sweep.sweep_feature(feature_name, allow_missing=True)
    assert value is not None


@pytest.mark.parametrize("kind", ["long_square", "short_square", "ramp"])
def test_spike_feature_access_on_spiking_sweeps(specimen_case, kind: str):
    record = _require_spiking_record(specimen_case, kind)
    sweep = specimen_case.make_sweep(record)
    sweep.process_spikes()
    for feature_name in sweep.spike_feature_keys():
        values = sweep.spike_feature(feature_name, include_clipped=True)
        assert values is not None


@pytest.mark.parametrize("feature_name", HYPERPOL_FEATURES)
def test_hyperpolarizing_long_square_features(specimen_case, feature_name: str):
    record = specimen_case.require_record(kind="long_square", is_hyperpol=True)
    sweep = specimen_case.make_sweep(record)
    value = sweep.sweep_feature(feature_name, allow_missing=True)
    assert value is not None


@pytest.mark.parametrize("kind", ["long_square", "short_square", "ramp"])
def test_sweep_set_feature_extractor_construction(specimen_case, kind: str):
    records = specimen_case.records(kind=kind)
    ext = specimen_case.make_sweep_set(records)
    assert isinstance(ext, EphysSweepSetFeatureExtractor)
    ext.process_spikes()
    assert ext.sweeps() is not None


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
