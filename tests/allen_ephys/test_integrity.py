import warnings

import pytest


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
CELL_FEATURE_KEYS = {
    "long_squares": [
        "rheobase_i",
        "fi_fit_slope",
        "sag",
        "vm_for_sag",
        "input_resistance",
        "tau",
        "v_baseline",
    ],
    "short_squares": ["stimulus_amplitude", "common_amp_sweeps"],
    "ramps": ["spiking_sweeps"],
}


def _assert_no_warnings(caught, context: str):
    assert not caught, f"Unexpected warning during {context}: {caught[0].message}"


@pytest.mark.parametrize(
    "sweep_kind",
    ["long_spiking", "long_no_spike", "long_hyperpol", "ramp", "short_square"],
)
def test_process_spikes_representative_sweeps(specimen_case, sweep_kind):
    sweep = specimen_case.representative_sweep(sweep_kind)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sweep.process_spikes()
    _assert_no_warnings(caught, f"process_spikes/{sweep_kind}")


@pytest.mark.parametrize("feature_name", BASE_SWEEP_FEATURES)
def test_sweep_feature_long_square(specimen_case, feature_name):
    sweep = specimen_case.representative_sweep("long_spiking")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = sweep.sweep_feature(feature_name, allow_missing=True)
    _assert_no_warnings(caught, f"long_square/{feature_name}")
    assert value is not None


@pytest.mark.parametrize("feature_name", BASE_SWEEP_FEATURES)
def test_sweep_feature_short_square(specimen_case, feature_name):
    sweep = specimen_case.representative_sweep("short_square")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = sweep.sweep_feature(feature_name, allow_missing=True)
    _assert_no_warnings(caught, f"short_square/{feature_name}")
    assert value is not None


@pytest.mark.parametrize("feature_name", BASE_SWEEP_FEATURES)
def test_sweep_feature_ramp(specimen_case, feature_name):
    sweep = specimen_case.representative_sweep("ramp")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = sweep.sweep_feature(feature_name, allow_missing=True)
    _assert_no_warnings(caught, f"ramp/{feature_name}")
    assert value is not None


def test_spike_detection_features_long_square(specimen_case):
    sweep = specimen_case.representative_sweep("long_spiking")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sweep.process_spikes()
    _assert_no_warnings(caught, "long_square/process_spikes")

    for feature_name in sweep.spike_feature_keys():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            values = sweep.spike_feature(feature_name, include_clipped=True)
        _assert_no_warnings(caught, f"long_square/spike_feature/{feature_name}")
        assert values is not None


def test_spike_detection_features_short_square(specimen_case):
    sweep = specimen_case.representative_sweep("short_square")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sweep.process_spikes()
    _assert_no_warnings(caught, "short_square/process_spikes")

    for feature_name in sweep.spike_feature_keys():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            values = sweep.spike_feature(feature_name, include_clipped=True)
        _assert_no_warnings(caught, f"short_square/spike_feature/{feature_name}")
        assert values is not None


def test_spike_detection_features_ramp(specimen_case):
    sweep = specimen_case.representative_sweep("ramp")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sweep.process_spikes()
    _assert_no_warnings(caught, "ramp/process_spikes")

    for feature_name in sweep.spike_feature_keys():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            values = sweep.spike_feature(feature_name, include_clipped=True)
        _assert_no_warnings(caught, f"ramp/spike_feature/{feature_name}")
        assert values is not None


@pytest.mark.parametrize("feature_name", HYPERPOL_FEATURES)
def test_hyperpolarizing_long_square_features(specimen_case, feature_name):
    sweep = specimen_case.representative_sweep("long_hyperpol")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = sweep.sweep_feature(feature_name, allow_missing=True)
    _assert_no_warnings(caught, f"long_hyperpol/{feature_name}")
    assert value is not None


def test_pause_metrics(specimen_case):
    sweep = specimen_case.representative_sweep("long_spiking")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sweep.process_spikes()
        value = sweep.pause_metrics()
    _assert_no_warnings(caught, "pause_metrics")
    assert value is not None


def test_burst_metrics(specimen_case):
    sweep = specimen_case.representative_sweep("long_spiking")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sweep.process_spikes()
        value = sweep.burst_metrics()
    _assert_no_warnings(caught, "burst_metrics")
    assert value is not None


def test_delay_metrics(specimen_case):
    sweep = specimen_case.representative_sweep("long_spiking")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sweep.process_spikes()
        value = sweep.delay_metrics()
    _assert_no_warnings(caught, "delay_metrics")
    assert value is not None


@pytest.mark.parametrize("key", CELL_FEATURE_KEYS["long_squares"])
def test_cell_feature_long_squares(specimen_case, key):
    features = specimen_case.compute_cell_features()
    assert features["long_squares"][key] is not None


@pytest.mark.parametrize("key", CELL_FEATURE_KEYS["short_squares"])
def test_cell_feature_short_squares(specimen_case, key):
    features = specimen_case.compute_cell_features()
    assert features["short_squares"][key] is not None


@pytest.mark.parametrize("key", CELL_FEATURE_KEYS["ramps"])
def test_cell_feature_ramps(specimen_case, key):
    features = specimen_case.compute_cell_features()
    assert features["ramps"][key] is not None
