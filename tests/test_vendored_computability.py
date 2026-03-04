import warnings

import numpy as np
import pytest

from tests.allen_ephys.ephys_extractor import (
    EphysCellFeatureExtractor,
    _short_step_stim_amp,
    _step_stim_amp,
)
from tests.utils.cached_testcase import (
    group_indices_by_stimulus,
    load_cached_specimen,
    make_sweep_set,
    sweep_from_index,
)


SPECIMEN_ID = 528642047
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


@pytest.fixture(scope="module")
def cached_specimen():
    try:
        return load_cached_specimen(SPECIMEN_ID)
    except FileNotFoundError as exc:
        pytest.fail(str(exc))


@pytest.fixture(scope="module")
def representative_sweeps(cached_specimen):
    meta, arr = cached_specimen
    groups = group_indices_by_stimulus(arr)
    if not all(groups.values()):
        pytest.fail(f"Missing stimulus group in cached data: {groups}")

    sweep_meta = {int(item["sweep_number"]): item for item in meta["selected_sweeps"]}

    def pick_index(indices, predicate):
        for idx in indices:
            sweep_number = int(arr["sweep_numbers"][idx])
            if predicate(sweep_meta[sweep_number]):
                return idx
        return None

    long_idx_spiking = pick_index(
        groups["long_square"], lambda sm: int(sm.get("num_spikes") or 0) > 2
    )
    long_idx_no_spike = pick_index(
        groups["long_square"], lambda sm: int(sm.get("num_spikes") or 0) == 0
    )
    long_idx_hyperpol = pick_index(
        groups["long_square"],
        lambda sm: float(sm.get("stimulus_absolute_amplitude") or np.nan) < 0,
    )
    ramp_idx = pick_index(groups["ramp"], lambda sm: True)
    short_idx = pick_index(groups["short_square"], lambda sm: True)

    if None in (
        long_idx_spiking,
        long_idx_no_spike,
        long_idx_hyperpol,
        ramp_idx,
        short_idx,
    ):
        pytest.fail("Could not find representative sweeps for all required checks.")

    sweeps = {
        "long_spiking": sweep_from_index(arr, long_idx_spiking),
        "long_no_spike": sweep_from_index(arr, long_idx_no_spike),
        "long_hyperpol": sweep_from_index(arr, long_idx_hyperpol),
        "ramp": sweep_from_index(arr, ramp_idx),
        "short_square": sweep_from_index(arr, short_idx),
    }

    for name, sweep in sweeps.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sweep.process_spikes()
        assert not caught, (
            f"Unexpected warning during process_spikes on {name}: {caught[0].message}"
        )

    return sweeps


@pytest.mark.parametrize(
    "sweep_kind",
    ["long_spiking", "long_no_spike", "long_hyperpol", "ramp", "short_square"],
)
def test_sweep_core_features_computable_all_stimuli(representative_sweeps, sweep_kind):
    sweep = representative_sweeps[sweep_kind]
    for feature_name in BASE_SWEEP_FEATURES:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = sweep.sweep_feature(feature_name, allow_missing=True)
        assert not caught, (
            f"Unexpected warning for sweep '{sweep_kind}' feature '{feature_name}': {caught[0].message}"
        )
        assert value is not None


@pytest.mark.parametrize(
    "sweep_kind",
    ["long_spiking", "long_no_spike", "long_hyperpol", "ramp", "short_square"],
)
def test_spike_feature_arrays_computable_all_stimuli(representative_sweeps, sweep_kind):
    sweep = representative_sweeps[sweep_kind]
    for feature_name in sweep.spike_feature_keys():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            values = sweep.spike_feature(feature_name, include_clipped=True)
        assert not caught, (
            f"Unexpected warning for spike feature '{feature_name}' on '{sweep_kind}': {caught[0].message}"
        )
        assert values is not None


@pytest.mark.parametrize("feature_name", ["tau", "sag"])
def test_hyperpolarizing_long_square_features_computable(
    representative_sweeps, feature_name
):
    sweep = representative_sweeps["long_hyperpol"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = sweep.sweep_feature(feature_name, allow_missing=True)
    assert not caught, (
        f"Unexpected warning for feature '{feature_name}': {caught[0].message}"
    )
    assert value is not None


@pytest.mark.parametrize(
    "method_name", ["pause_metrics", "burst_metrics", "delay_metrics"]
)
def test_spike_train_metrics_computable(representative_sweeps, method_name):
    sweep = representative_sweeps["long_spiking"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = getattr(sweep, method_name)()
    assert not caught, f"Unexpected warning for '{method_name}': {caught[0].message}"
    assert value is not None


@pytest.fixture(scope="module")
def cell_features_dict(cached_specimen):
    _, arr = cached_specimen
    groups = group_indices_by_stimulus(arr)
    if not all(groups.values()):
        pytest.fail(f"Missing stimulus group in cached data: {groups}")

    ramps_ext = make_sweep_set(arr, groups["ramp"])
    short_ext = make_sweep_set(arr, groups["short_square"])
    long_ext = make_sweep_set(arr, groups["long_square"])

    for sweep in ramps_ext.sweeps():
        sweep.set_stimulus_amplitude_calculator(_step_stim_amp)
    for sweep in long_ext.sweeps():
        sweep.set_stimulus_amplitude_calculator(_step_stim_amp)
    for sweep in short_ext.sweeps():
        sweep.set_stimulus_amplitude_calculator(_short_step_stim_amp)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cell = EphysCellFeatureExtractor(ramps_ext, short_ext, long_ext)
        cell.process()
        features = cell.as_dict()
    assert not caught, (
        f"Unexpected warning during cell feature extraction: {caught[0].message}"
    )
    return features


@pytest.mark.parametrize(
    "section,key",
    [
        ("long_squares", "rheobase_i"),
        ("long_squares", "fi_fit_slope"),
        ("long_squares", "sag"),
        ("long_squares", "vm_for_sag"),
        ("long_squares", "input_resistance"),
        ("long_squares", "tau"),
        ("long_squares", "v_baseline"),
        ("short_squares", "stimulus_amplitude"),
        ("short_squares", "common_amp_sweeps"),
        ("ramps", "spiking_sweeps"),
    ],
)
def test_cell_features_computable(cell_features_dict, section, key):
    assert section in cell_features_dict
    assert key in cell_features_dict[section]
    assert cell_features_dict[section][key] is not None
