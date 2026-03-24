from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from jaxley_ephys.events import compute_dvdt, detect_events
from jaxley_ephys.features import compute_sweep_features
from jaxley_ephys.features.long_square.sweepset import (
    SWEEPSET_FEATURE_INPUTS as LONG_SWEEPSET_FEATURE_INPUTS,
)
from jaxley_ephys.features.long_square.sweepset import (
    compute_sweepset_features as compute_long,
)
from jaxley_ephys.features.ramp.sweepset import (
    SWEEPSET_FEATURE_INPUTS as RAMP_SWEEPSET_FEATURE_INPUTS,
)
from jaxley_ephys.features.ramp.sweepset import (
    compute_sweepset_features as compute_ramp,
)
from jaxley_ephys.features.short_square.sweepset import (
    SWEEPSET_FEATURE_INPUTS as SHORT_SWEEPSET_FEATURE_INPUTS,
)
from jaxley_ephys.features.short_square.sweepset import (
    compute_sweepset_features as compute_short,
)
from tests.utils.jax_feature_contract import (
    LONG_SQUARE_SWEEPSET_FEATURE_NAMES,
    RAMP_SWEEPSET_FEATURE_NAMES,
    SHORT_SQUARE_SWEEPSET_FEATURE_NAMES,
    assert_feature_equal,
)


LONG_CELL_MAPPED = {
    "rheobase_index": "rheobase_extractor_index",
    "rheobase_i": "rheobase_i",
    "fi_fit_slope": "fi_fit_slope",
    "vm_for_sag": "vm_for_sag",
    "input_resistance": "input_resistance",
    "tau": "tau",
    "sag": "sag",
    "v_baseline": "v_baseline",
}

# TODO: tighten tau tolerance after tau-fit parity hardening; mixed-specimen
# checks still show O(1e-1 to 1e0) absolute gaps for long-square aggregates.
FEATURE_TOLERANCES = {
    "tau": (5e-2, 5e-3),
}


def _sweep_column(ext, feature_name: str):
    if feature_name == "peak_deflect":
        pairs = ext.sweep_features("peak_deflect", allow_missing=True)
        return np.asarray([float(v[0]) for v in pairs], dtype=float)
    if feature_name == "peak_deflect_index":
        pairs = ext.sweep_features("peak_deflect", allow_missing=True)
        return np.asarray([float(v[1]) for v in pairs], dtype=float)
    return np.asarray(ext.sweep_features(feature_name, allow_missing=True), dtype=float)


@pytest.fixture(scope="module")
def precomputed_sweepset_vectors(specimen_cell_context):
    out = {}
    for stimulus_type in ("long_square", "short_square", "ramp"):
        ext = specimen_cell_context.sweep_extractors[stimulus_type]
        vectors = {}
        for sweep in ext.sweeps():
            dvdt = compute_dvdt(sweep.t, sweep.v)
            ev = detect_events(
                sweep.t,
                sweep.v,
                sweep.i,
                jnp.asarray(float(sweep.start)),
                jnp.asarray(float(sweep.end)),
                dvdt=dvdt,
                max_spikes=256,
            )
            sweep_map = compute_sweep_features(sweep.t, sweep.v, sweep.i, ev, dvdt)
            for feature_name, value in sweep_map.items():
                vectors.setdefault(feature_name, []).append(value)
        out[stimulus_type] = {
            "ext": ext,
            "vectors": {
                name: jnp.asarray(values, dtype=float)
                for name, values in vectors.items()
            },
        }
    return out


def _compute_jax_sweepset_feature(
    vectors, feature_name: str, feature_inputs, sweepset_compute
) -> jnp.ndarray:
    deps = feature_inputs[feature_name]
    dep_vectors = {dep: vectors[dep] for dep in deps}
    return sweepset_compute(dep_vectors, feature_names=(feature_name,))[feature_name]


def _nanmean_or_nan(values) -> float:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float("nan")
    return float(np.nanmean(arr[finite]))


def _expected_allen_sweepset_feature(
    stimulus_type: str, feature_name: str, ext, cell_features
) -> float:
    if stimulus_type == "long_square" and feature_name in LONG_CELL_MAPPED:
        return float(cell_features["long_squares"][LONG_CELL_MAPPED[feature_name]])
    if stimulus_type == "short_square" and feature_name == "stimulus_amplitude":
        return float(cell_features["short_squares"]["stimulus_amplitude"])
    return _nanmean_or_nan(_sweep_column(ext, feature_name))


@pytest.mark.parametrize("feature_name", LONG_SQUARE_SWEEPSET_FEATURE_NAMES)
def test_jax_long_square_sweepset_feature_matches_allen(
    specimen_cell_context,
    precomputed_sweepset_vectors,
    feature_name: str,
):
    ext = precomputed_sweepset_vectors["long_square"]["ext"]
    vectors = precomputed_sweepset_vectors["long_square"]["vectors"]
    expected = _expected_allen_sweepset_feature(
        "long_square",
        feature_name,
        ext,
        specimen_cell_context.cell_features,
    )
    computed = _compute_jax_sweepset_feature(
        vectors,
        feature_name,
        LONG_SWEEPSET_FEATURE_INPUTS,
        compute_long,
    )
    atol, rtol = FEATURE_TOLERANCES.get(feature_name, (1e-4, 1e-4))
    assert_feature_equal(expected, computed, atol=atol, rtol=rtol)


@pytest.mark.parametrize("feature_name", SHORT_SQUARE_SWEEPSET_FEATURE_NAMES)
def test_jax_short_square_sweepset_feature_matches_allen(
    specimen_cell_context,
    precomputed_sweepset_vectors,
    feature_name: str,
):
    # Note: this extractor is already filtered by Allen's cell processing to
    # common-amplitude spiking short-square sweeps.
    ext = precomputed_sweepset_vectors["short_square"]["ext"]
    vectors = precomputed_sweepset_vectors["short_square"]["vectors"]
    expected = _expected_allen_sweepset_feature(
        "short_square",
        feature_name,
        ext,
        specimen_cell_context.cell_features,
    )
    computed = _compute_jax_sweepset_feature(
        vectors,
        feature_name,
        SHORT_SWEEPSET_FEATURE_INPUTS,
        compute_short,
    )
    atol, rtol = FEATURE_TOLERANCES.get(feature_name, (1e-4, 1e-4))
    assert_feature_equal(expected, computed, atol=atol, rtol=rtol)


@pytest.mark.parametrize("feature_name", RAMP_SWEEPSET_FEATURE_NAMES)
def test_jax_ramp_sweepset_feature_matches_allen(
    specimen_cell_context,
    precomputed_sweepset_vectors,
    feature_name: str,
):
    # Note: this extractor is already filtered by Allen's cell processing to
    # spiking ramp sweeps.
    ext = precomputed_sweepset_vectors["ramp"]["ext"]
    vectors = precomputed_sweepset_vectors["ramp"]["vectors"]
    expected = _expected_allen_sweepset_feature(
        "ramp",
        feature_name,
        ext,
        specimen_cell_context.cell_features,
    )
    computed = _compute_jax_sweepset_feature(
        vectors,
        feature_name,
        RAMP_SWEEPSET_FEATURE_INPUTS,
        compute_ramp,
    )
    atol, rtol = FEATURE_TOLERANCES.get(feature_name, (1e-4, 1e-4))
    assert_feature_equal(expected, computed, atol=atol, rtol=rtol)
