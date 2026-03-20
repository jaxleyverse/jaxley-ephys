from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxley_ephys.events import compute_dvdt, detect_events
from jaxley_ephys.features import compute_sweep_features
from jaxley_ephys.features.long_square.sweepset import (
    SWEEPSET_FEATURE_FUNCTIONS as LONG_SWEEPSET_FUNCTIONS,
)
from jaxley_ephys.features.long_square.sweepset import (
    compute_sweepset_features as compute_long,
)
from jaxley_ephys.features.ramp.sweepset import (
    SWEEPSET_FEATURE_FUNCTIONS as RAMP_SWEEPSET_FUNCTIONS,
)
from jaxley_ephys.features.ramp.sweepset import (
    compute_sweepset_features as compute_ramp,
)
from jaxley_ephys.features.short_square.sweepset import (
    SWEEPSET_FEATURE_FUNCTIONS as SHORT_SWEEPSET_FUNCTIONS,
)
from jaxley_ephys.features.short_square.sweepset import (
    compute_sweepset_features as compute_short,
)
from tests.utils.jax_feature_contract import (
    LONG_SQUARE_SWEEPSET_FEATURE_NAMES,
    RAMP_SWEEPSET_FEATURE_NAMES,
    SHORT_SQUARE_SWEEPSET_FEATURE_NAMES,
)
from tests.utils.simulate_testdata import simulate_sweepset


def _sweep_feature_vectors(t_set, v_set, i_set):
    vectors = {}
    for t, v, i in zip(t_set, v_set, i_set):
        ev = detect_events(t, v, i, jnp.asarray(0.02), jnp.asarray(0.1), max_spikes=32)
        dvdt = compute_dvdt(t, v)
        sweep_map = compute_sweep_features(t, v, i, ev, dvdt)
        for name, value in sweep_map.items():
            vectors.setdefault(name, []).append(value)
    return {name: jnp.stack(values) for name, values in vectors.items()}


def _assert_feature_map(computed, feature_names):
    for feature_name in feature_names:
        assert feature_name in computed
        assert computed[feature_name].shape == ()


@pytest.fixture(scope="module")
def synthetic_sweep_feature_vectors():
    t_set, v_set, i_set = simulate_sweepset(
        num_sweeps=6,
        dt=1e-4,
        t_max=0.12,
        i_range=(-0.3, 0.9),
    )
    return _sweep_feature_vectors(t_set, v_set, i_set)


def test_long_sweepset_features_compute_on_synthetic_vectors(
    synthetic_sweep_feature_vectors,
):
    vectors = synthetic_sweep_feature_vectors
    computed = compute_long(vectors)
    _assert_feature_map(computed, LONG_SQUARE_SWEEPSET_FEATURE_NAMES)


def test_ramp_sweepset_features_compute_on_synthetic_vectors(
    synthetic_sweep_feature_vectors,
):
    vectors = synthetic_sweep_feature_vectors
    computed = compute_ramp(vectors)
    _assert_feature_map(computed, RAMP_SWEEPSET_FEATURE_NAMES)


def test_short_sweepset_features_compute_on_synthetic_vectors(
    synthetic_sweep_feature_vectors,
):
    vectors = synthetic_sweep_feature_vectors
    computed = compute_short(vectors)
    _assert_feature_map(computed, SHORT_SQUARE_SWEEPSET_FEATURE_NAMES)


def test_sweepset_feature_registries_match_contract():
    for feature_name in LONG_SQUARE_SWEEPSET_FEATURE_NAMES:
        assert feature_name in LONG_SWEEPSET_FUNCTIONS
    for feature_name in RAMP_SWEEPSET_FEATURE_NAMES:
        assert feature_name in RAMP_SWEEPSET_FUNCTIONS
    for feature_name in SHORT_SQUARE_SWEEPSET_FEATURE_NAMES:
        assert feature_name in SHORT_SWEEPSET_FUNCTIONS


def test_compute_sweepset_features_jittable(synthetic_sweep_feature_vectors):
    vectors = synthetic_sweep_feature_vectors

    out_long = jax.jit(compute_long)(vectors)
    out_ramp = jax.jit(compute_ramp)(vectors)
    out_short = jax.jit(compute_short)(vectors)

    _assert_feature_map(out_long, LONG_SQUARE_SWEEPSET_FEATURE_NAMES)
    _assert_feature_map(out_ramp, RAMP_SWEEPSET_FEATURE_NAMES)
    _assert_feature_map(out_short, SHORT_SQUARE_SWEEPSET_FEATURE_NAMES)
