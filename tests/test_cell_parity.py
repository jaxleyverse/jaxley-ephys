"""Cell-level features vs Allen's ``EphysCellFeatureExtractor`` (vendored reference).

The JAX side aggregates the sweep features computed in the ``results`` fixture,
so these tests also depend on sweep-level parity.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxley_ephys.features.long_square.sweepset import (
    compute_sweepset_features as long_square_sweepset,
)
from jaxley_ephys.features.short_square.sweepset import (
    compute_sweepset_features as short_square_sweepset,
)
from tests.utils.allen_reference import SWEEP_FEATURES
from tests.utils.parity import mismatches

LONG_SQUARE_FEATURES = (
    "rheobase_index",
    "rheobase_i",
    "fi_fit_slope",
    "sag",
    "vm_for_sag",
    "input_resistance",
    "tau",
    "v_baseline",
)


def _vectors(results, kind: str) -> dict:
    if any(res["error"] for res in results[kind]):
        pytest.fail(
            f"JAX failed on some {kind} sweeps (see test_parity.py::test_spike_counts)"
        )
    return {
        name: jnp.asarray([res["features"][name] for res in results[kind]])
        for name in SWEEP_FEATURES
    }


@pytest.mark.parametrize("feature", LONG_SQUARE_FEATURES)
def test_long_square_cell_feature(reference, results, feature):
    vectors = _vectors(results, "long_square")
    computed = float(long_square_sweepset(vectors, feature_names=(feature,))[feature])
    expected = reference["cell"]["long_square"][feature]
    assert not mismatches(feature, [expected], [computed])[0], (
        f"long-square {feature}: allen {expected}, jax {computed}"
    )


def test_short_square_stimulus_amplitude(reference, results):
    vectors = _vectors(results, "short_square")
    computed = float(
        short_square_sweepset(vectors, feature_names=("stimulus_amplitude",))[
            "stimulus_amplitude"
        ]
    )
    expected = reference["cell"]["short_square"]["stimulus_amplitude"]
    assert not mismatches("stimulus_amplitude", [expected], [computed])[0], (
        f"short-square stimulus amplitude: allen {expected}, jax {computed}"
    )
