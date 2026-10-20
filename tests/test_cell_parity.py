"""Sweepset and cell features vs Allen's ``EphysCellFeatureExtractor`` and
``extract_cell_features`` (vendored reference).

The JAX side runs the whole cell pipeline (``cell_results`` fixture): it estimates
the short-square detection parameters itself and detects all spikes again, so
these tests also depend on spike- and sweep-level parity.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxley_ephys import long_square_features
from tests.utils.allen_reference import DB_SPIKE_FEATURES
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
    "hero_index",
)
DB_FEATURES = (
    "adaptation",
    "avg_isi",
    "latency",
    "f_i_curve_slope",
    "ri",
    "sag",
    "tau",
    "vm_for_sag",
    "vrest",
    *(
        f"{name}_{kind}"
        for kind in ("long_square", "short_square", "ramp")
        for name in DB_SPIKE_FEATURES
    ),
)


def _assert_matches(name: str, expected, actual) -> None:
    assert not mismatches(name, [expected], [float(actual)])[0], (
        f"{name}: allen {expected}, jax {float(actual)}"
    )


def _selected(sweeps, kind: str, mask) -> list[int]:
    return [int(s.id) for s, m in zip(sweeps[kind], mask) if m]


@pytest.mark.parametrize("feature", LONG_SQUARE_FEATURES)
def test_long_square_feature(reference, cell_results, feature):
    expected = reference["cell"]["long_square"][feature]
    _assert_matches(feature, expected, cell_results["long_squares"][feature])


def test_short_square_detection_parameters(reference, cell_results):
    for name in ("dv_cutoff", "thresh_frac"):
        expected = reference["cell"]["short_square"][name]
        _assert_matches(name, expected, cell_results["short_squares"][name])


def test_short_square_common_amplitude(reference, sweeps, cell_results):
    expected = reference["cell"]["short_square"]
    actual = cell_results["short_squares"]
    _assert_matches(
        "stimulus_amplitude",
        expected["stimulus_amplitude"],
        actual["stimulus_amplitude"],
    )
    common = _selected(sweeps, "short_square", actual["common_amp"])
    assert common == expected["common_amp_sweeps"]


def test_spiking_ramps(reference, sweeps, cell_results):
    spiking = _selected(sweeps, "ramp", cell_results["ramps"]["spiking"])
    assert spiking == reference["cell"]["ramp"]["spiking_sweeps"]


@pytest.mark.parametrize("kind", ["short_square", "ramp"])
def test_mean_first_spike(reference, cell_results, kind):
    actual = cell_results[f"{kind}s"]["mean_spike_0"]
    for name, expected in reference["cell"][kind]["mean_spike_0"].items():
        _assert_matches(name, expected, actual[name])


@pytest.mark.parametrize("feature", DB_FEATURES)
def test_cell_feature(reference, cell_results, feature):
    expected = reference["cell"]["record"][feature]
    _assert_matches(feature, expected, cell_results["cell"][feature])


def test_padded_and_batched_sweepsets_match(cell_results):
    """Padding sweeps (``valid``) and vmapping over cells give the same features."""
    ls = cell_results["long_squares"]
    sweeps, first = ls["sweeps"], ls["first_spikes"]
    single = long_square_features(sweeps, first)

    def pad(x):  # two junk sweeps at the end
        return jnp.concatenate([jnp.asarray(x), jnp.asarray(x)[:2] * 0 + 7])

    padded = jax.tree.map(pad, (sweeps, first))
    valid = jnp.arange(len(sweeps["avg_rate"]) + 2) < len(sweeps["avg_rate"])
    batched = jax.vmap(long_square_features)(
        *jax.tree.map(lambda x: jnp.stack([x, x]), padded), jnp.stack([valid, valid])
    )
    for name in LONG_SQUARE_FEATURES:
        np.testing.assert_allclose(
            batched[name], single[name], rtol=1e-12, equal_nan=True
        )
