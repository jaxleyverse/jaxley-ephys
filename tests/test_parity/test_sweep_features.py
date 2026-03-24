from __future__ import annotations

import pytest
import jax.numpy as jnp

from jaxley_ephys.events import compute_dvdt, detect_events
from jaxley_ephys.features import compute_sweep_features
from tests.utils.jax_feature_contract import SWEEP_FEATURE_NAMES, assert_feature_equal


REPRESENTATIVE_KINDS = (
    "long_spiking",
    "short_square",
    "ramp",
    "long_hyperpol",
)

# TODO: tighten tau tolerance after tau-fit parity hardening; mixed-specimen
# checks still show O(1e-1) absolute gaps for long-hyperpolarizing sweeps.
FEATURE_TOLERANCES = {
    "tau": (1e-2, 5e-3),
    "latency": (5e-5, 5e-5),
}


def _select_representative_sweep(kind: str, specimen_sweepset_extractors):
    """Frozen parity-selection policy for sweep-level checks."""
    if kind in {"long_spiking", "long_hyperpol"}:
        ext = specimen_sweepset_extractors.get("long_square")
    else:
        ext = specimen_sweepset_extractors.get(kind)
    if ext is None:
        return None

    sweeps = ext.sweeps()
    if kind == "long_spiking":
        selected = [s for s in sweeps if s.sweep_feature("avg_rate") > 0]
    elif kind == "long_hyperpol":
        selected = [
            s for s in sweeps if s.sweep_feature("stim_amp", allow_missing=True) < 0
        ]
    elif kind in {"short_square", "ramp"}:
        selected = [s for s in sweeps if s.sweep_feature("avg_rate") > 0]
    else:
        selected = sweeps

    if not selected:
        selected = sweeps
    if not selected:
        return None
    return selected[0]


def _feature_tolerance(feature_name: str) -> tuple[float, float]:
    return FEATURE_TOLERANCES.get(feature_name, (1e-6, 1e-6))


def _expected_allen_sweep_feature(sweep, feature_name: str):
    if feature_name == "peak_deflect":
        return sweep.sweep_feature("peak_deflect", allow_missing=True)[0]
    if feature_name == "peak_deflect_index":
        return sweep.sweep_feature("peak_deflect", allow_missing=True)[1]
    return sweep.sweep_feature(feature_name, allow_missing=True)


@pytest.fixture(scope="module")
def representative_sweep_primitives(specimen_sweepset_extractors):
    out = {}
    for kind in REPRESENTATIVE_KINDS:
        sweep = _select_representative_sweep(kind, specimen_sweepset_extractors)
        if sweep is None:
            pytest.fail(f"No {kind} sweep found for sweep parity checks.")
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
        out[kind] = (sweep, ev, dvdt)
    return out


@pytest.mark.parametrize("kind", REPRESENTATIVE_KINDS)
@pytest.mark.parametrize("feature_name", SWEEP_FEATURE_NAMES)
def test_jax_sweep_feature_matches_vendored_allen(
    representative_sweep_primitives,
    kind: str,
    feature_name: str,
):
    sweep, ev, dvdt = representative_sweep_primitives[kind]
    expected = _expected_allen_sweep_feature(sweep, feature_name)

    computed_map = compute_sweep_features(
        sweep.t,
        sweep.v,
        sweep.i,
        ev,
        dvdt,
        feature_names=(feature_name,),
    )

    assert feature_name in computed_map
    atol, rtol = _feature_tolerance(feature_name)
    assert_feature_equal(expected, computed_map[feature_name], atol=atol, rtol=rtol)
