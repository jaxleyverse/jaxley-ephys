"""Seeded synthetic traces vs vendored Allen (see ``tests/utils/synthetic.py``).

They cover branches the cached recordings rarely reach: merged crossings,
re-found thresholds and peaks, dropped spikes, clipped last spikes and missing
fast troughs. ``tau`` is excluded: on these non-exponential traces the fit is
ill-posed and different local optimizers legitimately disagree (see
allen_ephys/ALLEN_EPHYS_FIXES.md, section 6).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from allen_ephys.ephys_extractor import EphysSweepFeatureExtractor
from tests.utils.jax_runner import analyse_sweep
from tests.utils.parity import allen_values, jax_values, mismatches
from tests.utils.synthetic import random_trace

N_TRACES = 40
SWEEP_FEATURES = (
    "adapt",
    "latency",
    "isi_cv",
    "mean_isi",
    "median_isi",
    "first_isi",
    "avg_rate",
    "v_baseline",
    "sag",
)
CURRENT_GAPS = pytest.mark.xfail(
    reason="current implementation lacks Allen's merge, re-find and ADP branches",
    strict=False,
)


@pytest.fixture(scope="module")
def cases():
    rng = np.random.default_rng(1234)
    out = []
    while len(out) < N_TRACES:
        t, v, i, start, end = random_trace(rng)
        allen = EphysSweepFeatureExtractor(t, v, i, start=start, end=end)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                allen.process_spikes()
                features = {
                    name: allen.sweep_feature(name, allow_missing=True)
                    for name in SWEEP_FEATURES
                }
        except Exception:  # Allen crashes on some edge cases; those traces are skipped
            continue
        out.append((allen._spikes_df, features, analyse_sweep(t, v, i, start, end)))
    return out


@CURRENT_GAPS
def test_spike_tables_match_allen(cases):
    problems = []
    for k, (table, _, jax) in enumerate(cases):
        if jax["n_spikes"] != len(table):
            problems.append(
                f"trace {k}: {len(table)} spikes in Allen, {jax['n_spikes']} in JAX"
            )
            continue
        for column in table.columns:
            bad = mismatches(
                column,
                allen_values(column, table[column]),
                jax_values(column, jax["spikes"][column]),
            )
            if bad.any():
                problems.append(f"trace {k}: {column}")
    assert not problems, f"{len(problems)} problems, e.g. {problems[:5]}"


@CURRENT_GAPS
def test_sweep_features_match_allen(cases):
    problems = [
        f"trace {k}: {name} allen {features[name]}, jax {jax['features'][name]}"
        for k, (_, features, jax) in enumerate(cases)
        for name in SWEEP_FEATURES
        if mismatches(
            name,
            [np.nan if features[name] is None else features[name]],
            [jax["features"][name]],
        )[0]
    ]
    assert not problems, f"{len(problems)} problems, e.g. {problems[:5]}"
