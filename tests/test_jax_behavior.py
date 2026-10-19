"""JAX-specific behaviour: jit, vmap, gradients, input dtypes and overflow."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxley_ephys import detect_spikes
from tests.utils.jax_runner import extract, extract_batch
from tests.utils.simulate_testdata import simulate_sweep, simulate_sweepset

START, END = 0.02, 0.08


@pytest.fixture(scope="module")
def sweep():
    return simulate_sweep(dt=2.5e-5, t_max=0.1, stim_delay=START, stim_dur=END - START)


def _assert_same(a, b):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(
            np.asarray(x, float), np.asarray(y, float), equal_nan=True
        )


def test_jit_matches_eager(sweep):
    eager = extract(*sweep, START, END)
    jitted = jax.jit(extract)(*sweep, START, END)
    assert int(eager[0]) > 0
    _assert_same(eager, jitted)


def test_vmap_matches_loop():
    t, v, i = simulate_sweepset(
        num_sweeps=4,
        dt=2.5e-5,
        t_max=0.1,
        stim_delay=START,
        stim_dur=END - START,
        i_range=(-0.3, 1.0),
    )
    start, end = jnp.full(4, START), jnp.full(4, END)
    batched = extract_batch(t, v, i, start, end)
    for k in range(4):
        single = extract(t[k], v[k], i[k], START, END)
        _assert_same(jax.tree_util.tree_map(lambda x: x[k], batched), single)


def test_gradients_are_finite(sweep):
    t, v, i = sweep

    def mean_peak_voltage(v):
        return jnp.nanmean(extract(t, v, i, START, END)[2]["peak_v"])

    grad = jax.grad(mean_peak_voltage)(v)
    assert bool(jnp.all(jnp.isfinite(grad))) and bool(jnp.any(grad != 0))


def test_float32_inputs_are_accepted(sweep):
    as32 = [np.asarray(x, np.float32) for x in sweep]
    reference = extract(*sweep, START, END)
    _assert_same(extract(*as32, START, END)[0], reference[0])


def test_overflow_raises_when_not_traced(sweep):
    with pytest.raises(ValueError, match="max_spikes"):
        extract(*sweep, START, END, max_spikes=1)


def test_mismatched_shapes_raise(sweep):
    t, v, i = sweep
    with pytest.raises(ValueError):
        extract(t, v[:-1], i, START, END)


def test_filter_above_nyquist_raises_like_allen(sweep):
    t, v, i = sweep  # 40 kHz sampling: Nyquist frequency 20 kHz
    with pytest.raises(ValueError, match="Nyquist"):
        detect_spikes(t, v, i, START, END, filter=25.0)
    traced = jax.jit(lambda t: detect_spikes(t, v, i, START, END, filter=25.0).dvdt)(t)
    assert np.all(np.isnan(traced))


def test_empty_window_gives_nan_instead_of_raising(sweep):
    """Allen raises on an empty analysis window (argmin of an empty slice)."""
    features = extract(*sweep, START, START)[3]
    assert int(features["peak_deflect_index"]) == -1
    for name in ("peak_deflect", "sag", "tau"):
        assert np.isnan(features[name]), name
