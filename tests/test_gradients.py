"""Gradients of the features with respect to the voltage trace.

On simulated Hodgkin-Huxley sweeps (spiking, silent, hyperpolarizing, a single
spike, a clipped last spike): ``jax.grad`` works and stays finite for every
feature, also where features are NaN; fitted quantities (``tau``, the delay) have
the gradients of their least-squares optimum; event times and the features built
from them (ISIs, latency, width, ...) have the gradients of the times between
samples, which ``interpolate=True`` returns as values; spike counts (``avg_rate``,
the f-I slope) have the gradients of a soft count; ``nan_to`` replaces exactly the
NaN values.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxley_ephys import (
    cell_features,
    detect_spikes,
    long_square_features,
    spike_features,
    sweep_features,
)
from jaxley_ephys.sweep import _soft_spike_count
from tests.utils.simulate_testdata import simulate_sweep, simulate_sweepset

START, END = 0.02, 0.08
PATTERNS = {"bursts": True, "pauses": True, "delays": True}


def _sweep(stim_amp: float):
    return simulate_sweep(
        dt=2.5e-5, t_max=0.1, stim_amp=stim_amp, stim_delay=START, stim_dur=END - START
    )


def _spike_times(t, v, i) -> np.ndarray:
    sp = detect_spikes(t, v, i, START, END, max_spikes=16)
    n = int(sp.n_spikes)
    return np.asarray(t)[np.asarray(sp.threshold[:n])], np.asarray(t)[
        np.asarray(sp.peak[:n])
    ]


@pytest.fixture(scope="module")
def sweeps() -> dict:
    """``(t, v, i, end)`` of sweeps covering the edge cases of the features."""
    spiking = _sweep(0.5)
    thresholds, peaks = _spike_times(*spiking)
    return {
        "spiking": (*spiking, END),
        "silent": (*_sweep(0.05), END),
        "hyperpolarizing": (*_sweep(-0.3), END),
        "single_spike": (*spiking, float(thresholds[0] + 0.005)),
        "clipped": (*spiking, float(peaks[1] - 1e-4)),
    }


def _features(t, i, end, nan_to=None, interpolate=False):
    """All spike and sweep features of the trace ``v`` as one flat dict."""

    def f(v):
        sp = detect_spikes(t, v, i, START, end, max_spikes=16, interpolate=interpolate)
        out = sweep_features(sp, "long_square", nan_to=nan_to, **PATTERNS)
        for name, x in spike_features(sp, nan_to=nan_to).items():
            out[f"spike_{name}"] = x
        return {
            name: x for name, x in out.items() if jnp.issubdtype(x.dtype, jnp.floating)
        }

    return f


def _directional(f, v, u, h=1e-4):
    """Central finite difference of ``f`` along ``u``."""
    return (float(f(v + h * u)) - float(f(v - h * u))) / (2 * h)


@pytest.mark.parametrize(
    "stim_amp, name",
    [(-0.3, "tau"), (0.5, "delay_tau"), (0.5, "delay_ratio"), (0.2, "delay_tau")],
)
def test_fit_gradients_match_finite_differences(stim_amp, name):
    """The fits are differentiated implicitly at their optimum (with interpolated
    times, so that the latency in the delay ratio is smooth, too)."""
    t, v, i = _sweep(stim_amp)

    def f(v):
        return _features(t, i, END, interpolate=True)(v)[name]

    u = jnp.asarray(np.random.default_rng(0).normal(size=v.shape))
    grad = jax.grad(f)(v)
    assert np.isfinite(float(f(v)))
    np.testing.assert_allclose(float(grad @ u), _directional(f, v, u), rtol=1e-4)


EVENT_FEATURES = [
    "spike_threshold_t",
    "spike_peak_t",
    "spike_trough_t",
    "spike_upstroke_t",
    "spike_downstroke_t",
    "spike_fast_trough_t",
    "spike_width",
    "latency",
    "first_isi",
    "mean_isi",
    "median_isi",
    "isi_cv",
    "adapt",
]


# Their gradients at Allen's values differ from those at the interpolated values:
# the derivatives of a standard deviation or ratio of nearly equal intervals depend
# on sub-sample differences, and equal intervals on the sample grid make the
# median's choice a tie-break.
NONLINEAR_IN_TIMES = {"isi_cv", "adapt", "median_isi"}


@pytest.mark.parametrize("name", EVENT_FEATURES)
def test_event_time_gradients(sweeps, name):
    """Allen's sample times have the gradients of the times between samples;
    ``interpolate=True`` returns those times, smooth and within a sample."""
    t, v, i, end = sweeps["spiking"]
    dt = float(t[1] - t[0])

    def scalar(interpolate):
        # Sum over the spikes for spike features (all defined on this sweep).
        def f(v):
            x = _features(t, i, end, nan_to=0.0, interpolate=interpolate)(v)[name]
            return jnp.sum(x)

        return f

    exact, smooth = scalar(False), scalar(True)
    u = jnp.asarray(np.random.default_rng(1).normal(size=v.shape))
    grad_exact, grad_smooth = jax.grad(exact)(v), jax.grad(smooth)(v)
    assert np.any(grad_exact != 0) and np.all(np.isfinite(grad_exact))
    if name not in NONLINEAR_IN_TIMES:
        np.testing.assert_allclose(grad_exact, grad_smooth, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        float(grad_smooth @ u), _directional(smooth, v, u, h=1e-5), rtol=1e-3
    )
    if name.startswith("spike_") and name.endswith("_t"):
        n = int(detect_spikes(t, v, i, START, end, max_spikes=16).n_spikes)
        assert abs(float(smooth(v)) - float(exact(v))) <= n * dt


@pytest.mark.parametrize("kind", ["spiking", "silent"])
def test_spike_count_gradients(sweeps, kind):
    """``avg_rate`` keeps the exact count, with the gradient of the soft count of
    upward crossings of ``min_peak``, which also reaches silent sweeps."""
    t, v, i, end = sweeps[kind]

    def detect(v):
        return detect_spikes(t, v, i, START, end, max_spikes=16)

    def rate(v):
        return sweep_features(detect(v), "long_square")["avg_rate"]

    def soft(v):
        return _soft_spike_count(detect(v)) / (end - START)

    n = int(detect(v).n_spikes)
    assert float(rate(v)) == n / (end - START)
    grad = jax.grad(rate)(v)
    np.testing.assert_allclose(grad, jax.grad(soft)(v), rtol=1e-12, atol=1e-15)
    assert np.any(grad != 0)
    # A small step: the soft count has kinks where the voltage turns (relu).
    u = jnp.asarray(np.random.default_rng(2).normal(size=v.shape))
    fd = _directional(soft, v, u, h=1e-6)
    np.testing.assert_allclose(float(grad @ u), fd, rtol=1e-6)


def _long_squares(t, i, nan_to=None):
    """``long_square_features`` of a sweepset as a function of its ``[S, T]`` traces."""

    def f(v):
        per_sweep, first = [], []
        for k in range(v.shape[0]):
            sp = detect_spikes(t[k], v[k], i[k], START, END, max_spikes=16)
            per_sweep.append(sweep_features(sp, "long_square", **PATTERNS))
            first.append({name: x[0] for name, x in spike_features(sp).items()})
        sweeps = jax.tree.map(lambda *x: jnp.stack(x), *per_sweep)
        first = jax.tree.map(lambda *x: jnp.stack(x), *first)
        return long_square_features(sweeps, first, nan_to=nan_to)

    return f


def _sweepset(i_range, num_sweeps=4):
    return simulate_sweepset(
        num_sweeps=num_sweeps,
        dt=2.5e-5,
        t_max=0.1,
        stim_delay=START,
        stim_dur=END - START,
        i_range=i_range,
    )


def test_fi_slope_gradient():
    t, v, i = _sweepset((0.2, 1.0))
    slope = _long_squares(t, i)
    grad = jax.grad(lambda v: slope(v)["fi_fit_slope"])(v)
    assert np.isfinite(float(slope(v)["fi_fit_slope"]))
    assert np.all(np.isfinite(grad)) and np.any(grad != 0)


def test_rheobase_gradient():
    """Allen's rheobase, with gradients for the sweeps up to the rheobase sweep
    (a lower sweep starting to spike would lower it) and none for later ones."""
    t, v, i = _sweepset((-0.2, 0.6), num_sweeps=6)
    features = _long_squares(t, i)
    out = features(v)
    index = int(out["rheobase_index"])
    amp = np.asarray(_long_squares_amplitudes(t, v, i))
    assert 0 < index < 5 and float(out["rheobase_i"]) == amp[index]
    grad = np.asarray(jax.grad(lambda v: features(v)["rheobase_i"])(v))
    assert np.all(np.isfinite(grad))
    reached = np.any(grad != 0, axis=1)
    assert reached[: index + 1].all() and not reached[index + 1 :].any()


def _long_squares_amplitudes(t, v, i):
    return [
        sweep_features(detect_spikes(t[k], v[k], i[k], START, END), "long_square")[
            "stim_amp"
        ]
        for k in range(v.shape[0])
    ]


def test_sweepset_and_cell_gradients_are_finite():
    t, v, i = _sweepset((-0.3, 1.0), num_sweeps=5)
    long_squares = _long_squares(t, i)

    def total(v):
        out = long_squares(v)
        cell = cell_features(out, nan_to=0.0)
        leaves = jax.tree.leaves((out, cell))
        floats = [
            x for x in leaves if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating)
        ]
        return sum(jnp.sum(jnp.nan_to_num(x)) for x in floats)

    grad = jax.grad(total)(v)
    assert np.all(np.isfinite(grad)) and np.any(grad != 0)


@pytest.mark.parametrize(
    "kind", ["spiking", "silent", "hyperpolarizing", "single_spike", "clipped"]
)
def test_gradients_are_finite(sweeps, kind):
    """For every float feature, with NaN values replaced as in a loss."""
    t, v, i, end = sweeps[kind]
    features = _features(t, i, end, nan_to=0.0)

    def total(v):
        return sum(jnp.sum(x) for x in features(v).values())

    grad = jax.grad(total)(v)
    if not np.all(np.isfinite(grad)):
        bad = [
            name
            for name in features(v)
            if not np.all(
                np.isfinite(jax.grad(lambda v: jnp.sum(features(v)[name]))(v))
            )
        ]
        pytest.fail(f"{kind}: non-finite gradients of {bad}")


def test_gradients_under_jit_and_vmap():
    t, v, i = simulate_sweepset(
        num_sweeps=4,
        dt=2.5e-5,
        t_max=0.1,
        stim_delay=START,
        stim_dur=END - START,
        i_range=(-0.3, 1.0),
    )

    def loss(t, v, i):
        sp = detect_spikes(t, v, i, START, END, max_spikes=16)
        features = sweep_features(sp, "long_square", nan_to=0.0, **PATTERNS)
        return features["tau"] + features["delay_tau"] + features["peak_deflect"]

    batched = jax.jit(jax.vmap(jax.grad(loss, argnums=1)))(t, v, i)
    for k in range(4):
        single = jax.grad(loss, argnums=1)(t[k], v[k], i[k])
        np.testing.assert_allclose(batched[k], single, rtol=1e-6, atol=1e-12)
    assert np.all(np.isfinite(batched))


def test_nan_to_replaces_only_nan(sweeps):
    t, v, i, end = sweeps["silent"]
    plain = _features(t, i, end)(v)
    filled = _features(t, i, end, nan_to=-1.0)(v)
    assert any(np.isnan(x).any() for x in plain.values())  # the case is covered
    for name, x in plain.items():
        x = np.asarray(x)
        expected = np.where(np.isnan(x), -1.0, x)
        np.testing.assert_array_equal(np.asarray(filled[name]), expected, err_msg=name)


def test_nan_to_at_sweepset_level():
    t, v, i = simulate_sweepset(
        num_sweeps=4,
        dt=2.5e-5,
        t_max=0.1,
        stim_delay=START,
        stim_dur=END - START,
        i_range=(-0.3, 1.0),
    )
    per_sweep, first = [], []
    for k in range(4):
        sp = detect_spikes(t[k], v[k], i[k], START, END, max_spikes=16)
        per_sweep.append(sweep_features(sp, "long_square"))
        first.append({name: x[0] for name, x in spike_features(sp).items()})
    sweeps = jax.tree.map(lambda *x: jnp.stack(x), *per_sweep)
    first = jax.tree.map(lambda *x: jnp.stack(x), *first)
    plain = long_square_features(sweeps, first)
    filled = long_square_features(sweeps, first, nan_to=0.0)
    for x, y in zip(jax.tree.leaves(plain), jax.tree.leaves(filled)):
        x, y = np.asarray(x), np.asarray(y)
        if np.issubdtype(x.dtype, np.floating):
            np.testing.assert_array_equal(y, np.where(np.isnan(x), 0.0, x))
        else:
            np.testing.assert_array_equal(y, x)
