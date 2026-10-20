"""``t_cutoff`` and trailing NaN times: samples that are not part of the sweep."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxley_ephys import calculate_dvdt, detect_spikes, spike_features, sweep_features
from tests.utils.simulate_testdata import simulate_sweepset


def _features(sp, stimulus=None):
    return spike_features(sp), sweep_features(sp, stimulus)


def _assert_same(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(
            np.asarray(x, float), np.asarray(y, float), rtol=1e-12, equal_nan=True
        )


def test_default_cutoff_changes_no_result(sweeps):
    """By default, samples later than 20 ms after the window are not analysed
    (NaN in the returned traces), with the same results as the whole sweep."""
    s = sweeps["long_square"][-1]
    default = detect_spikes(s.t, s.v, s.i, s.start, s.end)
    whole = detect_spikes(s.t, s.v, s.i, s.start, s.end, t_cutoff=np.inf)
    assert default.t.shape == default.dvdt.shape == s.t.shape
    analysed = ~np.isnan(np.asarray(default.t))
    assert analysed.sum() < s.t.size / 3  # a long sweep with a 1 s window
    assert np.all(np.isnan(np.asarray(default.v)[~analysed]))
    _assert_same(_features(default, "long_square"), _features(whole, "long_square"))


def test_explicit_cutoff_equals_a_shorter_sweep(sweeps):
    s = sweeps["long_square"][-1]
    cutoff = s.end + 0.05
    n = int(np.searchsorted(s.t, cutoff, side="right"))
    shorter = detect_spikes(s.t[:n], s.v[:n], s.i[:n], s.start, s.end, t_cutoff=np.inf)
    eager = detect_spikes(s.t, s.v, s.i, s.start, s.end, t_cutoff=cutoff)
    jitted = jax.jit(
        lambda t, v, i: detect_spikes(t, v, i, s.start, s.end, t_cutoff=cutoff)
    )(s.t, s.v, s.i)
    for sp in (eager, jitted):
        _assert_same(_features(sp, "long_square"), _features(shorter, "long_square"))


def test_sweeps_of_unequal_length_batch_with_nan_padding():
    t, v, i = simulate_sweepset(
        num_sweeps=2, dt=2.5e-5, t_max=0.1, stim_delay=0.02, stim_dur=0.06
    )
    t, v, i = (np.array(x) for x in (t, v, i))  # writable copies
    n = 3400  # the second sweep ends at 85 ms
    single = [
        _features(detect_spikes(t[0], v[0], i[0], 0.02, 0.08)),
        _features(detect_spikes(t[1, :n], v[1, :n], i[1, :n], 0.02, 0.08)),
    ]
    t[1, n:] = np.nan  # padding: NaN times (v and i may hold anything)
    batched = jax.vmap(
        lambda t, v, i: _features(detect_spikes(t, v, i, 0.02, 0.08, max_spikes=256))
    )(jnp.asarray(t), jnp.asarray(v), jnp.asarray(i))
    for k in range(2):
        _assert_same(jax.tree.map(lambda x: x[k], batched), single[k])


def test_dvdt_with_nan_padding_matches_the_unpadded_trace():
    rng = np.random.default_rng(0)
    t = np.arange(5000) * 2e-5
    v = np.cumsum(rng.standard_normal(5000)) * 0.1 - 65
    unpadded = np.asarray(calculate_dvdt(v[:4000], t[:4000], 10.0))
    t[4000:] = np.nan
    padded = np.asarray(calculate_dvdt(v, t, 10.0))
    np.testing.assert_array_equal(padded[:4000], unpadded)
    assert np.all(np.isnan(padded[4000:]))
