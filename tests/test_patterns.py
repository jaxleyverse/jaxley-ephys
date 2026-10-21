"""Burst, pause and delay metrics vs the vendored-Allen reference.

Burst and pause metrics agree exactly on every sweep, and on seeded random spike
trains that reach their rarer branches. Delay metrics are compared on long
squares (on ramps and short squares the rise before the first spike is not
exponential and Allen's fit returns arbitrary values): where Allen's
double-exponential fit separated two time constants, the delay agrees to the
precision of its optimizer; where the fit stayed at its start (both components
at the single exponential's time constant), the single exponentials agree and a
double exponential used instead fits better. See allen_ephys/ALLEN_EPHYS_FIXES.md,
section 6.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from allen_ephys import ephys_features as ft
from jaxley_ephys import (
    ISI_DETOUR,
    ISI_DIRECT,
    ISI_MISSING,
    Spikes,
    burst_metrics,
    delay_metrics,
    detect_spikes,
    pause_metrics,
    sweep_features,
)
from jaxley_ephys.patterns import _prespike_fit
from tests.utils.allen_reference import KINDS
from tests.utils.parity import mismatches
from tests.utils.simulate_testdata import simulate_sweepset

# Allen's curve_fit stops up to ~1e-4 (relative) short of the optimum.
DELAY_RTOL = 2e-4


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", ["burst", "pause"])
def test_burst_and_pause_metrics(reference, results, kind, name):
    problems = [
        f"sweep {ref['sweep_number']}: allen {ref['patterns'][name]}, "
        f"jax {res['patterns'][name]}"
        for ref, res in zip(reference["sweeps"][kind], results[kind])
        if not res["error"]
        and mismatches(name, ref["patterns"][name], res["patterns"][name]).any()
    ]
    assert not problems, f"{kind} {name}: {problems[:5]}"


def _allen_fit_kind(fit) -> str:
    """How Allen's double-exponential fit ended: ``"double"`` (two time constants
    as jaxley_ephys accepts them: distinct, below 100x the window and not a step
    at its first sample), ``"single"`` (both still at the same time constant),
    ``"none"`` (no fit) or ``"other"`` (curve_fit failed, or a time constant ran
    off towards a constant)."""
    if fit is None:
        return "none"
    if fit == "failed":
        return "other"
    _, tau1, _, tau2 = fit["params"]
    if tau1 <= 0 or tau2 <= 0:
        return "other"
    ratio = abs(np.log(tau1 / tau2))
    if ratio < 1e-3:
        return "single"
    k = 1.0 / np.array([tau1, tau2])
    if ratio > 1e-2 and all(k * fit["span"] > 1e-2) and all(k * fit["first"] <= 50):
        return "double"
    return "other"


def test_delay_metrics_on_long_squares(reference, results):
    problems = []
    for ref, res in zip(reference["sweeps"]["long_square"], results["long_square"]):
        if res["error"]:
            continue
        number = ref["sweep_number"]
        fit = ref["patterns"]["delay_fit"]
        allen_ratio, allen_tau = ref["patterns"]["delay"]  # tau in ms
        ratio, tau = res["patterns"]["delay"]
        tau_fit, single_tau, sse = res["patterns"]["delay_fit"]  # in s
        kind = _allen_fit_kind(fit)
        if kind == "none":  # Allen gives 0 where its fit cannot run
            ok = np.isnan(ratio) and np.isnan(tau) and allen_tau == 0
        elif kind == "double":
            ok = np.allclose([ratio, tau], [allen_ratio, allen_tau], rtol=DELAY_RTOL)
        elif kind == "single":
            better = tau_fit == single_tau or sse < fit["sse"]
            ok = np.isclose(1e3 * single_tau, allen_tau, rtol=DELAY_RTOL) and better
        else:
            continue  # Allen's fit failed or ran off; see ALLEN_EPHYS_FIXES.md
        if not ok:
            problems.append(
                f"sweep {number} ({kind}): allen tau {allen_tau}, jax {tau} "
                f"(single exponential {single_tau}, sse {sse} vs allen "
                f"{fit['sse'] if isinstance(fit, dict) else fit})"
            )
    assert not problems, problems[:5]


# Random spike trains, built directly as Spikes: bursts (short direct-reset ISIs
# between long detour-reset ones) or regular firing with occasional long ISIs,
# with some reset types changed at random, and random trough voltages.
N_TRAINS, MAX_SPIKES, N_SAMPLES, DT = 300, 32, 40000, 1e-4


def _random_intervals(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """ISIs (samples) and the reset type before each of them."""
    isis, codes = [], []
    if rng.random() < 0.65:
        for _ in range(rng.integers(1, 7)):
            m = int(rng.integers(1, 5))
            isis += list(rng.integers(15, 80, m - 1)) + [rng.integers(150, 1500)]
            codes += [ISI_DIRECT] * (m - 1) + [ISI_DETOUR]
    else:
        base = int(rng.integers(80, 500))
        for _ in range(rng.integers(1, 25)):
            long = rng.random() < 0.12
            isis.append(rng.integers(4, 9) * base if long else rng.poisson(base))
            codes.append(ISI_DETOUR if long and rng.random() < 0.6 else ISI_DIRECT)
    codes = np.array(codes, int)
    changed = rng.random(len(codes)) < 0.1
    codes[changed] = rng.choice([ISI_DIRECT, ISI_DETOUR, ISI_MISSING], changed.sum())
    return np.maximum(isis, 12), codes


def _random_train(rng: np.random.Generator):
    """``(thr, codes, fast, slow, v)``: threshold, fast and slow trough indices
    (-1 where missing) and reset type of each spike, and the voltage trace."""
    isis, codes = _random_intervals(rng)
    thr = 50 + np.concatenate([[0], np.cumsum(isis)])
    n = min(int(np.sum(thr < N_SAMPLES - 20)), MAX_SPIKES)
    thr = thr[:n]
    codes = np.append(codes, rng.choice([ISI_DIRECT, ISI_DETOUR]))[:n]
    fast = np.where(codes != ISI_MISSING, thr + 3, -1)
    # Slow troughs of detours 10-60% into the following ISI.
    after = np.diff(np.append(thr, N_SAMPLES - 1)) * rng.uniform(0.1, 0.6, n)
    slow = np.where(codes == ISI_DETOUR, thr + 3 + after.astype(int), -1)
    if n:
        slow[-1] = -1  # Allen drops the last spike's slow trough
    v = rng.normal(-55.0, 1.0, N_SAMPLES)
    v[thr] = rng.normal(-45.0, 1.5, n)
    detour = slow >= 0
    v[slow[detour]] = v[thr[detour]] - rng.uniform(-1.5, 6.0, np.sum(detour))
    return thr, codes, fast, slow, v


def _spikes(thr, codes, fast, slow, v) -> Spikes:
    """The spike train as ``Spikes`` (with ``MAX_SPIKES`` slots)."""
    n = len(thr)

    def buffer(x, fill=-1):
        return jnp.asarray(np.concatenate([x, np.full(MAX_SPIKES - n, fill)]))

    none = buffer(np.full(n, -1))
    return Spikes(
        t=jnp.arange(N_SAMPLES) * DT,
        v=jnp.asarray(v),
        i=None,
        dvdt=jnp.zeros(N_SAMPLES),
        start=jnp.asarray(0.0),
        end=jnp.asarray(N_SAMPLES * DT),
        threshold=buffer(thr),
        peak=none,
        trough=none,
        upstroke=none,
        downstroke=none,
        fast_trough=buffer(fast),
        adp=none,
        slow_trough=buffer(slow),
        isi_type=buffer(codes, ISI_MISSING),
        clipped=buffer(np.zeros(n, bool), False),
        overflow=jnp.asarray(False),
        thresh_frac=jnp.asarray(0.05),
        min_peak=jnp.asarray(-30.0),
        count_width=jnp.asarray(5.0),
    )


def _allen_patterns(thr, codes, fast, slow, v):
    """Allen's pause and burst metrics, as its sweep extractor computes them."""
    t = np.arange(N_SAMPLES) * DT
    isis = ft.get_isis(t, thr)
    names = {ISI_DIRECT: "direct", ISI_DETOUR: "detour", ISI_MISSING: np.nan}
    types = np.array([names[c] for c in codes[:-1]], dtype=object)

    def at(x, idx):
        return np.where(idx >= 0, x[np.maximum(idx, 0)], np.nan)

    with np.errstate(divide="ignore"):
        pauses = ft.detect_pauses(isis, types, 1.0).astype(int)
        bursts = ft.detect_bursts(
            isis,
            types,
            at(v, fast),
            at(t, fast),
            at(v, slow),
            at(t, slow),
            v[thr],
            0.5,
            1.0,
        )
    pause = (len(pauses), isis[pauses].sum() / (N_SAMPLES * DT) if len(pauses) else 0)
    bursts = np.array(bursts)
    burst = (bursts[:, 0].max(), len(bursts)) if len(bursts) else (0.0, 0)
    return pause, burst


def test_burst_and_pause_metrics_on_random_trains():
    rng = np.random.default_rng(7)
    problems, found = [], {"pause": 0, "burst": 0}
    for k in range(N_TRAINS):
        train = _random_train(rng)
        sp = _spikes(*train)
        expected = dict(zip(("pause", "burst"), _allen_patterns(*train)))
        actual = {"pause": pause_metrics(sp), "burst": burst_metrics(sp)}
        for name in expected:
            found[name] += expected[name][1] > 0
            if mismatches(name, expected[name], actual[name]).any():
                problems.append(f"train {k} {name}: allen {expected[name]}")
    assert not problems, problems[:5]
    # The trains must reach the interesting branches.
    assert found["pause"] > N_TRAINS / 10 and found["burst"] > N_TRAINS / 10


@pytest.mark.parametrize(
    "resets, isis, n_bursts",
    [
        # Several candidates: Allen's detect_bursts used them up (a zip) before
        # scoring them and raised.
        ("--d--d--", [13, 13, 49, 18, 34, 86, 35, 26], 3),
        # One candidate running to the end of the train: Allen appended its end to
        # an empty float array and raised on slicing with it.
        ("dd---", [80, 80, 20, 40, 30], 1),
    ],
)
def test_burst_detection_fixed_in_allen(resets, isis, n_bursts):
    """Bursts found where the original AllenSDK raises (ALLEN_EPHYS_FIXES.md, 2)."""
    codes = np.array([ISI_DETOUR if c == "d" else ISI_DIRECT for c in resets + "-"])
    thr = 50 + np.concatenate([[0], np.cumsum(isis)])
    fast = thr + 3
    after = (0.4 * np.append(isis, 0)).astype(int)
    slow = np.where(codes == ISI_DETOUR, fast + after, -1)
    v = np.full(N_SAMPLES, -55.0)
    v[thr], v[slow[slow >= 0]] = -45.0, -50.0
    train = (thr, codes, fast, slow, v)
    (_, allen_burst) = _allen_patterns(*train)
    assert allen_burst[1] == n_bursts
    assert not mismatches("burst", allen_burst, burst_metrics(_spikes(*train))).any()


def _double_exponential_rise(tau_fast, tau_slow, a_fast, a_slow, dt=2e-5, n=20000):
    t = np.arange(n) * dt
    v = -50.0 - a_fast * np.exp(-t / tau_fast) - a_slow * np.exp(-t / tau_slow)
    return jnp.asarray(t), jnp.asarray(v)


@pytest.mark.parametrize(
    "a_fast, a_slow, expected",
    [(20.0, 10.0, "fast"), (5.0, 25.0, "slow")],  # the larger weight dominates
)
def test_prespike_fit_recovers_double_exponential(a_fast, a_slow, expected):
    t, v = _double_exponential_rise(0.004, 0.03, a_fast, a_slow)
    tau, single_tau, _ = _prespike_fit(t, v, 0, t.shape[0], 0.3)
    np.testing.assert_allclose(tau, 0.004 if expected == "fast" else 0.03, rtol=1e-3)
    assert not np.isclose(single_tau, tau, rtol=1e-2)


def test_prespike_fit_of_single_exponential():
    t, v = _double_exponential_rise(0.01, 0.01, 15.0, 15.0)
    tau, single_tau, _ = _prespike_fit(t, v, 0, t.shape[0], 0.3)
    np.testing.assert_allclose([tau, single_tau], 0.01, rtol=1e-4)


PATTERN_FEATURES = (
    "max_burstiness",
    "n_bursts",
    "n_pauses",
    "pause_frac",
    "delay_ratio",
    "delay_tau",
)


def test_sweep_feature_toggles_under_jit_and_vmap():
    """The toggles add what the separate functions return; off by default."""
    start, end = 0.02, 0.08
    t, v, i = simulate_sweepset(
        num_sweeps=4,
        dt=2.5e-5,
        t_max=0.1,
        stim_delay=start,
        stim_dur=end - start,
        i_range=(-0.3, 1.0),
    )

    def features(t, v, i):
        sp = detect_spikes(t, v, i, start, end, max_spikes=32)
        return sweep_features(sp, bursts=True, pauses=True, delays=True)

    batched = jax.jit(jax.vmap(features))(t, v, i)
    for k in range(4):
        sp = detect_spikes(t[k], v[k], i[k], start, end, max_spikes=32)
        alone = (*burst_metrics(sp), *pause_metrics(sp), *delay_metrics(sp))
        for name, value in zip(PATTERN_FEATURES, alone):
            np.testing.assert_allclose(batched[name][k], value, rtol=1e-6)
        assert not set(PATTERN_FEATURES) & set(sweep_features(sp))
    assert np.isfinite(np.asarray(batched["delay_tau"])).any()  # some sweeps spike
