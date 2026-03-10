from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from allen_ephys.ephys_extractor import EphysSweepFeatureExtractor

from jaxley_ephys.base import compute_dvdt, detect_events
from tests.utils.simulate_testdata import simulate_sweep, simulate_sweepset


def _padded_indices(
    values: list[float | int], max_spikes: int
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.full(max_spikes, -1, dtype=np.int32)
    valid = np.zeros(max_spikes, dtype=bool)
    for k, value in enumerate(values[:max_spikes]):
        if value is None or not np.isfinite(value):
            continue
        idx[k] = int(value)
        valid[k] = True
    return idx, valid


def _reformat_allen_events(sweep, max_spikes: int):
    spikes = sweep.spikes()
    threshold_raw = [sp.get("threshold_index") for sp in spikes]
    peak_raw = [sp.get("peak_index") for sp in spikes]
    trough_raw = [sp.get("trough_index") for sp in spikes]

    threshold_idx, threshold_valid = _padded_indices(threshold_raw, max_spikes)
    peak_idx, peak_valid = _padded_indices(peak_raw, max_spikes)
    trough_idx, trough_valid = _padded_indices(trough_raw, max_spikes)

    T = len(sweep.t)
    threshold_mask = np.zeros(T, dtype=bool)
    peak_mask = np.zeros(T, dtype=bool)
    trough_mask = np.zeros(T, dtype=bool)

    threshold_mask[threshold_idx[threshold_valid]] = True
    peak_mask[peak_idx[peak_valid]] = True
    trough_mask[trough_idx[trough_valid]] = True

    return {
        "threshold_idx": threshold_idx,
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
        "threshold_valid": threshold_valid,
        "peak_valid": peak_valid,
        "trough_valid": trough_valid,
        "threshold_mask": threshold_mask,
        "peak_mask": peak_mask,
        "trough_mask": trough_mask,
        "n_spikes": min(len(spikes), max_spikes),
        "overflow": len(spikes) > max_spikes,
    }


@pytest.mark.parametrize("sweep_kind", ["long_spiking", "short_square", "ramp"])
def test_detect_events_matches_allen_spike_detection(specimen_case, sweep_kind):
    max_spikes = 100
    sweep = specimen_case.representative_sweep(sweep_kind)
    sweep.process_spikes()

    t0 = float(sweep.start) if sweep.start is not None else float(sweep.t[0])
    t1 = float(sweep.end) if sweep.end is not None else float(sweep.t[-1])
    ev = detect_events(
        sweep.t,
        sweep.v,
        sweep.i,
        t0,
        t1,
        max_spikes=max_spikes,
    )
    expected = _reformat_allen_events(sweep, max_spikes=max_spikes)

    assert abs(int(ev.n_spikes) - expected["n_spikes"]) <= 1
    assert bool(ev.overflow) == expected["overflow"]

    n = min(int(ev.n_spikes), expected["n_spikes"])
    got_thr = np.asarray(ev.threshold_idxs)[:n]
    exp_thr = expected["threshold_idx"][:n]
    got_peak = np.asarray(ev.peak_idxs)[:n]
    exp_peak = expected["peak_idx"][:n]
    got_tr = np.asarray(ev.trough_idxs)[:n]
    exp_tr = expected["trough_idx"][:n]

    if n > 0:
        assert np.all(np.abs(got_thr - exp_thr) <= 20)
        assert np.all(np.abs(got_peak - exp_peak) <= 20)
        tr_ok = (got_tr < 0) | (exp_tr < 0) | (np.abs(got_tr - exp_tr) <= 20)
        assert np.all(tr_ok)


def test_matches_allen_synthetic():
    t0 = 0.02
    t1 = 0.1
    tmax = 0.12
    dt = 1e-4
    t, v, i = simulate_sweep(dt=dt, stim_delay=t0, stim_dur=t1 - t0, t_max=tmax)

    sweep = EphysSweepFeatureExtractor(
        np.asarray(t),
        np.asarray(v),
        np.asarray(i),
        start=t0,
        end=t1,
        filter=4.0,
    )
    sweep.process_spikes()

    max_spikes = 128
    expected = _reformat_allen_events(sweep, max_spikes=max_spikes)
    ev = detect_events(t, v, i, t0, t1, max_spikes=max_spikes, dvdt_filter=4.0)

    assert int(ev.n_spikes) == expected["n_spikes"]
    assert bool(ev.overflow) == expected["overflow"]

    assert np.array_equal(np.asarray(ev.threshold_idxs), expected["threshold_idx"])
    assert np.array_equal(np.asarray(ev.peak_idxs), expected["peak_idx"])
    assert np.array_equal(np.asarray(ev.trough_idxs), expected["trough_idx"])
    assert np.array_equal(np.asarray(ev.threshold_valid), expected["threshold_valid"])
    assert np.array_equal(np.asarray(ev.peak_valid), expected["peak_valid"])
    assert np.array_equal(np.asarray(ev.trough_valid), expected["trough_valid"])


def test_compute_dvdt_jittable():
    t, v, _ = simulate_sweep(dt=1e-4, t_max=0.12, stim_amp=0.7)
    fn = jax.jit(lambda t_, v_: compute_dvdt(t_, v_))
    out = fn(t, v)
    assert out.shape == t.shape


def test_compute_dvdt_vmappable():
    t_batch, v_batch, _ = simulate_sweepset(
        num_sweeps=2,
        dt=1e-4,
        t_max=0.12,
        i_range=(0.3, 0.8),
    )

    fn = jax.vmap(compute_dvdt, in_axes=(0, 0))
    out = fn(t_batch, v_batch)
    assert out.shape == t_batch.shape


def test_detect_events_jittable():
    t, v, i = simulate_sweep(dt=1e-4, t_max=0.12, stim_amp=0.7)
    t0 = jnp.asarray(0.02)
    t1 = jnp.asarray(0.1)

    f = jax.jit(lambda tt, vv, ii: detect_events(tt, vv, ii, t0, t1, max_spikes=16))
    out = f(t, v, i)
    assert out.spikes.shape == t.shape
    assert out.threshold_idxs.shape == (16,)
    assert out.peak_idxs.shape == (16,)
    assert out.trough_idxs.shape == (16,)
    assert out.threshold_valid.shape == (16,)
    assert out.peak_valid.shape == (16,)
    assert out.trough_valid.shape == (16,)


def test_detect_events_vmappable():
    t_batch, v_batch, i_batch = simulate_sweepset(
        num_sweeps=3,
        dt=1e-4,
        t_max=0.12,
        i_range=(0.2, 0.9),
    )
    t0 = jnp.full((t_batch.shape[0],), 0.02)
    t1 = jnp.full((t_batch.shape[0],), 0.1)

    f = jax.vmap(
        lambda tt, vv, ii, t0_, t1_: detect_events(tt, vv, ii, t0_, t1_, max_spikes=16),
        in_axes=(0, 0, 0, 0, 0),
    )
    out = f(t_batch, v_batch, i_batch, t0, t1)
    assert out.spikes.shape == t_batch.shape
    assert out.thresholds.shape == t_batch.shape
    assert out.peaks.shape == t_batch.shape
    assert out.troughs.shape == t_batch.shape
    assert out.threshold_idxs.shape == (3, 16)
    assert out.peak_idxs.shape == (3, 16)
    assert out.trough_idxs.shape == (3, 16)
    assert out.threshold_valid.shape == (3, 16)
    assert out.peak_valid.shape == (3, 16)
    assert out.trough_valid.shape == (3, 16)


@pytest.mark.parametrize("noise_level", [0.01, 0.05, 0.1, 0.5])
def test_detect_events(noise_level):
    t0 = 0.02
    t1 = 0.1
    tmax = 0.12
    dT = t1 - t0
    t, v, i = simulate_sweep(
        dt=1e-4,
        stim_delay=t0,
        stim_dur=dT,
        t_max=tmax,
        noise_level=noise_level,
    )

    ev = detect_events(t, v, i, t0, t1, max_spikes=8)
    assert int(ev.n_spikes) == 5
    assert bool(ev.overflow) is False


def test_detect_events_overflow():
    t0 = 0.02
    t1 = 0.1
    tmax = 0.12
    dT = t1 - t0
    t, v, i = simulate_sweep(dt=1e-4, stim_delay=t0, stim_dur=dT, t_max=tmax)

    ev = detect_events(t, v, i, t0, t1, max_spikes=3)
    assert bool(ev.overflow) is True
