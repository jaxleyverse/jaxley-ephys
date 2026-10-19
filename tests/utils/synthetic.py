"""Synthetic traces that exercise Allen's rarer spike-detection branches.

Random sums of spike-like waveforms with random heights, rise and decay times,
afterdepolarizations, tight bursts and slow ramps, on top of noise and passive
responses, analysed in random windows (sometimes ending inside a spike). They
trigger merged crossings, re-found thresholds and peaks, dropped spikes, clipped
last spikes and missing fast troughs far more often than recorded sweeps do.
"""

from __future__ import annotations

import numpy as np

N_SAMPLES = 6000  # fixed length, so the JAX side compiles once


def _spike(t, t0, height, rise, decay, adp):
    x = t - t0
    up = np.exp(np.minimum(x, 0.0) / rise)
    down = np.exp(-np.maximum(x, 0.0) / decay)
    late = np.maximum(x - 3 * decay, 0.0) / 0.002
    bump = adp * np.exp(-late) * late
    return height * up * down * (x > -10 * rise) + bump


def random_trace(rng: np.random.Generator):
    """One trace ``(t, v, i, start, end)`` in s, mV and pA."""
    dt = rng.choice([2e-5, 1e-5, 5e-6])
    t = np.arange(N_SAMPLES) * dt
    duration = t[-1]
    v = -70.0 + rng.normal(0.0, rng.choice([0.0, 0.05, 0.3]), N_SAMPLES)
    start, end = np.sort(rng.uniform(0.05, 0.95, 2) * duration)
    if rng.random() < 0.5:  # passive step response
        tau = rng.uniform(0.002, 0.02)
        v += (
            rng.uniform(-15, 15)
            * (1 - np.exp(-np.clip(t - start, 0, None) / tau))
            * (t >= start)
        )
    if rng.random() < 0.3:  # slow ramp
        v += rng.uniform(0, 20) * np.clip((t - start) / (duration - start), 0, 1)
    n = rng.integers(0, 12)
    times = np.sort(rng.uniform(0, duration, n))
    if rng.random() < 0.3 and n > 1:  # tight burst, overlapping spikes
        times[1:] = times[0] + np.cumsum(rng.uniform(0.0005, 0.004, n - 1))
    for t0 in times:
        v += _spike(
            t,
            t0,
            height=rng.uniform(5, 110),
            rise=rng.uniform(1e-4, 4e-3),
            decay=rng.uniform(2e-4, 3e-3),
            adp=rng.choice([0.0, rng.uniform(0, 8)]),
        )
    i = np.where((t >= start) & (t < end), rng.uniform(-200, 400), 0.0)
    if n and rng.random() < 0.2:  # window ending during the last spike
        end = min(times[-1] + rng.uniform(0, 0.002), t[-1])
    return t, v, i, float(start), float(end)
