"""The one place where the parity tests call jaxley_ephys.

``extract`` runs spike detection plus all spike and sweep features on one sweep
as a pure JAX function (used for the jit/vmap/grad tests); ``analyse_sweep``
runs the public API eagerly for the parity tests and returns NumPy outputs
keyed like Allen's spike table, sweep features and firing-pattern metrics;
``analyse_cell`` runs the cell pipeline. Only this module changes when the
library API does.
"""

from __future__ import annotations

from functools import partial

import jax
import numpy as np

from jaxley_ephys import (
    detect_spikes,
    extract_cell_features,
    spike_features,
    sweep_features,
)
from jaxley_ephys.patterns import _delay_fit


def extract(
    t,
    v,
    i,
    start,
    end,
    *,
    stimulus=None,
    dv_cutoff=20.0,
    thresh_frac=0.05,
    max_spikes=32,
):
    """Spike and sweep features of one sweep (``[T]`` traces; s, mV, pA).

    Returns ``(n_spikes, overflow, spike_features, sweep_features)``.
    """
    sp = detect_spikes(
        t,
        v,
        i,
        start,
        end,
        dv_cutoff=dv_cutoff,
        thresh_frac=thresh_frac,
        max_spikes=max_spikes,
    )
    return sp.n_spikes, sp.overflow, spike_features(sp), sweep_features(sp, stimulus)


def analyse_sweep(
    t,
    v,
    i,
    start,
    end,
    *,
    stimulus=None,
    dv_cutoff=20.0,
    thresh_frac=0.05,
    delay=False,
) -> dict:
    """Features of one sweep, with as many spike slots as needed.

    ``"patterns"`` holds the burst and pause metrics of the sweep features, like
    Allen's ``(max_burstiness, n_bursts)`` and ``(n_pauses, pause_frac)``; with
    ``delay`` also ``(delay_ratio, delay_tau)`` and the fit behind them (dominant
    and single-exponential time constant in s, SSE of the fit used). Errors (e.g.
    running out of memory) are recorded in ``"error"`` so that the parity tests
    report them per sweep.
    """
    try:
        sp = detect_spikes(
            t,
            v,
            i,
            start,
            end,
            dv_cutoff=dv_cutoff,
            thresh_frac=thresh_frac,
            max_spikes=None,
        )
        spikes = spike_features(sp)
        features = sweep_features(sp, stimulus, bursts=True, pauses=True, delays=delay)
        patterns = {
            "burst": (features["max_burstiness"], features["n_bursts"]),
            "pause": (features["n_pauses"], features["pause_frac"]),
        }
        if delay:
            patterns["delay"] = (features["delay_ratio"], features["delay_tau"])
            patterns["delay_fit"] = _delay_fit(sp, 0.3)
    except Exception as exc:  # reported by the parity tests
        return {
            "n_spikes": -1,
            "spikes": {},
            "features": {},
            "patterns": {},
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }
    n = int(sp.n_spikes)
    return {
        "n_spikes": n,
        "spikes": {name: np.asarray(value)[:n] for name, value in spikes.items()},
        "features": {name: float(value) for name, value in features.items()},
        "patterns": {
            name: [float(x) for x in value] for name, value in patterns.items()
        },
        "error": None,
    }


def analyse_cell(long_squares, short_squares, ramps) -> dict:
    """``extract_cell_features`` on lists of ``(t, v, i, start, end)``, as NumPy."""
    out = extract_cell_features(long_squares, short_squares, ramps)
    return jax.tree.map(np.asarray, out)


@partial(jax.jit, static_argnames=("stimulus", "max_spikes"))
def extract_batch(t, v, i, start, end, *, stimulus=None, max_spikes=32):
    """``extract`` vmapped over a leading sweep axis."""
    return jax.vmap(partial(extract, stimulus=stimulus, max_spikes=max_spikes))(
        t, v, i, start, end
    )
