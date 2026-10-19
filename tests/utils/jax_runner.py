"""The one place where the parity tests call jaxley_ephys.

``extract`` runs spike detection plus all spike and sweep features on one sweep
as a pure JAX function (used for the jit/vmap/grad tests); ``analyse_sweep``
wraps it for the parity tests and returns NumPy outputs keyed like Allen's
spike table and sweep features. Only this module changes when the library API
does.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jaxley_ephys.events import compute_dvdt, detect_events
from jaxley_ephys.features import SPIKE_FEATURE_FUNCTIONS, SWEEP_FEATURE_FUNCTIONS


def extract(t, v, i, start, end, *, dv_cutoff=20.0, thresh_frac=0.05, max_spikes=32):
    """Spike and sweep features of one sweep (``[T]`` traces; s, mV, pA).

    Returns ``(n_spikes, overflow, spike_features, sweep_features)``. The
    current API has no ``thresh_frac`` parameter (it is fixed at 0.05).
    """
    del thresh_frac
    dvdt = compute_dvdt(t, v, filter=10.0)
    ev = detect_events(
        t,
        v,
        i,
        start,
        end,
        dvdt=dvdt,
        max_spikes=max_spikes,
        threshold_cutoff=dv_cutoff,
    )
    spikes = {
        name: fn(t, v, i, ev, dvdt) for name, fn in SPIKE_FEATURE_FUNCTIONS.items()
    }
    sweep = {
        name: fn(t, v, i, ev, dvdt) for name, fn in SWEEP_FEATURE_FUNCTIONS.items()
    }
    return jnp.sum(ev.spike_valid), ev.overflow, spikes, sweep


_extract = jax.jit(extract, static_argnames=("max_spikes",))


def analyse_sweep(t, v, i, start, end, dv_cutoff=20.0, thresh_frac=0.05) -> dict:
    """Run ``extract`` on one sweep, growing ``max_spikes`` until nothing overflows.

    Errors (e.g. running out of memory) are recorded in ``"error"`` so that the
    parity tests report them per sweep.
    """
    t, v, i = (jnp.asarray(x, dtype=jnp.float64) for x in (t, v, i))
    start, end = jnp.asarray(start), jnp.asarray(end)
    max_spikes = 32
    try:
        while True:
            n, overflow, spikes, sweep = _extract(
                t,
                v,
                i,
                start,
                end,
                dv_cutoff=dv_cutoff,
                thresh_frac=thresh_frac,
                max_spikes=max_spikes,
            )
            if not bool(overflow):
                break
            max_spikes *= 4
    except Exception as exc:  # reported by the parity tests
        return {
            "n_spikes": -1,
            "spikes": {},
            "features": {},
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }
    n = int(n)
    return {
        "n_spikes": n,
        "spikes": {name: np.asarray(value)[:n] for name, value in spikes.items()},
        "features": {name: float(value) for name, value in sweep.items()},
        "error": None,
    }


@partial(jax.jit, static_argnames=("max_spikes",))
def extract_batch(t, v, i, start, end, *, max_spikes=32):
    """``extract`` vmapped over a leading sweep axis."""
    return jax.vmap(partial(extract, max_spikes=max_spikes))(t, v, i, start, end)
