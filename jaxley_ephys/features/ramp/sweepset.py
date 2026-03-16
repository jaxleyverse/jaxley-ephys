from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

from jaxley_ephys.features.long_square.sweepset import _as_vector, _spiking_mask


def _spiking_nanmean(values: Array, avg_rate: Array) -> Array:
    vec = _as_vector(values)
    mask = _spiking_mask(avg_rate) & jnp.isfinite(vec)
    return jnp.where(jnp.any(mask), jnp.nanmean(jnp.where(mask, vec, jnp.nan)), jnp.nan)


def adapt(adapt: Array, avg_rate: Array) -> Array:
    return _spiking_nanmean(adapt, avg_rate)


def latency(latency: Array, avg_rate: Array) -> Array:
    return _spiking_nanmean(latency, avg_rate)


def isi_cv(isi_cv: Array, avg_rate: Array) -> Array:
    return _spiking_nanmean(isi_cv, avg_rate)


def mean_isi(mean_isi: Array, avg_rate: Array) -> Array:
    return _spiking_nanmean(mean_isi, avg_rate)


def median_isi(median_isi: Array, avg_rate: Array) -> Array:
    return _spiking_nanmean(median_isi, avg_rate)


def first_isi(first_isi: Array, avg_rate: Array) -> Array:
    return _spiking_nanmean(first_isi, avg_rate)


def avg_rate(avg_rate: Array) -> Array:
    return _spiking_nanmean(avg_rate, avg_rate)


def v_baseline(v_baseline: Array, avg_rate: Array) -> Array:
    return _spiking_nanmean(v_baseline, avg_rate)


def tau(tau: Array, avg_rate: Array) -> Array:
    return _spiking_nanmean(tau, avg_rate)


def sag(sag: Array, avg_rate: Array) -> Array:
    return _spiking_nanmean(sag, avg_rate)


def peak_deflect(peak_deflect: Array, avg_rate: Array) -> Array:
    return _spiking_nanmean(peak_deflect, avg_rate)


def peak_deflect_index(peak_deflect_index: Array, avg_rate: Array) -> Array:
    return _spiking_nanmean(peak_deflect_index, avg_rate)


def stim_amp(stim_amp: Array, avg_rate: Array) -> Array:
    return _spiking_nanmean(stim_amp, avg_rate)


SWEEPSET_FEATURE_FUNCTIONS: dict[str, Callable[..., Array]] = {
    "adapt": adapt,
    "latency": latency,
    "isi_cv": isi_cv,
    "mean_isi": mean_isi,
    "median_isi": median_isi,
    "first_isi": first_isi,
    "avg_rate": avg_rate,
    "v_baseline": v_baseline,
    "tau": tau,
    "sag": sag,
    "peak_deflect": peak_deflect,
    "peak_deflect_index": peak_deflect_index,
    "stim_amp": stim_amp,
}

SWEEPSET_FEATURE_INPUTS: dict[str, tuple[str, ...]] = {
    "adapt": ("adapt", "avg_rate"),
    "latency": ("latency", "avg_rate"),
    "isi_cv": ("isi_cv", "avg_rate"),
    "mean_isi": ("mean_isi", "avg_rate"),
    "median_isi": ("median_isi", "avg_rate"),
    "first_isi": ("first_isi", "avg_rate"),
    "avg_rate": ("avg_rate",),
    "v_baseline": ("v_baseline", "avg_rate"),
    "tau": ("tau", "avg_rate"),
    "sag": ("sag", "avg_rate"),
    "peak_deflect": ("peak_deflect", "avg_rate"),
    "peak_deflect_index": ("peak_deflect_index", "avg_rate"),
    "stim_amp": ("stim_amp", "avg_rate"),
}


def compute_sweepset_features(
    sweep_feature_vectors: dict[str, Array],
    *,
    feature_names: tuple[str, ...] | None = None,
) -> dict[str, Array]:
    out: dict[str, Array] = {}
    names = feature_names if feature_names is not None else tuple(SWEEPSET_FEATURE_FUNCTIONS.keys())
    for name in names:
        if name not in SWEEPSET_FEATURE_FUNCTIONS:
            raise KeyError(f"unknown sweepset feature '{name}'")
        fn = SWEEPSET_FEATURE_FUNCTIONS[name]
        deps = SWEEPSET_FEATURE_INPUTS[name]
        missing = [dep for dep in deps if dep not in sweep_feature_vectors]
        if missing:
            raise KeyError(f"missing sweepset input vectors for {name}: {missing}")
        args = [sweep_feature_vectors[dep] for dep in deps]
        out[name] = fn(*args)
    return out


__all__ = ["SWEEPSET_FEATURE_FUNCTIONS", "compute_sweepset_features"]
