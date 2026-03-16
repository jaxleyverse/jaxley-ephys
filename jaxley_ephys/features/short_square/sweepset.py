from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

from jaxley_ephys.features.long_square.sweepset import _as_vector, _mode_with_tie_to_lower, _spiking_mask


def stimulus_amplitude(avg_rate: Array, stim_amp: Array) -> Array:
    rate = _as_vector(avg_rate)
    amp = _as_vector(stim_amp)
    spiking = _spiking_mask(rate) & jnp.isfinite(amp)
    values = jnp.where(spiking, amp, jnp.nan)
    return _mode_with_tie_to_lower(values)


def _common_amp_mask(avg_rate: Array, stim_amp: Array) -> Array:
    rate = _as_vector(avg_rate)
    amp = _as_vector(stim_amp)
    common = stimulus_amplitude(avg_rate, stim_amp)
    return _spiking_mask(rate) & jnp.isfinite(amp) & jnp.isclose(amp, common, rtol=0.0, atol=1e-9)


def _common_amp_nanmean(values: Array, avg_rate: Array, stim_amp: Array) -> Array:
    vec = _as_vector(values)
    mask = _common_amp_mask(avg_rate, stim_amp) & jnp.isfinite(vec)
    return jnp.where(jnp.any(mask), jnp.nanmean(jnp.where(mask, vec, jnp.nan)), jnp.nan)


def adapt(adapt: Array, avg_rate: Array, stim_amp: Array) -> Array:
    return _common_amp_nanmean(adapt, avg_rate, stim_amp)


def latency(latency: Array, avg_rate: Array, stim_amp: Array) -> Array:
    return _common_amp_nanmean(latency, avg_rate, stim_amp)


def isi_cv(isi_cv: Array, avg_rate: Array, stim_amp: Array) -> Array:
    return _common_amp_nanmean(isi_cv, avg_rate, stim_amp)


def mean_isi(mean_isi: Array, avg_rate: Array, stim_amp: Array) -> Array:
    return _common_amp_nanmean(mean_isi, avg_rate, stim_amp)


def median_isi(median_isi: Array, avg_rate: Array, stim_amp: Array) -> Array:
    return _common_amp_nanmean(median_isi, avg_rate, stim_amp)


def first_isi(first_isi: Array, avg_rate: Array, stim_amp: Array) -> Array:
    return _common_amp_nanmean(first_isi, avg_rate, stim_amp)


def avg_rate(avg_rate: Array, stim_amp: Array) -> Array:
    return _common_amp_nanmean(avg_rate, avg_rate, stim_amp)


def v_baseline(v_baseline: Array, avg_rate: Array, stim_amp: Array) -> Array:
    return _common_amp_nanmean(v_baseline, avg_rate, stim_amp)


def tau(tau: Array, avg_rate: Array, stim_amp: Array) -> Array:
    return _common_amp_nanmean(tau, avg_rate, stim_amp)


def sag(sag: Array, avg_rate: Array, stim_amp: Array) -> Array:
    return _common_amp_nanmean(sag, avg_rate, stim_amp)


def peak_deflect(peak_deflect: Array, avg_rate: Array, stim_amp: Array) -> Array:
    return _common_amp_nanmean(peak_deflect, avg_rate, stim_amp)


def peak_deflect_index(peak_deflect_index: Array, avg_rate: Array, stim_amp: Array) -> Array:
    return _common_amp_nanmean(peak_deflect_index, avg_rate, stim_amp)


def stim_amp(stim_amp: Array, avg_rate: Array) -> Array:
    return stimulus_amplitude(avg_rate, stim_amp)


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
    "stimulus_amplitude": stimulus_amplitude,
}

SWEEPSET_FEATURE_INPUTS: dict[str, tuple[str, ...]] = {
    "adapt": ("adapt", "avg_rate", "stim_amp"),
    "latency": ("latency", "avg_rate", "stim_amp"),
    "isi_cv": ("isi_cv", "avg_rate", "stim_amp"),
    "mean_isi": ("mean_isi", "avg_rate", "stim_amp"),
    "median_isi": ("median_isi", "avg_rate", "stim_amp"),
    "first_isi": ("first_isi", "avg_rate", "stim_amp"),
    "avg_rate": ("avg_rate", "stim_amp"),
    "v_baseline": ("v_baseline", "avg_rate", "stim_amp"),
    "tau": ("tau", "avg_rate", "stim_amp"),
    "sag": ("sag", "avg_rate", "stim_amp"),
    "peak_deflect": ("peak_deflect", "avg_rate", "stim_amp"),
    "peak_deflect_index": ("peak_deflect_index", "avg_rate", "stim_amp"),
    "stim_amp": ("stim_amp", "avg_rate"),
    "stimulus_amplitude": ("avg_rate", "stim_amp"),
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
