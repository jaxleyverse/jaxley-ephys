from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array


SAG_TARGET_MV = -100.0
SUBTHRESH_MIN_AMP = -100.0
SUBTHRESH_MAX_AMP = 0.0


def _as_vector(values: Array) -> Array:
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    raise ValueError(
        f"sweepset feature vectors must be [S] or [S, 1], got {values.shape}"
    )


def _nanmean_or_nan(values: Array, mask: Array | None = None) -> Array:
    vec = _as_vector(values)
    finite = jnp.isfinite(vec)
    if mask is not None:
        finite = finite & mask
    return jnp.where(jnp.any(finite), jnp.nanmean(jnp.where(finite, vec, jnp.nan)), jnp.nan)


def _spiking_mask(avg_rate: Array) -> Array:
    rate = _as_vector(avg_rate)
    return jnp.isfinite(rate) & (rate > 0.0)


def _subthreshold_mask(avg_rate: Array) -> Array:
    rate = _as_vector(avg_rate)
    return jnp.isfinite(rate) & (rate == 0.0)


def _membrane_property_mask(avg_rate: Array, stim_amp: Array) -> Array:
    sub = _subthreshold_mask(avg_rate)
    amp = _as_vector(stim_amp)
    return sub & jnp.isfinite(amp) & (amp > SUBTHRESH_MIN_AMP) & (amp < SUBTHRESH_MAX_AMP)


def _mode_with_tie_to_lower(values: Array) -> Array:
    vec = _as_vector(values)
    finite = jnp.isfinite(vec)
    vec = jnp.where(finite, vec, jnp.nan)
    eq = jnp.isclose(vec[:, None], vec[None, :], rtol=0.0, atol=1e-9)
    eq = eq & finite[:, None] & finite[None, :]
    counts = jnp.sum(eq, axis=1)
    max_count = jnp.max(jnp.where(finite, counts, -1))
    candidates = jnp.where(finite & (counts == max_count), vec, jnp.inf)
    mode = jnp.min(candidates)
    return jnp.where(jnp.any(finite), mode, jnp.nan)


def adapt(adapt: Array) -> Array:
    return _nanmean_or_nan(adapt)


def latency(latency: Array) -> Array:
    return _nanmean_or_nan(latency)


def isi_cv(isi_cv: Array) -> Array:
    return _nanmean_or_nan(isi_cv)


def mean_isi(mean_isi: Array) -> Array:
    return _nanmean_or_nan(mean_isi)


def median_isi(median_isi: Array) -> Array:
    return _nanmean_or_nan(median_isi)


def first_isi(first_isi: Array) -> Array:
    return _nanmean_or_nan(first_isi)


def avg_rate(avg_rate: Array) -> Array:
    return _nanmean_or_nan(avg_rate)


def v_baseline(v_baseline: Array) -> Array:
    return _nanmean_or_nan(v_baseline)


def tau(tau: Array, avg_rate: Array, stim_amp: Array) -> Array:
    mask = _membrane_property_mask(avg_rate, stim_amp)
    return _nanmean_or_nan(tau, mask)


def peak_deflect(peak_deflect: Array) -> Array:
    return _nanmean_or_nan(peak_deflect)


def peak_deflect_index(peak_deflect_index: Array) -> Array:
    return _nanmean_or_nan(peak_deflect_index)


def stim_amp(stim_amp: Array) -> Array:
    return _nanmean_or_nan(stim_amp)


def rheobase_index(avg_rate: Array, stim_amp: Array) -> Array:
    rate = _as_vector(avg_rate)
    amp = _as_vector(stim_amp)
    spiking = _spiking_mask(rate) & jnp.isfinite(amp)
    min_amp = jnp.min(jnp.where(spiking, amp, jnp.inf))
    candidates = spiking & jnp.isclose(amp, min_amp, rtol=0.0, atol=1e-9)
    candidate_rates = jnp.where(candidates, rate, -jnp.inf)
    best = jnp.argmax(candidate_rates)
    return jnp.where(jnp.any(spiking), best, jnp.asarray(-1))


def rheobase_i(avg_rate: Array, stim_amp: Array) -> Array:
    idx = rheobase_index(avg_rate, stim_amp)
    amp = _as_vector(stim_amp)
    valid = idx >= 0
    safe_idx = jnp.where(valid, idx, 0)
    return jnp.where(valid, amp[safe_idx], jnp.nan)


def fi_fit_slope(avg_rate: Array, stim_amp: Array) -> Array:
    rate = _as_vector(avg_rate)
    amp = _as_vector(stim_amp)
    spiking = _spiking_mask(rate) & jnp.isfinite(amp)
    n = jnp.sum(spiking)
    x = jnp.where(spiking, amp, 0.0)
    y = jnp.where(spiking, rate, 0.0)
    n_f = jnp.maximum(n.astype(amp.dtype), 1.0)
    xm = jnp.sum(x) / n_f
    ym = jnp.sum(y) / n_f
    x0 = jnp.where(spiking, x - xm, 0.0)
    y0 = jnp.where(spiking, y - ym, 0.0)
    den = jnp.sum(x0 * x0)
    num = jnp.sum(x0 * y0)
    slope = jnp.where(den > 0.0, num / den, jnp.nan)
    return jnp.where(n >= 2, slope, jnp.nan)


def _sag_choice_index(peak_deflect: Array, avg_rate: Array) -> Array:
    vm = _as_vector(peak_deflect)
    sub = _subthreshold_mask(avg_rate) & jnp.isfinite(vm)
    dist = jnp.abs(vm - SAG_TARGET_MV)
    idx = jnp.argmin(jnp.where(sub, dist, jnp.inf))
    return jnp.where(jnp.any(sub), idx, jnp.asarray(-1))


def sag(sag: Array, peak_deflect: Array, avg_rate: Array) -> Array:
    values = _as_vector(sag)
    idx = _sag_choice_index(peak_deflect, avg_rate)
    valid = idx >= 0
    safe_idx = jnp.where(valid, idx, 0)
    return jnp.where(valid, values[safe_idx], jnp.nan)


def vm_for_sag(peak_deflect: Array, avg_rate: Array) -> Array:
    vm = _as_vector(peak_deflect)
    idx = _sag_choice_index(peak_deflect, avg_rate)
    valid = idx >= 0
    safe_idx = jnp.where(valid, idx, 0)
    return jnp.where(valid, vm[safe_idx], jnp.nan)


def input_resistance(peak_deflect: Array, stim_amp: Array, avg_rate: Array, v_baseline: Array) -> Array:
    v_peak = _as_vector(peak_deflect)
    amp = _as_vector(stim_amp)
    base = _as_vector(v_baseline)
    mask = _membrane_property_mask(avg_rate, stim_amp) & jnp.isfinite(v_peak)
    n = jnp.sum(mask)
    idx = jnp.arange(v_peak.shape[0])
    first_idx = jnp.argmin(jnp.where(mask, idx, jnp.inf))
    x = jnp.where(mask, amp, 0.0)
    y = jnp.where(mask, v_peak, 0.0)

    n_f = n.astype(amp.dtype)
    sum_x = jnp.sum(x)
    sum_y = jnp.sum(y)
    sum_xx = jnp.sum(x * x)
    sum_xy = jnp.sum(x * y)

    add_baseline = n == 1
    baseline = jnp.where(add_baseline, base[first_idx], 0.0)
    n_eff = jnp.where(add_baseline, n_f + 1.0, n_f)
    sum_y_eff = jnp.where(add_baseline, sum_y + baseline, sum_y)

    den = n_eff * sum_xx - sum_x * sum_x
    num = n_eff * sum_xy - sum_x * sum_y_eff
    slope = jnp.where(den != 0.0, num / den, jnp.nan)
    return jnp.where(n > 0, 1e3 * slope, jnp.nan)


def stimulus_amplitude(avg_rate: Array, stim_amp: Array) -> Array:
    rate = _as_vector(avg_rate)
    amp = _as_vector(stim_amp)
    spiking = _spiking_mask(rate) & jnp.isfinite(amp)
    values = jnp.where(spiking, amp, jnp.nan)
    return _mode_with_tie_to_lower(values)


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
    "rheobase_index": rheobase_index,
    "rheobase_i": rheobase_i,
    "fi_fit_slope": fi_fit_slope,
    "vm_for_sag": vm_for_sag,
    "input_resistance": input_resistance,
    "stimulus_amplitude": stimulus_amplitude,
}

SWEEPSET_FEATURE_INPUTS: dict[str, tuple[str, ...]] = {
    "adapt": ("adapt",),
    "latency": ("latency",),
    "isi_cv": ("isi_cv",),
    "mean_isi": ("mean_isi",),
    "median_isi": ("median_isi",),
    "first_isi": ("first_isi",),
    "avg_rate": ("avg_rate",),
    "v_baseline": ("v_baseline",),
    "tau": ("tau", "avg_rate", "stim_amp"),
    "sag": ("sag", "peak_deflect", "avg_rate"),
    "peak_deflect": ("peak_deflect",),
    "peak_deflect_index": ("peak_deflect_index",),
    "stim_amp": ("stim_amp",),
    "rheobase_index": ("avg_rate", "stim_amp"),
    "rheobase_i": ("avg_rate", "stim_amp"),
    "fi_fit_slope": ("avg_rate", "stim_amp"),
    "vm_for_sag": ("peak_deflect", "avg_rate"),
    "input_resistance": ("peak_deflect", "stim_amp", "avg_rate", "v_baseline"),
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
