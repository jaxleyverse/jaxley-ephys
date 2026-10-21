"""Low-pass filtering and dV/dt, matching Allen's ``calculate_dvdt``.

Allen filters with ``scipy.signal.filtfilt`` and a 4-pole Bessel filter; this
module uses the same filter as second-order sections, which agrees to about
1e-10 V/s in dV/dt.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax

# Poles of the 4th-order analog Bessel prototype ("phase" normalization, as
# used by scipy.signal.bessel), one per conjugate pair.
_BESSEL4_POLES = (
    -0.6572111716718827 + 0.8301614350048730j,
    -0.9047587967882447 + 0.2709187330038746j,
)
# scipy.signal.filtfilt's default padding for a 4th-order filter: 3 * max(len(a), len(b))
_PADLEN = 15


def bessel4_sos(wn: Array) -> Array:
    """Second-order sections of a digital 4-pole low-pass Bessel filter.

    Equivalent to ``scipy.signal.bessel(4, wn, "low", output="sos")``; ``wn`` is
    the cutoff as a fraction of the Nyquist frequency. Returns a ``[2, 6]`` array
    of ``(b0, b1, b2, a0, a1, a2)`` rows.
    """
    warped = 4.0 * jnp.tan(jnp.pi * wn / 2.0)  # bilinear pre-warping with fs = 2
    rows = []
    for p in _BESSEL4_POLES:
        z = (4.0 + p * warped) / (4.0 - p * warped)  # pole of the digital filter
        a = jnp.stack([jnp.ones_like(wn), -2.0 * z.real, jnp.abs(z) ** 2])
        b = jnp.stack([jnp.ones_like(wn), 2.0 * jnp.ones_like(wn), jnp.ones_like(wn)])
        b = b * jnp.sum(a) / 4.0  # unit DC gain per section
        rows.append(jnp.concatenate([b, a]))
    return jnp.stack(rows)


def _sos_steady_state(sos: Array) -> Array:
    """Initial state for a unit step input (``scipy.signal.sosfilt_zi``)."""
    zi, gain = [], 1.0
    for b0, b1, b2, _, a1, a2 in sos:
        z1 = (b1 - a1 * b0 + b2 - a2 * b0) / (1.0 + a1 + a2)
        zi.append(gain * jnp.stack([z1, b2 - a2 * b0 - a2 * z1]))
        gain = gain * (b0 + b1 + b2) / (1.0 + a1 + a2)
    return jnp.stack(zi)


def _sosfilt(sos: Array, x: Array, zi: Array) -> Array:
    """Cascade of biquads (transposed direct form II) in a single scan."""

    def step(state, x_n):
        new_state = []
        for (b0, b1, b2, _, a1, a2), (s1, s2) in zip(sos, state):
            y_n = b0 * x_n + s1
            new_state.append(jnp.stack([b1 * x_n - a1 * y_n + s2, b2 * x_n - a2 * y_n]))
            x_n = y_n
        return jnp.stack(new_state), x_n

    return lax.scan(step, zi, x, unroll=4)[1]  # unrolling halves the loop overhead


def _odd_extension(v: Array, lo: Array, hi: Array) -> Array:
    """``v[lo:hi + 1]`` with ``_PADLEN`` reflected samples on both sides.

    The window bounds may be traced, so the window is laid out at the start of a
    buffer of static length ``len(v) + 2 * _PADLEN``; trailing entries repeat
    the last padded value and do not affect the window after reversal.
    """
    size = v.shape[0]
    n = hi - lo + 1
    j = jnp.arange(size + 2 * _PADLEN) - _PADLEN  # index relative to window start
    j = jnp.minimum(j, n + _PADLEN - 1)
    left = 2.0 * v[lo] - v[jnp.clip(lo - j, 0, size - 1)]
    mid = v[jnp.clip(lo + j, 0, size - 1)]
    right = 2.0 * v[hi] - v[jnp.clip(hi - (j - n + 1), 0, size - 1)]
    return jnp.where(j < 0, left, jnp.where(j < n, mid, right))


def _filtfilt(sos: Array, v: Array, lo: Array, hi: Array) -> Array:
    """Zero-phase filtering of ``v[lo:hi + 1]`` as ``scipy.signal.filtfilt`` does
    it (odd padding, steady-state initial conditions), on the full time axis."""
    size = v.shape[0]
    ext = _odd_extension(v, lo, hi)
    m = hi - lo + 1 + 2 * _PADLEN  # used length of the extended buffer
    zi = _sos_steady_state(sos)
    fwd = _sosfilt(sos, ext, zi * ext[0])
    rev = fwd[jnp.clip(m - 1 - jnp.arange(ext.shape[0]), 0, ext.shape[0] - 1)]
    bwd = _sosfilt(sos, rev, zi * rev[0])
    pos = m - 1 - _PADLEN - (jnp.arange(size) - lo)
    return bwd[jnp.clip(pos, 0, ext.shape[0] - 1)]


def _is_evenly_sampled(dt: Array, step: Array, t: Array) -> Array:
    """All sampling intervals ``dt`` equal ``step``, up to Allen's tolerance
    (``np.allclose``) plus the rounding of ``t`` in its precision (in float32,
    times around 1 s are only resolved to about 1e-7 s)."""
    rounding = (
        2 * jnp.finfo(t.dtype).eps * jnp.max(jnp.where(jnp.isnan(t), 0, jnp.abs(t)))
    )
    return jnp.all(jnp.abs(dt - step) <= 1e-8 + 1e-5 * jnp.abs(step) + rounding)


@partial(jax.jit, static_argnames=("filter",))
def calculate_dvdt(
    v: Array,
    t: Array,
    filter: float | None = None,
    window: tuple[Array, Array] | None = None,
) -> Array:
    """dV/dt in V/s (= mV/ms) of ``v`` (mV) over ``t`` (s), both ``[T]``.

    Like Allen's ``calculate_dvdt``, ``v`` is first low-pass filtered with a
    4-pole Bessel filter of cutoff ``filter`` (kHz) if the sampling interval is
    fixed; ``filter=None`` disables filtering. The result keeps the time axis:
    ``dvdt[k] = (v[k + 1] - v[k]) / (t[k + 1] - t[k])``, NaN in the last sample.
    Where ``dt == 0`` the result is NaN; Allen drops these samples instead, which
    shifts all later indices (see ``allen_ephys/ALLEN_EPHYS_FIXES.md``, section 6).

    For evenly sampled traces the sampling interval is taken from the whole time
    axis, ``(t[hi] - t[lo]) / (hi - lo)``, for the filter and the differences.
    In float64 this equals Allen's per-sample intervals to rounding; in float32
    the individual intervals are too coarse (several % off late in a sweep).

    Samples with NaN time at the end of ``t`` are not part of the sweep (e.g.
    padding, or samples after ``detect_spikes``' ``t_cutoff``); dV/dt is NaN there.
    With ``window=(lo, hi)`` only ``v[lo:hi + 1]`` is filtered, as Allen does for
    the analysis window during spike detection; values outside are meaningless.
    A cutoff at or above the Nyquist frequency gives NaN (Allen raises).
    Computed in the default float precision (float64 with ``jax_enable_x64``).
    """
    v, t = jnp.asarray(v, dtype=float), jnp.asarray(t, dtype=float)
    dt = jnp.diff(t)
    n = jnp.searchsorted(t, jnp.inf)  # samples of the sweep (NaN times sort last)
    lo, hi = (0, n - 1) if window is None else window
    step = (t[hi] - t[lo]) / jnp.maximum(hi - lo, 1)
    x = jnp.arange(dt.shape[0])
    inside = (x >= lo) & (x < hi)
    evenly_sampled = _is_evenly_sampled(jnp.where(inside, dt, step), step, t)
    if filter:
        wn = filter * 1e3 / (0.5 / step)  # cutoff / Nyquist frequency
        filtered = _filtfilt(bessel4_sos(wn), v, lo, hi)
        filtered = jnp.where((wn >= 0) & (wn < 1), filtered, jnp.nan)
        v = jnp.where(evenly_sampled, filtered, v)
    dt = jnp.where(evenly_sampled & inside, step, dt)
    # NaN where dt is 0 or NaN (after the sweep); dividing only by valid intervals
    # also keeps gradients finite there.
    ok = jnp.isfinite(dt) & (dt != 0)
    dvdt = jnp.where(ok, 1e-3 * jnp.diff(v) / jnp.where(ok, dt, 1.0), jnp.nan)
    return jnp.concatenate([dvdt, jnp.full((1,), jnp.nan, dtype=dvdt.dtype)])
