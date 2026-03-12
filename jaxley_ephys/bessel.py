from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


_BESSEL4_PHASE_POLES = jnp.asarray(
    [
        -0.6572111716718828 + 0.8301614350048733j,
        -0.9047587967882450 + 0.2709187330038748j,
        -0.9047587967882450 - 0.2709187330038748j,
        -0.6572111716718828 - 0.8301614350048733j,
    ],
    dtype=jnp.complex64,
)


def _pair_poly_from_roots(r1: jax.Array, r2: jax.Array) -> jax.Array:
    c0 = jnp.asarray(1.0, dtype=jnp.float32)
    c1 = jnp.real(-(r1 + r2)).astype(jnp.float32)
    c2 = jnp.real(r1 * r2).astype(jnp.float32)
    return jnp.asarray([c0, c1, c2], dtype=jnp.float32)


def _odd_ext_1d(x: jax.Array, padlen: int) -> jax.Array:
    if padlen < 1:
        return x
    if x.shape[0] <= padlen:
        raise ValueError(f"Input length must be > padlen={padlen}")
    left = 2.0 * x[0] - x[1 : padlen + 1][::-1]
    right = 2.0 * x[-1] - x[-padlen - 1 : -1][::-1]
    return jnp.concatenate([left, x, right], axis=0)


def bessel4_lowpass_sos(wn: float) -> jax.Array:
    """Create order-4 digital low-pass Bessel filter as SOS.

    `wn` is normalized cutoff in (0, 1), where 1 is Nyquist.
    Returns shape `(2, 6)` with rows `[b0, b1, b2, a0, a1, a2]`.
    """
    if not (0.0 < wn < 1.0):
        raise ValueError("wn must be in (0, 1).")

    fs = 2.0
    wc = 2.0 * fs * jnp.tan(jnp.pi * wn / fs)
    p_a = _BESSEL4_PHASE_POLES * wc
    p_z = (2.0 * fs + p_a) / (2.0 * fs - p_a)

    zero_row = jnp.asarray([1.0, 2.0, 1.0], dtype=jnp.float32)
    pole_pairs = ((p_z[0], p_z[3]), (p_z[1], p_z[2]))

    sos_rows = []
    for p1, p2 in pole_pairs:
        a = _pair_poly_from_roots(p1, p2)
        b = zero_row
        b = b / a[0]
        a = a / a[0]
        sos_rows.append(jnp.concatenate([b, a]))

    sos = jnp.stack(sos_rows, axis=0)

    def section_dc_gain(row: jax.Array) -> jax.Array:
        b0, b1, b2, a0, a1, a2 = row
        return (b0 + b1 + b2) / (a0 + a1 + a2)

    g = jnp.prod(jax.vmap(section_dc_gain)(sos))
    return sos.at[0, :3].set(sos[0, :3] / g)


def _biquad_df2t(row: jax.Array, x: jax.Array, zi: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    b0, b1, b2, a0, a1, a2 = jnp.asarray(row)
    b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
    a1, a2 = a1 / a0, a2 / a0

    x = jnp.asarray(x)
    dtype = jnp.result_type(x, row)
    state = jnp.zeros((2,), dtype=dtype) if zi is None else jnp.asarray(zi, dtype=dtype)

    def step(z: jax.Array, x_n: jax.Array) -> tuple[jax.Array, jax.Array]:
        s1, s2 = z
        y_n = b0 * x_n + s1
        s1_new = b1 * x_n - a1 * y_n + s2
        s2_new = b2 * x_n - a2 * y_n
        return jnp.asarray([s1_new, s2_new], dtype=dtype), y_n

    zf, y = lax.scan(step, state, x)
    return y, zf


def sosfilt(sos: jax.Array, x: jax.Array, zi: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """Apply SOS cascade to 1D signal."""
    sos = jnp.asarray(sos)
    x = jnp.asarray(x)

    n_sections = int(sos.shape[0])
    dtype = jnp.result_type(x, sos)
    if zi is None:
        zi = jnp.zeros((n_sections, 2), dtype=dtype)
    else:
        zi = jnp.asarray(zi, dtype=dtype)
        if zi.shape != (n_sections, 2):
            raise ValueError(f"zi must have shape {(n_sections, 2)}")

    def section_step(signal_in: jax.Array, inputs: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        row, zi_row = inputs
        signal_out, zf_row = _biquad_df2t(row, signal_in, zi_row)
        return signal_out, zf_row

    y, zf = lax.scan(section_step, x, (sos, zi))
    return y, zf


def sosfilt_zi(sos: jax.Array) -> jax.Array:
    """Initial conditions for SOS step steady-state."""
    sos = jnp.asarray(sos)
    rows = []
    gain = jnp.asarray(1.0, dtype=sos.dtype)
    for i in range(int(sos.shape[0])):
        b0, b1, b2, a0, a1, a2 = sos[i]
        b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
        a1, a2 = a1 / a0, a2 / a0

        a_mat = jnp.asarray([[-a1, 1.0], [-a2, 0.0]], dtype=sos.dtype)
        b_vec = jnp.asarray([b1 - a1 * b0, b2 - a2 * b0], dtype=sos.dtype)
        zi = jnp.linalg.solve(jnp.eye(2, dtype=sos.dtype) - a_mat, b_vec) * gain
        rows.append(zi)
        gain = gain * (b0 + b1 + b2) / (1.0 + a1 + a2)
    return jnp.stack(rows, axis=0)


def sosfiltfilt(sos: jax.Array, x: jax.Array, padlen: int | None = None) -> jax.Array:
    """Forward-backward SOS filtering with odd extension padding."""
    x = jnp.asarray(x)
    sos = jnp.asarray(sos)
    if x.ndim != 1:
        raise ValueError("Only 1D input is supported.")

    n_sections = int(sos.shape[0])
    if padlen is None:
        n_zero_num = int(jnp.sum(sos[:, 2] == 0.0))
        n_zero_den = int(jnp.sum(sos[:, 5] == 0.0))
        padlen = 3 * (2 * n_sections + 1 - min(n_zero_num, n_zero_den))

    if x.shape[0] <= padlen:
        raise ValueError(f"Input length must be > padlen={padlen}")

    zi = sosfilt_zi(sos)
    x_pad = _odd_ext_1d(x, padlen)

    y_fwd, _ = sosfilt(sos, x_pad, zi * x_pad[0])
    y_rev = y_fwd[::-1]
    y_bwd, _ = sosfilt(sos, y_rev, zi * y_rev[0])
    y = y_bwd[::-1]
    return y[padlen:-padlen]


def allen_bessel_sosfiltfilt(v: jax.Array, t: jax.Array, cutoff_khz: float = 10.0) -> jax.Array:
    """Allen-style Bessel low-pass filter using JAX SOS + filtfilt."""
    v = jnp.asarray(v)
    t = jnp.asarray(t)
    if t.ndim != 1 or v.ndim != 1 or t.shape[0] != v.shape[0]:
        raise ValueError("allen_bessel_sosfiltfilt expects 1D aligned t and v")
    if t.shape[0] < 2:
        return v

    delta_t = t[1] - t[0]
    sample_freq = 1.0 / delta_t
    wn = (cutoff_khz * 1e3) / (sample_freq / 2.0)
    if wn < 0 or wn >= 1:
        raise ValueError(
            "bessel coeff ({:f}) is outside of valid range [0,1); cannot filter sampling frequency {:.1f} kHz with cutoff frequency {:.1f} kHz.".format(
                float(wn), float(sample_freq / 1e3), float(cutoff_khz)
            )
        )

    sos = bessel4_lowpass_sos(float(wn))
    return sosfiltfilt(sos, v)
