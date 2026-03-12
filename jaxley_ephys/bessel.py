from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


_BESSEL_PHASE_POLES = {
    1: (-1.0 + 0.0j,),
    2: (
        -0.8660254037844385 + 0.4999999999999999j,
        -0.8660254037844385 - 0.4999999999999999j,
    ),
    3: (
        -0.7456403858480766 + 0.7113666249728351j,
        -0.9416000265332067 + 0.0j,
        -0.7456403858480766 - 0.7113666249728351j,
    ),
    4: (
        -0.6572111716718827 + 0.8301614350048730j,
        -0.9047587967882447 + 0.2709187330038746j,
        -0.9047587967882447 - 0.2709187330038746j,
        -0.6572111716718827 - 0.8301614350048730j,
    ),
    5: (
        -0.5905759446119191 + 0.9072067564574549j,
        -0.8515536193688396 + 0.4427174639443326j,
        -0.9264420773877602 + 0.0j,
        -0.8515536193688396 - 0.4427174639443326j,
        -0.5905759446119191 - 0.9072067564574549j,
    ),
    6: (
        -0.5385526816693109 + 0.9616876881954278j,
        -0.7996541858328288 + 0.5621717346937318j,
        -0.9093906830472270 + 0.1856964396793047j,
        -0.9093906830472270 - 0.1856964396793047j,
        -0.7996541858328288 - 0.5621717346937318j,
        -0.5385526816693109 - 0.9616876881954278j,
    ),
    7: (
        -0.4966917256672317 + 1.0025085084544205j,
        -0.7527355434093214 + 0.6504696305522552j,
        -0.8800029341523375 + 0.3216652762307740j,
        -0.9194871556490291 + 0.0j,
        -0.8800029341523375 - 0.3216652762307740j,
        -0.7527355434093214 - 0.6504696305522552j,
        -0.4966917256672317 - 1.0025085084544205j,
    ),
    8: (
        -0.4621740412532123 + 1.0343886811269012j,
        -0.7111381808485397 + 0.7186517314108402j,
        -0.8473250802359334 + 0.4259017538272934j,
        -0.9096831546652910 + 0.1412437976671423j,
        -0.9096831546652910 - 0.1412437976671423j,
        -0.8473250802359334 - 0.4259017538272934j,
        -0.7111381808485397 - 0.7186517314108402j,
        -0.4621740412532123 - 1.0343886811269012j,
    ),
    9: (
        -0.4331415561553621 + 1.0600736701359301j,
        -0.6743622686854763 + 0.7730546212691185j,
        -0.8148021112269013 + 0.5085815689631500j,
        -0.8911217017079759 + 0.2526580934582164j,
        -0.9154957797499037 + 0.0j,
        -0.8911217017079759 - 0.2526580934582164j,
        -0.8148021112269013 - 0.5085815689631500j,
        -0.6743622686854763 - 0.7730546212691185j,
        -0.4331415561553621 - 1.0600736701359301j,
    ),
    10: (
        -0.4083220732868862 + 1.0812748428191246j,
        -0.6417513866988319 + 0.8175836167191021j,
        -0.7837694413101444 + 0.5759147538499949j,
        -0.8688459641284764 + 0.3430008233766310j,
        -0.9091347320900502 + 0.1139583137335511j,
        -0.9091347320900502 - 0.1139583137335511j,
        -0.8688459641284764 - 0.3430008233766310j,
        -0.7837694413101444 - 0.5759147538499949j,
        -0.6417513866988319 - 0.8175836167191021j,
        -0.4083220732868862 - 1.0812748428191246j,
    ),
}
def _poly_from_root_pair(r1: jax.Array, r2: jax.Array) -> jax.Array:
    return jnp.asarray([1.0, jnp.real(-(r1 + r2)), jnp.real(r1 * r2)])


def _poly_from_single_root(r: jax.Array) -> jax.Array:
    return jnp.asarray([1.0, jnp.real(-r), 0.0])


def _odd_ext_1d(x: jax.Array, padlen: int) -> jax.Array:
    if padlen < 1:
        return x
    if x.shape[0] <= padlen:
        raise ValueError(f"Input length must be > padlen={padlen}")

    left = 2.0 * x[0] - x[1 : padlen + 1][::-1]
    right = 2.0 * x[-1] - x[-padlen - 1 : -1][::-1]
    return jnp.concatenate([left, x, right], axis=0)


def bessel_lowpass_sos(wn: float | jax.Array, poles: int = 4) -> jax.Array:
    """Create digital low-pass Bessel SOS coefficients.

    `wn` is normalized to Nyquist and must lie in (0, 1).
    `poles` is the filter order in [1, 10].
    """
    if poles not in _BESSEL_PHASE_POLES:
        raise ValueError("poles must be an integer in [1, 10]")

    n = poles
    fs = 2.0
    wc = 2.0 * fs * jnp.tan(jnp.pi * wn / fs)

    proto_poles = jnp.asarray(_BESSEL_PHASE_POLES[n])
    analog_poles = proto_poles * wc
    digital_poles = (2.0 * fs + analog_poles) / (2.0 * fs - analog_poles)

    second_order_rows = []
    for k in range(n // 2):
        p1 = digital_poles[k]
        p2 = digital_poles[n - 1 - k]
        b = jnp.asarray([1.0, 2.0, 1.0])
        a = _poly_from_root_pair(p1, p2)
        second_order_rows.append(jnp.concatenate([b / a[0], a / a[0]]))

    sos_rows = second_order_rows
    if n % 2 == 1:
        p = digital_poles[n // 2]
        b = jnp.asarray([1.0, 1.0, 0.0])
        a = _poly_from_single_root(p)
        sos_rows = [jnp.concatenate([b / a[0], a / a[0]])] + sos_rows

    sos = jnp.stack(sos_rows, axis=0)

    def section_dc_gain(row: jax.Array) -> jax.Array:
        b0, b1, b2, a0, a1, a2 = row
        return (b0 + b1 + b2) / (a0 + a1 + a2)

    gain = jnp.prod(jax.vmap(section_dc_gain)(sos))
    return sos.at[0, :3].set(sos[0, :3] / gain)


def _biquad_df2t(
    row: jax.Array,
    x: jax.Array,
    zi: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    b0, b1, b2, a0, a1, a2 = row
    b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
    a1, a2 = a1 / a0, a2 / a0

    state = jnp.zeros((2,)) if zi is None else zi

    def step(z: jax.Array, x_n: jax.Array) -> tuple[jax.Array, jax.Array]:
        s1, s2 = z
        y_n = b0 * x_n + s1
        s1_new = b1 * x_n - a1 * y_n + s2
        s2_new = b2 * x_n - a2 * y_n
        return jnp.asarray([s1_new, s2_new]), y_n

    zf, y = lax.scan(step, state, x)
    return y, zf


def sosfilt(
    sos: jax.Array,
    x: jax.Array,
    zi: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    n_sections = int(sos.shape[0])
    if zi is None:
        zi = jnp.zeros((n_sections, 2))
    else:
        if zi.shape != (n_sections, 2):
            raise ValueError(f"zi must have shape {(n_sections, 2)}")

    def section_step(
        signal_in: jax.Array,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        row, zi_row = inputs
        signal_out, zf_row = _biquad_df2t(row, signal_in, zi_row)
        return signal_out, zf_row

    y, zf = lax.scan(section_step, x, (sos, zi))
    return y, zf


def sosfilt_zi(sos: jax.Array) -> jax.Array:
    rows = []
    gain = 1.0

    for i in range(int(sos.shape[0])):
        b0, b1, b2, a0, a1, a2 = sos[i]
        b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
        a1, a2 = a1 / a0, a2 / a0

        a_mat = jnp.asarray([[-a1, 1.0], [-a2, 0.0]])
        b_vec = jnp.asarray([b1 - a1 * b0, b2 - a2 * b0])
        zi = jnp.linalg.solve(jnp.eye(2) - a_mat, b_vec) * gain
        rows.append(zi)

        gain = gain * (b0 + b1 + b2) / (1.0 + a1 + a2)

    return jnp.stack(rows, axis=0)


def sosfiltfilt(
    sos: jax.Array,
    x: jax.Array,
    padlen: int | None = None,
) -> jax.Array:
    if x.ndim != 1:
        raise ValueError("Only 1D input is supported")

    n_sections = int(sos.shape[0])
    if padlen is None:
        padlen = 3 * (2 * n_sections + 1)

    if x.shape[0] <= padlen:
        raise ValueError(f"Input length must be > padlen={padlen}")

    zi = sosfilt_zi(sos)
    x_pad = _odd_ext_1d(x, padlen)

    y_fwd, _ = sosfilt(sos, x_pad, zi * x_pad[0])
    y_rev = y_fwd[::-1]
    y_bwd, _ = sosfilt(sos, y_rev, zi * y_rev[0])
    y = y_bwd[::-1]
    return y[padlen:-padlen]


def allen_bessel_sosfiltfilt(
    v: jax.Array,
    t: jax.Array,
    cutoff_khz: float = 10.0,
) -> jax.Array:
    """Allen-style low-pass filtering via JAX Bessel SOS + filtfilt."""
    if t.ndim != 1 or v.ndim != 1 or t.shape[0] != v.shape[0]:
        raise ValueError("allen_bessel_sosfiltfilt expects 1D aligned t and v")
    if t.shape[0] < 2:
        return v

    delta_t = t[1] - t[0]
    sample_freq = 1.0 / delta_t
    wn = (cutoff_khz * 1e3) / (sample_freq / 2.0)

    sos = bessel_lowpass_sos(wn, poles=4)
    return sosfiltfilt(sos, v)
