import jax.numpy as jnp
import jax.random as jr
import jax
from jax import Array

C_MUF_PER_CM2 = 1.0
G_NA = 120.0
G_K = 36.0
G_L = 0.3
E_NA = 50.0
E_K = -77.0
E_L = -54.387
SIM_DT_MS = 0.025
CURRENT_SCALE_UA_PER_CM2 = 16.0

def _alpha_n(v_mV: Array | float) -> Array:
    x = v_mV + 55.0
    return jnp.where(jnp.abs(x) < 1e-6, 0.1, 0.01 * x / (1.0 - jnp.exp(-x / 10.0)))


def _beta_n(v_mV: Array | float) -> Array:
    return 0.125 * jnp.exp(-(v_mV + 65.0) / 80.0)


def _alpha_m(v_mV: Array | float) -> Array:
    x = v_mV + 40.0
    return jnp.where(jnp.abs(x) < 1e-6, 1.0, 0.1 * x / (1.0 - jnp.exp(-x / 10.0)))


def _beta_m(v_mV: Array | float) -> Array:
    return 4.0 * jnp.exp(-(v_mV + 65.0) / 18.0)


def _alpha_h(v_mV: Array | float) -> Array:
    return 0.07 * jnp.exp(-(v_mV + 65.0) / 20.0)


def _beta_h(v_mV: Array | float) -> Array:
    return 1.0 / (1.0 + jnp.exp(-(v_mV + 35.0) / 10.0))

def _step_current(t: Array, stim_delay: float, stim_dur: float, stim_amp: float) -> Array:
    step_mask = (t >= stim_delay) & (t < (stim_delay + stim_dur))
    return jnp.where(step_mask, stim_amp, 0.0)


@jax.jit
def _simulate_hh_voltage(
    i_ext_uA_cm2: Array, dt_ms: float, v0_mV: float = -65.0
) -> Array:
    m0 = _alpha_m(v0_mV) / (_alpha_m(v0_mV) + _beta_m(v0_mV))
    h0 = _alpha_h(v0_mV) / (_alpha_h(v0_mV) + _beta_h(v0_mV))
    n0 = _alpha_n(v0_mV) / (_alpha_n(v0_mV) + _beta_n(v0_mV))

    def step(state, i_ext):
        v_mV, m, h, n = state
        i_na = G_NA * (m**3) * h * (v_mV - E_NA)
        i_k = G_K * (n**4) * (v_mV - E_K)
        i_l = G_L * (v_mV - E_L)

        dv = (i_ext - i_na - i_k - i_l) / C_MUF_PER_CM2
        dm = _alpha_m(v_mV) * (1.0 - m) - _beta_m(v_mV) * m
        dh = _alpha_h(v_mV) * (1.0 - h) - _beta_h(v_mV) * h
        dn = _alpha_n(v_mV) * (1.0 - n) - _beta_n(v_mV) * n

        m_next = jnp.clip(m + dt_ms * dm, 0.0, 1.0)
        h_next = jnp.clip(h + dt_ms * dh, 0.0, 1.0)
        n_next = jnp.clip(n + dt_ms * dn, 0.0, 1.0)
        v_next = v_mV + dt_ms * dv
        return (v_next, m_next, h_next, n_next), v_next

    _, v_trace = jax.lax.scan(step, (v0_mV, m0, h0, n0), i_ext_uA_cm2)
    return v_trace


def simulate_sweep(
    dt=1e-4,
    t_max=0.1,
    stim_amp=0.5,
    stim_delay=0.02,
    stim_dur=0.06,
    noise_level: float = 0.1,
    key: Array = jr.key(0),
) -> tuple[Array, Array, Array]:
    dt_sim = SIM_DT_MS * 1e-3
    ts_sim = jnp.arange(0.0, t_max, dt_sim)

    i_cmd_sim = _step_current(ts_sim, stim_delay, stim_dur, stim_amp)
    i_sim = i_cmd_sim * CURRENT_SCALE_UA_PER_CM2
    v_sim = _simulate_hh_voltage(i_sim, SIM_DT_MS)

    ts_obs = jnp.arange(0.0, t_max, dt)
    v_obs = jnp.interp(ts_obs, ts_sim, v_sim)
    v_obs += noise_level * jr.normal(key, shape=v_obs.shape)
    i_obs = _step_current(ts_obs, stim_delay, stim_dur, stim_amp)
    return ts_obs, v_obs, i_obs


def simulate_sweepset(num_sweeps: int = 10, i_range: tuple = (-1.0, 1.0), **kwargs):
    keys = jr.split(jr.key(0), num_sweeps)
    i_min, i_max = i_range
    amps = jnp.linspace(i_min, i_max, num_sweeps)
    vsim = jax.vmap(
        lambda amp, key: simulate_sweep(stim_amp=amp, key=key, **kwargs)
    )
    ts_obs, v_obs, i_obs = vsim(amps, keys)
    return ts_obs, v_obs, i_obs
