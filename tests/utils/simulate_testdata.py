import jaxley as jx
import jax.numpy as jnp
import jax.random as jr
import jax
from jaxley.channels import HH
from jax import Array


comp = jx.Compartment()
comp.insert(HH())
comp.set("v", -65.0)
comp.set("length", 100.0)
comp.set("radius", 10.0)
comp.init_states()
comp.record("v")

def simulate_sweep(
    dt = 1e-4,
    t_max = 0.2,
    stim_amp = 0.5,
    stim_delay = 0.002,
    stim_dur = 0.16,
    noise_level: float = 0.1,
    key: int = jr.key(0),
) -> tuple[Array, Array, Array]:
    
    dt_sim = 0.025 # ms
    t_max_ms = t_max * 1e3 # convert to ms
    stim_delay_ms = stim_delay * 1e3 # convert to ms
    stim_dur_ms = stim_dur * 1e3 # convert to ms

    ts_sim_ms = jnp.arange(0.0, t_max_ms + dt_sim, dt_sim)
    i_sim = jx.step_current(stim_delay_ms, stim_dur_ms, stim_amp, dt_sim, t_max_ms)

    data_stimuli = comp.data_stimulate(i_sim, None)
    v_sim = jx.integrate(comp, data_stimuli=data_stimuli, delta_t=dt_sim, t_max=t_max_ms)[0]
    
    ts_obs = jnp.arange(0.0, t_max, dt)
    v_obs = jnp.interp(ts_obs, ts_sim_ms * 1e-3, v_sim)
    v_obs += noise_level * jr.normal(key, shape=v_obs.shape)
    i_obs = jnp.interp(ts_obs, ts_sim_ms * 1e-3, i_sim)
    return ts_obs, v_obs, i_obs


def simulate_sweepset(num_sweeps: int = 10, i_range: tuple = (-1.0, 1.0), **kwargs):
    keys = jr.split(jr.key(0), num_sweeps)
    i_min, i_max = i_range
    amps = jnp.linspace(i_min, i_max, num_sweeps)
    vsim = jax.vmap(lambda amp, key: simulate_sweep(stim_amp=amp, key=key, **kwargs))
    ts_obs, v_obs, i_obs = vsim(amps, keys)
    return ts_obs, v_obs, i_obs