# jaxley-ephys

Electrophysiological features of voltage recordings in [JAX](https://docs.jax.dev):
spike detection and spike, sweep, sweepset and cell features that match the
[AllenSDK](https://github.com/AllenInstitute/AllenSDK) feature extraction behind the
[Allen Cell Types Database](https://celltypes.brain-map.org). Everything is
compiled, so it is fast, works on batches with `jax.vmap` and can be differentiated
with `jax.grad`.

```python
import jax

jax.config.update("jax_enable_x64", True)  # float64: matches Allen exactly

from jaxley_ephys import detect_spikes, spike_features, sweep_features

# One sweep: t (s), v (mV) and optionally i (pA); the stimulus lasts from start to end
sp = detect_spikes(t, v, i, start=1.02, end=2.02)
spikes = spike_features(sp)  # Allen's spike table
features = sweep_features(sp, stimulus="long_square")  # avg_rate, latency, tau, ...
```

- **Matches Allen:** on every sweep of eight cells of the Allen Cell Types
  Database, spike tables, sweep features and cell features agree with a
  maintained copy of the AllenSDK feature extraction to floating-point precision
  (see [](allen.md)).
- **Fast:** about 2.5 times faster than the AllenSDK code for a whole cell on a
  CPU, more for long recordings.
- **JAX:** compiled once per trace length, batched with `jax.vmap`, and in float32
  within about 1e-4 of Allen.

```{toctree}
:hidden:
:caption: Getting started

installation
tutorials
performance
```

```{toctree}
:hidden:
:caption: Reference

api
allen
changelog
```
