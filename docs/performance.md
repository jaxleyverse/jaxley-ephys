# Performance

On a CPU, jaxley-ephys analyses a whole cell of the Allen Cell Types Database about
2.5 times faster than the AllenSDK code, long ramps about 7 times faster and long
squares about as fast; batches of short (e.g. simulated) traces take about 3 ms per
trace. `uv run python benchmarks/run.py` measures this on your machine.

The firing-pattern metrics, which `sweep_features` adds on request, take about
0.2 ms (`pauses=True`) and 0.4 ms (`bursts=True`) per long square, 2.5-5 times
faster than Allen's; the delay (`delays=True`), which fits exponentials to the
rise before the first spike, takes about 5 ms (up to 60 ms), about as fast as
Allen's fit. Its cost grows with the time from `start` to the first spike, not
with the length of the sweep.

## Compilation

Everything is compiled on the first call for each trace length (about 2 s per set
of functions); later calls with the same length reuse it. JAX can keep compiled
code across sessions:

```python
jax.config.update("jax_compilation_cache_dir", "/path/to/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
```

## Samples after the analysis window

`detect_spikes(..., t_cutoff=...)` stops the analysis at `t_cutoff`; later samples
are NaN in the returned traces. The default, 20 ms after the end of the analysis
window, changes no result (nothing in Allen's algorithm reaches further) and makes
long sweeps with short stimuli fast. Samples are only skipped when `t` is concrete;
with a traced `t` (inside your own `jit`), an explicit `t_cutoff` masks them (same
results) and the default analyses the whole sweep.

## Batches

Sweeps of the same length can be batched with `jax.vmap`; sweeps of different
lengths, too, if the shorter ones are padded with NaN times. Inside `jit` and
`vmap`, check `Spikes.overflow`: no error can be raised there if a sweep has more
putative spikes than `max_spikes`.

On a CPU, a loop over recorded sweeps is the fastest: `vmap` cannot skip work per
sweep (e.g. the 1 kHz dV/dt of sweeps without spikes), so it is slower for long
sweeps, and about as fast for short traces. Batching is meant for GPUs, which are
not tuned yet.

## Precision

float64 (`jax_enable_x64`) matches Allen exactly. float32 is faster on GPUs and uses
half the memory; features then deviate by about 1e-4 (`tau` and `sag` by up to
1.5e-3).
