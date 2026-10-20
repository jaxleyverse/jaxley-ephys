# jaxley-ephys

`jaxley-ephys` is a JAX-first electrophysiology feature toolbox.

The core goal is to provide Allen-style ephys features implemented in JAX and
extending the set of features beyond what Allen offers. 

All features also available in the AllenSDK are validated against specimens from
the AllenDB.

## Setup

```bash
pip install git+https://github.com/jaxleyverse/jaxley-ephys
# also allen_ephys, the maintained copy of the AllenSDK feature extraction:
pip install "jaxley-ephys[allen] @ git+https://github.com/jaxleyverse/jaxley-ephys"
```

For development (tests, examples):

```bash
git clone https://github.com/jaxleyverse/jaxley-ephys
cd jaxley-ephys
uv sync  # includes the dev dependency group
```

## Quickstart

```python
import jax

jax.config.update("jax_enable_x64", True)  # float64 matches Allen exactly (float32: ~1e-4)

from jaxley_ephys import detect_spikes, spike_features, sweep_features

# One sweep: t (s), v (mV) and optionally i (pA), all of shape [T];
# the stimulus runs from start to end (s).
sp = detect_spikes(t, v, i, start=1.02, end=2.02)
n = int(sp.n_spikes)
spikes = spike_features(sp)  # Allen's spike table as [K] arrays, valid up to n
print(spikes["threshold_t"][:n], spikes["width"][:n])
features = sweep_features(sp, stimulus="long_square")  # adapt, latency, tau, ...
```

Parameter names, defaults and units follow Allen's `EphysSweepFeatureExtractor`
(t in s, v in mV, i in pA, dV/dt in V/s; interspike intervals and `tau` in ms,
`latency` and `width` in s). Undefined values are NaN, missing indices -1.
All functions are compiled internally, so repeated calls on traces of the same
shape are fast. Sweeps of equal length can be batched with `jax.vmap`. Spikes
are stored in buffers of static size `max_spikes` (default 256); inside
`jit`/`vmap` no error can be raised when they overflow, so check `sp.overflow`:

```python
@jax.jit
def analyse(t, v, i):
    sp = detect_spikes(t, v, i, start=1.02, end=2.02)
    return sp.overflow, sweep_features(sp, stimulus="long_square")

overflow, features = jax.vmap(analyse)(t, v, i)  # t, v, i of shape [S, T]
```

Cell-level features follow Allen's `extract_cell_features`: spike features ->
sweep features -> sweepset features per stimulus type -> cell features, named as
in the Allen Cell Types Database (`ri`, `tau`, `vrest`, `f_i_curve_slope`,
`threshold_v_long_square`, ...):

```python
from jaxley_ephys import extract_cell_features

# Lists of (t, v, i, start, end) per stimulus type; sweep lengths may differ.
cell = extract_cell_features(long_squares, short_squares, ramps)
cell["cell"]["ri"], cell["long_squares"]["rheobase_i"]
```

The sweepset functions (`long_square_features`, `short_square_features`,
`ramp_features`) only take the stacked per-sweep features, so they are cheap
and can be `vmap`ped over cells.

See `examples/` for a walkthrough on Allen Cell Types data.

## Performance

On CPU (one cached Allen cell, after compilation), jaxley_ephys analyses a whole
cell about 2.5x faster than the vendored Allen code, ramps about 7x faster, long
squares as fast and short squares at about 0.7x; batches of short (e.g.
simulated) traces take about 3 ms per trace, 2.5x faster.

- Everything is compiled on the first call for each trace length (about 2 s per
  function set); later calls with the same length reuse it. JAX can keep
  compiled code across sessions:

  ```python
  jax.config.update("jax_compilation_cache_dir", "/path/to/jax_cache")
  jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
  ```

- `detect_spikes(..., t_cutoff=...)` stops the analysis at `t_cutoff`; later
  samples are NaN in the returned traces. The default, 20 ms after the end of the
  analysis window, changes no result (nothing in Allen's algorithm reaches
  further) and makes long sweeps with short stimuli fast. Samples are only
  skipped when `t` is concrete; with a traced `t` (inside your own `jit`), an
  explicit `t_cutoff` masks them (same results) and the default analyses the
  whole sweep.
- Sweeps of different lengths can be batched: pad them with NaN times.
- On CPU, loop over recorded sweeps: `vmap` cannot skip work per sweep (e.g.
  the 1 kHz dV/dt of sweeps without spikes) and is slower for long sweeps; for
  short traces both take about the same time. Batching is meant for GPUs, which
  are not tuned yet.
- `uv run python benchmarks/run.py` measures this on your machine.

## Repository structure

- `jaxley_ephys/`: the library
  - `dvdt.py`: dV/dt with Allen's 4-pole Bessel low-pass filter (`calculate_dvdt`)
  - `spikes.py`: spike detection (`detect_spikes`, `Spikes`) and spike features
    (`spike_features`)
  - `sweep.py`: sweep features (`sweep_features` and the individual features)
  - `sweepset.py`: sweepset features for long squares, short squares and ramps,
    and the adjusted short-square detection parameters
  - `cell.py`: cell features (`cell_features`) and the whole pipeline for a cell
    (`extract_cell_features`)
  - `_ops.py`: array primitives shared by the above
- `allen_ephys/`: maintained copy of the (unmaintained) AllenSDK feature
  extraction with fixes for current dependencies and edge cases; the parity
  reference (see `allen_ephys/ALLEN_EPHYS_FIXES.md`)
- `tests/`: parity tests of `jaxley_ephys` against `allen_ephys`;
  `tests/allen_ephys/` tests `allen_ephys` itself (run with `--allen`)
- `examples/`: notebooks

## Testing

```bash
uv run python tests/load_testdata.py    # download the Allen test specimens
uv run python tests/build_reference.py  # run allen_ephys on them once
uv run python -m pytest                 # --allen: also test allen_ephys
```

## Licensing

- Original repository code is MIT-licensed (`LICENSE`).
- Vendored Allen files in `allen_ephys/` follow Allen's original license terms
  (`THIRD_PARTY_LICENSES/ALLENSDK_LICENSE.txt`).

## Disclaimer 
- This package was almost entirely vibe-coded so be aware of this when using it.
