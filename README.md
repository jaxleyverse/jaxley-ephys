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

jax.config.update("jax_enable_x64", True)  # float64, needed to match Allen exactly

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

See `examples/` for a walkthrough on Allen Cell Types data.

## Repository structure

- `jaxley_ephys/`: the library
  - `dvdt.py`: dV/dt with Allen's 4-pole Bessel low-pass filter (`calculate_dvdt`)
  - `spikes.py`: spike detection (`detect_spikes`, `Spikes`) and spike features
    (`spike_features`)
  - `sweep.py`: sweep features (`sweep_features` and the individual features)
  - `_ops.py`: array primitives shared by the above
  - `features/`: sweepset features for long squares, short squares and ramps
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
