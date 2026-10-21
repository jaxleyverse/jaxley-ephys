# jaxley-ephys

`jaxley-ephys` provides differentiable implementations of the AllenSDK ephys features
in JAX.

Documentation: <https://jaxley-ephys.readthedocs.io>

## Installation

```bash
pip install git+https://github.com/jaxleyverse/jaxley-ephys
```

## Quickstart

```python
import jax

jax.config.update("jax_enable_x64", True)  # float64 matches Allen exactly (float32: ~1e-4)

from jaxley_ephys import detect_spikes, extract_cell_features, spike_features, sweep_features

# One sweep: t (s), v (mV) and optionally i (pA); the stimulus lasts from start to end (s)
sp = detect_spikes(t, v, i, start=1.02, end=2.02)
spikes = spike_features(sp)  # Allen's spike table
features = sweep_features(sp, stimulus="long_square")  # avg_rate, latency, tau, ...
# bursts=True, pauses=True and delays=True add Allen's burst, pause and delay metrics

# A whole cell: lists of (t, v, i, start, end) per stimulus type
cell = extract_cell_features(long_squares, short_squares, ramps)
cell["cell"]["ri"]  # cell features are named like in the Allen Cell Types Database
```

Every feature can be differentiated, e.g. to fit a model to a recording:

```python
def loss(v):
    sp = detect_spikes(t, v, i, start=1.02, end=2.02, interpolate=True)  # smooth event times
    features = sweep_features(sp, nan_to=0.0)  # NaN (undefined) would make gradients NaN
    return (features["mean_isi"] - 50.0) ** 2  # target: a mean interspike interval of 50 ms

grad = jax.grad(loss)(v)  # [T]; through a differentiable simulator, w.r.t. its parameters
```

`examples/06_fitting.ipynb` fits a Hodgkin-Huxley model to features this way.

Parameter names, defaults and units (with exceptions) follow the AllenSDK. 
The notebooks in `examples/` walk you through the library and how to use it.

## Allen parity

This repository also contains a maintained and working copy of `allen_ephys`, with its 
original license (install with
`pip install "jaxley-ephys[allen] @ git+https://github.com/jaxleyverse/jaxley-ephys"`).
`jaxley-ephys` matches it on every sweep of eight cells of the Allen Cell Types
Database; the few intended differences are listed in
`allen_ephys/ALLEN_EPHYS_FIXES.md`.

## Repository structure

- `jaxley_ephys/`: the library
- `allen_ephys/`: the maintained copy of the AllenSDK feature extraction
- `examples/`: tutorial notebooks
- `docs/`: documentation (Sphinx)
- `tests/`: parity tests against `allen_ephys`; `benchmarks/`: speed against it

## Development

```bash
git clone https://github.com/jaxleyverse/jaxley-ephys
cd jaxley-ephys
uv sync                                 # with the dev tools (pytest, ruff, AllenSDK)
uv run python tests/load_testdata.py    # download the Allen test cells
uv run python tests/build_reference.py  # run allen_ephys on them once
uv run python -m pytest                 # --allen: also test allen_ephys
```

## Licensing

- Original repository code is MIT-licensed (`LICENSE`).
- Vendored Allen files in `allen_ephys/` follow Allen's original license terms
  (`THIRD_PARTY_LICENSES/ALLENSDK_LICENSE.txt`).

## Disclaimer
- This package was almost entirely vibe-coded so be aware of this when using it.
