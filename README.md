# jaxley-ephys

`jaxley-ephys` computes electrophysiological features of voltage recordings in
JAX: spike detection and spike, sweep, sweepset and cell features that match the
AllenSDK feature extraction. It is fast (a whole cell of the Allen Cell Types
Database takes about 2 s on a CPU, 2.5x faster than the AllenSDK code), compiled,
batches with `jax.vmap` and is differentiable.

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

# A whole cell: lists of (t, v, i, start, end) per stimulus type
cell = extract_cell_features(long_squares, short_squares, ramps)
cell["cell"]["ri"]  # cell features are named like in the Allen Cell Types Database
```

Parameter names, defaults and units follow the AllenSDK. The notebooks in
`examples/` walk through spikes, sweep, sweepset and cell features, and batches.

## Allen parity

The AllenSDK feature extraction is no longer maintained. This repository contains
a maintained copy, `allen_ephys`, fixed for current NumPy, SciPy and pandas, with
its original license (install with
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
