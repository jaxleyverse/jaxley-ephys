# Installation

jaxley-ephys needs Python 3.12 or later. Install it from GitHub:

```bash
pip install git+https://github.com/jaxleyverse/jaxley-ephys
```

This installs JAX for CPUs; for GPUs, follow the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

The repository also contains `allen_ephys`, a maintained copy of the AllenSDK
feature extraction that jaxley-ephys is tested against (see [](allen.md)). It needs
pandas and SciPy:

```bash
pip install "jaxley-ephys[allen] @ git+https://github.com/jaxleyverse/jaxley-ephys"
```

## Precision

JAX computes in float32 unless `jax_enable_x64` is set. With float64, the results
match Allen exactly; in float32 they deviate slightly (features by about 1e-4,
`tau` and `sag` by up to 1.5e-3).

## Development

```bash
git clone https://github.com/jaxleyverse/jaxley-ephys
cd jaxley-ephys
uv sync                                 # with the dev tools (pytest, ruff, AllenSDK)
uv run python tests/load_testdata.py    # download the Allen test cells
uv run python tests/build_reference.py  # run allen_ephys on them once
uv run python -m pytest                 # --allen: also test allen_ephys
uv run --group docs sphinx-build -W docs docs/_build/html  # these docs
```
