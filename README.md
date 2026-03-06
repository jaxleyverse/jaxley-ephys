# jaxley-ephys

`jaxley-ephys` is a JAX-first electrophysiology feature toolbox.

The core goal is to provide Allen-style ephys features implemented in JAX, and
then extend them with differentiable variants for optimization and learning
workflows.

AllenSDK is used for **ground-truth validation**.

## Setup

```bash
git clone https://github.com/jaxleyverse/jaxley-ephys
uv sync
```

## Repository structure

- `jaxley_ephys/`: JAX implementation package (main product surface)
- `allen_ephys/`: Allen reference code (incl. fixups) for parity/ground truth
- `tests/allen_ephys/`: Allen verification

## Licensing

- Original repository code is MIT-licensed (`LICENSE`).
- Vendored Allen files in `allen_ephys/` follow Allen's original license terms
  (`THIRD_PARTY_LICENSES/ALLENSDK_LICENSE.txt`).
