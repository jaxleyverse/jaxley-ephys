# jaxley-ephys

`jaxley-ephys` is a JAX-first electrophysiology feature toolbox.

The core goal is to provide Allen-style ephys features implemented in JAX and
extending the set of features beyond what Allen offers. 

All features also available in the AllenSDK are validated against specimens from
the AllenDB.

## Setup

```bash
git clone https://github.com/jaxleyverse/jaxley-ephys
uv sync
```

## Repository structure

- `allen_ephys/`: Fixed up Allen code (w. minor usability improvemnts) for parity testing
- `jaxley_ephys/`: JAX features
  - `spike.py`: Spike level features
  - `cell.py`: Cell level features (selects from sweepset features (ramp, long_square, short_square))
  - `long_square/`: Sweep and sweepset level features for long square stimuli 
    - `sweep.py`
    - `sweepset.py`
  - `short_square/`: Sweep and sweepset level features for short square stimuli
    - `sweep.py`
    - `sweepset.py`
  - `ramp/`: Sweep and sweepset level features for ramp stimuli
    - `sweep.py`
    - `sweepset.py`
  - `bessel.py`: Custom Bessel filter implementation in JAX for filtering
  - `events.py`: Spike detection
- `tests/allen_ephys/`: Allen verification

## Licensing

- Original repository code is MIT-licensed (`LICENSE`).
- Vendored Allen files in `allen_ephys/` follow Allen's original license terms
  (`THIRD_PARTY_LICENSES/ALLENSDK_LICENSE.txt`).

## Disclaimer 
- This package was almost entirely vibe-coded so be aware of this when using it.
