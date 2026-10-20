# Changelog

## 0.1.0 (unreleased)

First release.

- Spike detection (`detect_spikes`) and Allen's spike table (`spike_features`):
  thresholds, peaks, troughs, upstrokes, downstrokes, fast and slow troughs,
  afterdepolarizations, ISI types and widths.
- Sweep features (`sweep_features`): firing rate, latency, interspike intervals
  (ms), adaptation, baseline, voltage deflection, `tau`, `sag` and stimulus
  amplitude.
- Sweepset features of long squares, short squares and ramps, and cell features
  named like the Allen Cell Types Database (`cell_features`,
  `extract_cell_features`).
- Parity with the AllenSDK feature extraction (the maintained copy `allen_ephys`)
  on every sweep of eight Allen cells; differences are documented in
  `allen_ephys/ALLEN_EPHYS_FIXES.md`.
- Compiled once per trace length, batches with `jax.vmap` (also of sweeps of
  unequal length, padded with NaN times), differentiable; float32 within about
  1e-4 of Allen; `t_cutoff` skips samples after the analysis window.
- `allen_ephys`: a maintained copy of the AllenSDK feature extraction
  (`jaxley-ephys[allen]`).
