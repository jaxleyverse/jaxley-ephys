# Changelog

## 0.1.0 (unreleased)

First release.

- Spike detection (`detect_spikes`) and Allen's spike table (`spike_features`):
  thresholds, peaks, troughs, upstrokes, downstrokes, fast and slow troughs,
  afterdepolarizations, ISI types and widths.
- Sweep features (`sweep_features`): firing rate, latency, interspike intervals
  (ms), adaptation, baseline, voltage deflection, `tau`, `sag` and stimulus
  amplitude.
- Firing-pattern metrics, like Allen's `pause_metrics`, `burst_metrics` and
  `delay_metrics`: as functions of the same names and, on request, as sweep
  features (`sweep_features(..., bursts=True, pauses=True, delays=True)`). The
  delay's time constant is in ms; its double-exponential fit finds separated time
  constants where Allen's optimizer stays at its start.
- Sweepset features of long squares, short squares and ramps, and cell features
  named like the Allen Cell Types Database (`cell_features`,
  `extract_cell_features`); `threshold_i_short_square` is the short-square
  stimulus amplitude, as in Allen's current pipeline (ipfx).
- Parity with the AllenSDK feature extraction (the maintained copy `allen_ephys`)
  on every sweep of eight Allen cells; differences are documented in
  `allen_ephys/ALLEN_EPHYS_FIXES.md`.
- Compiled once per trace length, batches with `jax.vmap` (also of sweeps of
  unequal length, padded with NaN times); float32 within about 1e-4 of Allen;
  `t_cutoff` skips samples after the analysis window.
- Gradients of every feature (`jax.grad`), finite also where features are NaN:
  exact for voltages and the fits (implicit differentiation of `tau` and the
  delay), of times between samples for event times and what is computed from
  them, of soft spike counts for `avg_rate`, the f-I slope and the rheobase; the
  values stay Allen's. `detect_spikes(..., interpolate=True)` returns the times
  between samples as values, `count_width` sets the reach of the soft counts, and
  `nan_to` replaces NaN features for losses. Example: fitting a Hodgkin-Huxley
  model to features (`examples/06_fitting.ipynb`).
- `allen_ephys`: a maintained copy of the AllenSDK feature extraction
  (`jaxley-ephys[allen]`), with fixes such as for `detect_bursts`, which raised
  on every spike train with burst candidates.
