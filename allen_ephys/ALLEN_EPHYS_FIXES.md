# Vendored Allen Ephys Fixes

Changes relative to original AllenSDK implementation.

## 1) Compatibility with newer dependencies
- `allen_ephys/ephys_extractor.py`
  - `spikes_df.ix[...]` -> `spikes_df.loc[...]` (pandas compatibility)
  - `six.iteritems(...)` -> `.items()` (Python 3 compatibility)
- `allen_ephys/ephys_features.py`
  - `calculate_dvdt`: guard `dt == 0` before division (NumPy warning-safe)

## 2) Robustness fixes for edge-case sweeps
- `allen_ephys/ephys_extractor.py`
  - `pause_metrics`: neutral return for no-spike sweeps
  - `burst_metrics`: neutral return / guarded burst detection failures
  - `delay_metrics`: warning-safe and exception-safe prespike fit path
  - long-square rheobase tie-break: when multiple spiking sweeps share the
    minimal stimulus amplitude, select the candidate with highest firing rate
    (`avg_rate`) for stable alignment with Allen DB rheobase sweep IDs. This
    matches the DB on 7 of the 8 cached specimens; on 605889373 the DB picks
    the first minimal-amplitude sweep (53) instead of the highest-rate one (57),
    and the upstream rule (first minimal-amplitude sweep) fails on 469801138
    and 537256313, so the DB choice likely depends on information not in the
    cached data (e.g. QC exclusions)
  - `membrane_time_constant`: avoid `np.nanmean` all-NaN warning by averaging
    only finite `tau` values and returning `nan` when none are finite
- `allen_ephys/ephys_features.py`
  - `fit_membrane_time_constant`: fix RMSE calculation from
    `sqrt(mean(pred - v))` to `sqrt(mean((pred - v)^2))` to avoid invalid
    square-root warnings and to use a mathematically correct RMSE.
- `allen_ephys/ephys_extractor.py`
  - `estimate_time_constant`: pass `min_rsme=1.0` to
    `fit_membrane_time_constant` so corrected RMSE values for valid traces are
    not rejected wholesale by the legacy `1e-4` threshold.
  - `estimate_time_constant`: when time-constant fitting fails (fit-window
    detection or optimizer failures), emit `RuntimeWarning` including the
    quoted exception and return `nan` instead of raising. This keeps downstream
    sweep/cell feature computation resilient while surfacing the failure cause.
  - `estimate_time_constant` / `estimate_sag`: gate to hyperpolarizing sweeps
    and return `nan` for non-hyperpolarizing contexts. This avoids attempting
    hyperpolarization-only computations on depolarizing traces.

## 3) Unit alignment with Allen DB
- `allen_ephys/ephys_extractor.py`
  - `mean_isi`: s -> ms (`1e3 * isis.mean()`)
  - `median_isi`, `first_isi`: s -> ms, so all interspike-interval features
    share the unit of the DB's `avg_isi` (2026-10-19)
  - `tau`: s -> ms (`1e3 / inv_tau`)
  - `membrane_time_constant` docs updated to ms

## 4) DB parity mapping convention used in tests
- `tests/allen_ephys/test_parity.py` (runs with `--allen`)
  - long-square adaptation/latency/avg-ISI are mapped from the DB
    `thumbnail_sweep_id` when present; this is Allen's hero sweep (the
    smallest-amplitude spiking long square in (rheobase + 39, rheobase + 61)
    pA) on all cached specimens
  - long-square first-spike timing/voltage fields are mapped from the first
    spike of `rheobase_sweep`; `threshold_i_long_square` is mapped to
    `rheobase_i`
  - extractor `long_squares.input_resistance` is mapped to DB `ri`
    (Allen DB `input_resistance_mohm` is not a direct output of
    `EphysCellFeatureExtractor`)
  - `threshold_i_short_square` uses a DB-aligned aggregation of the first-spike
    `threshold_i` over the common-amplitude short-square sweeps: the mean for
    small sets (<= 2 sweeps), otherwise the mean of the nonzero values, falling
    back to the common stimulus amplitude when no value is available
  - short-square `slow_trough_t/v` use first-spike `slow_trough_t/v` with
    fallback to `trough_t/v` when slow-trough values are not present
  - sweep-level `num_spikes` parity uses NWB `index_range` windows to align
    with Allen DB sweep metadata semantics
  - sweep-level `peak_deflect` parity compares value and index separately
    (`peak_deflect`, `peak_deflect_index`)
- These DB-matching heuristics are not part of Allen's code; they stay in the
  DB comparison tests and are not reproduced by jaxley_ephys.

## 5) Reference outputs for the jaxley_ephys parity tests
- `tests/utils/allen_reference.py` (run via `tests/build_reference.py`) runs
  this vendored code once per cached specimen with the parameters of
  `cell_extractor_for_nwb` and stores the results in
  `tests/testdata/allen_reference/`:
  - inputs are float64 copies of the float32 NWB traces (the JAX side gets the
    same arrays)
  - stimulus amplitude: long squares use `_step_stim_amp`, short squares use
    `_short_step_stim_amp` (set on all short squares, not only the
    common-amplitude ones), ramps have no calculator (`stim_amp = nan`)
  - cell level: the `EphysCellFeatureExtractor` outputs, the hero sweep, the
    long-square baseline (`extract_cell_features` overrides the extractor's
    `nanmean` with a plain mean over all long squares) and first-spike means of
    `extract_cell_features`, the adjusted short-square detection parameters, and
    these values under their Allen DB names (`record`, without the DB-matching
    heuristics of section 4)
- Only canonical stimuli are used: `Long Square`, `Ramp`, `Short Square`
  (`tests/utils/allen_dataset.py` skips variants such as `Ramp to Rheobase`
  and `Short Square - Triple` when reading `ephys_sweeps.json`).

## 6) Known differences between jaxley_ephys and this reference
Intended differences that the parity tests account for; on everything else
jaxley_ephys agrees with this reference to floating-point precision.

- **tau: Allen's optimizer stops early.** `fit_membrane_time_constant` calls
  `scipy.optimize.curve_fit` with its default tolerances
  (`ftol = xtol = 1.49e-8`). Levenberg-Marquardt stops before the
  least-squares optimum: on the cached specimens Allen's tau is up to 1.1e-3
  (relative; median 4e-7) away from a fully converged fit. jaxley_ephys fits
  to the optimum, so sweep-level `tau` and the cell-level mean `tau` are
  compared with `rtol = 2e-3` (`tests/utils/parity.py`).
- **tau on ill-posed fits.** Without a clean exponential relaxation (e.g. the
  synthetic fuzz traces) the fit is ill-posed: Allen's local search may hit
  `maxfev` and return NaN, or end in a different local minimum than
  jaxley_ephys. The values (sometimes several seconds) are not meaningful
  either way, so `tests/test_fuzz.py` does not compare `tau`.
- **dt == 0.** `calculate_dvdt` drops samples where `dt == 0`, which shifts
  every later dV/dt index by one relative to `v` (upstream does the same). On
  `tests/testdata/spike_test_var_dt.txt` this moves all threshold, upstroke,
  downstroke and fast-trough indices by one sample; peaks and troughs (from
  `v`) agree. jaxley_ephys keeps dV/dt aligned (NaN at `dt == 0`); see
  `tests/test_allen_fixtures.py`.
- **Baseline without a start time.** With `start=None`, `_get_baseline_voltage`
  searches for a flat interval using `np.abs(dv >= thresh)`; `abs` is applied
  to the comparison, so only dV/dt >= thresh counts as non-flat (upstream
  bug). jaxley_ephys needs a start time for `v_baseline` and returns NaN for
  it (and hence for `tau` and `sag`) without one.
- **Inputs on which Allen raises** (`find_time_index` beyond the end of the
  trace, analysis windows shorter than the 15-sample filter padding, empty
  windows, `argmax` of empty slices, a filter cutoff at or above the Nyquist
  frequency): jaxley_ephys cannot raise under `jit` and returns NaN or -1
  instead (NaN dV/dt for the cutoff, so no spikes are detected).
- **Cell level where Allen raises.** Allen aborts the whole cell when there
  are no spiking or no subthreshold long squares, no hero sweep, fewer than two
  spiking sweeps for the f-I fit or no sweeps of a stimulus type. jaxley_ephys
  returns NaN (indices -1) for the affected features only and computes the
  rest.
- **Degenerate line fits.** If all sweeps of the f-I or input-resistance fit
  have the same current, `np.linalg.lstsq` returns a minimum-norm solution;
  jaxley_ephys returns NaN.
- **Float precision.** Run directly on the float32 NWB data, Allen computes
  some means (`v_baseline`, `sag`) in float32, about 1e-5 mV away from the
  float64 result; the reference therefore uses float64 inputs (section 5).
  jaxley_ephys matches it exactly in float64 (`jax_enable_x64`). In float32 it
  takes the sampling interval of evenly sampled traces from the whole time axis
  (individual float32 intervals are several % off late in a sweep) and fits
  `tau` around the mean voltage. The remaining deviations come from the float32
  time resolution (about 1e-7 s at 1 s). On the 8 cached specimens spike counts
  are identical; of about 14,000 spike indices a few move by one sample and 2
  by more (a trough at the window end, which float32 rounds onto a sample, and
  a fast trough in a flat stretch of dV/dt). Features agree to about 2e-4, `tau`
  and `sag` (a ratio with a small numerator) to 1.5e-3 (`tests/test_float32.py`).
- **Filter numerics.** jaxley_ephys filters with second-order sections, Allen
  with `filtfilt(b, a)`; dV/dt agrees to about 1e-10 V/s, well within the
  1e-9 relative tolerance of the parity tests.

## Validation

Tests of this vendored code (`tests/allen_ephys/`: Allen's unit tests, integrity
checks on the cached specimens, comparison against the Allen DB) are skipped
unless `--allen` is given:
```bash
uv run python -m pytest --allen tests/allen_ephys -q
```

jaxley_ephys against this reference:
```bash
uv run python tests/build_reference.py
uv run python -m pytest -q   # add --num_specimens 8 for all cached specimens
```
