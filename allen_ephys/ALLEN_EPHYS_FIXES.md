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
  - `burst_metrics`: neutral return for no-spike sweeps
  - `delay_metrics`: warning-safe and exception-safe prespike fit path
  - long-square rheobase tie-break: when multiple spiking sweeps share the
    minimal stimulus amplitude, select the candidate with highest firing rate
    (`avg_rate`) for stable alignment with Allen DB rheobase sweep IDs. This
    matches the DB on 7 of the 8 cached specimens; on 605889373 the DB picks
    the first minimal-amplitude sweep (53) instead of the highest-rate one (57),
    and the upstream rule (first minimal-amplitude sweep) fails on 469801138
    and 537256313. ipfx's rule (an unstable sort by amplitude) does not
    reproduce the DB either, and excluding sweeps (e.g. by QC) would change the
    DB's f-I slope, so the DB's choice cannot be reproduced from the cached data
    (`tests/allen_ephys/test_parity.py` expects this failure; 2026-10-21)
  - `membrane_time_constant`: avoid `np.nanmean` all-NaN warning by averaging
    only finite `tau` values and returning `nan` when none are finite
- `allen_ephys/ephys_features.py`
  - `detect_bursts`: `inout_pairs = list(zip(...))` instead of `zip(...)`. The
    iterator was used up while scoring the burst candidates, so the function
    raised (`attempt to get argmin of an empty sequence`) on every spike train
    with candidates: on 27 long squares and 4 ramps of the cached specimens,
    where it now finds bursts on 22 sweeps. ipfx fixed it the same way
    (2026-10-21).
  - `detect_bursts`: `out_of_burst` is an integer array. With a single burst
    candidate running to the end of the spike train, its end was appended to an
    empty float array and slicing with it raised `TypeError` (found by
    `tests/test_patterns.py`; none of the cached specimens has such a train)
    (2026-10-21).
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
  - `delay_metrics`: the time constant s -> ms, like the other time constants
    (no DB counterpart; the delay ratio is unchanged) (2026-10-21)

## 4) DB parity mapping convention used in tests
- `tests/allen_ephys/test_parity.py` (runs with `--allen`)
  - long-square adaptation/latency/avg-ISI are mapped from the DB
    `thumbnail_sweep_id` when present; this is Allen's hero sweep (the
    smallest-amplitude spiking long square in (rheobase + 39, rheobase + 61)
    pA) on all cached specimens
  - long-square first-spike timing/voltage fields are mapped from the first
    spike of the DB's rheobase sweep (`rheobase_sweep_id`), as its choice among
    sweeps of equal amplitude cannot always be reproduced (section 2); the
    choice itself is compared separately and expected to fail on 605889373;
    `threshold_i_long_square` is mapped to `rheobase_i`
  - extractor `long_squares.input_resistance` is mapped to DB `ri`
    (Allen DB `input_resistance_mohm` is not a direct output of
    `EphysCellFeatureExtractor`)
  - short-square first-spike features are averaged over the common-amplitude
    sweeps of the fine protocol (`C1SSFINEST`): on 565871768 the DB leaves out
    a sweep of the coarse search (`C1SSCOARSE`) at the common amplitude. Allen's
    code (`extract_cell_features`, and ipfx) averages over all short squares
    with spikes, which matches none of the cached specimens (2026-10-21)
  - `threshold_i_short_square` is the common stimulus amplitude, as stored by
    Allen's current pipeline (ipfx, `feature_record.build_cell_feature_record`)
    and returned by jaxley_ephys (section 5). The DB rows of the two oldest
    cached specimens (469801138, 485909730; NWB files from 2016) hold the older
    value, the mean first-spike `threshold_i`, which is 0 or the amplitude per
    sweep, as short-square spikes start at the end of the 3 ms pulse
    (2026-10-21)
  - short-square `slow_trough_t/v` use first-spike `slow_trough_t/v` with
    fallback to `trough_t/v` when slow-trough values are not present
  - sweep-level `num_spikes` parity uses NWB `index_range` windows to align
    with Allen DB sweep metadata semantics
  - sweep-level `peak_deflect` parity compares value and index separately
    (`peak_deflect`, `peak_deflect_index`)
- These DB-matching heuristics are not part of Allen's code; they stay in the
  DB comparison tests and are not reproduced by jaxley_ephys, apart from the
  current definition of `threshold_i_short_square`.

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
  - burst and pause metrics of every sweep; for long squares also the delay
    metrics and the double-exponential fit behind them (parameters, SSE and
    window, recorded by wrapping `curve_fit`). On ramps and short squares the
    delay is not meaningful (section 6) and its fit is slow (seconds per ramp)
  - cell level: the `EphysCellFeatureExtractor` outputs, the hero sweep, the
    long-square baseline (`extract_cell_features` overrides the extractor's
    `nanmean` with a plain mean over all long squares) and first-spike means as
    in `extract_cell_features`, but for short squares over the common-amplitude
    sweeps (closer to the DB, section 4), the adjusted short-square detection
    parameters, and these values under their Allen DB names (`record`, without
    the DB-matching heuristics of section 4). Like ipfx, the record's
    `threshold_i_short_square` is the common short-square amplitude, not the
    mean first-spike `threshold_i`, which is 0 or that amplitude per sweep
    (2026-10-22)
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
  (relative; median 4e-7) away from a fully converged fit; Gauss-Newton-type
  steps converge only linearly when the exponential fits the response poorly.
  jaxley_ephys fits to the optimum (Newton steps with secant curvature, 3-10
  iterations on the cached sweeps), so sweep-level `tau` and the cell-level
  mean `tau` are compared with `rtol = 2e-3` (`tests/utils/parity.py`).
- **tau on ill-posed fits.** Without a clean exponential relaxation (e.g. the
  synthetic fuzz traces) the fit is ill-posed: Allen's local search may hit
  `maxfev` and return NaN, or end in a different local minimum than
  jaxley_ephys. The values (sometimes several seconds) are not meaningful
  either way, so `tests/test_fuzz.py` does not compare `tau`. jaxley_ephys
  stops such fits once the exponential is indistinguishable from a straight
  line or a step, or its trust region collapses.
- **Delay metrics (`delay_metrics`, `sweep_features(..., delays=True)`): Allen's
  optimizer stays at its start.** `fit_prespike_time_constant` fits `y0 + A1 exp(-t/tau1) + A2 exp(-t/tau2)`
  (`y0` just above the window's maximum) with `curve_fit`, started with both
  components at the same time constant (from a straight-line fit of
  `log(y0 - v)`). There the components are interchangeable, and
  Levenberg-Marquardt mostly keeps them equal, i.e. returns a single
  exponential. Of the 85 spiking long squares of the cached specimens, Allen
  separates two time constants on 25 and stays at a single exponential on 55;
  on 29 of those a separate double exponential fits better. Examples: specimen
  605889373, sweeps 53 and 55 give 885 and 808 ms (delay ratio about 1), but
  6.6 and 6.8 ms (ratios 130 and 116) with the double exponential, like
  Allen's own results on sweeps 56-58 (ratios 77-88). On 4 sweeps `curve_fit`
  fails and Allen falls back to the straight-line fit; on 1 (605889373,
  sweep 50) its slower time constant runs off to 27,000 s, a constant offset.
  jaxley_ephys fits a single exponential and double exponentials from two
  starts (one component 100x faster than the single exponential, the other at
  its time constant or 100x slower). It uses the best double exponential with
  distinct time constants below 100x the window that fits better than the
  single exponential, else the single exponential, and picks the dominant time
  constant with Allen's rules. `tests/test_patterns.py` checks:
  - where Allen separated the components, the delay agrees within 2.1e-5
    (relative);
  - where Allen stayed at its start, the single exponentials agree within
    8.1e-5, and the double exponential used instead always fits better (on 18
    of the 29 sweeps within 1% of Allen's value).
- **Delay metrics on ramps and short squares** are not compared: the rise
  before the first spike is not exponential (the short squares' window starts
  at the beginning of the recording), and Allen's fit returns time constants
  from 1e-6 to 1e5 s, or negative ones from its fallback.
- **Delay metrics without a fit.** Without spikes, or with fewer than 4 samples
  before the first spike, Allen returns `(0, 0)`; jaxley_ephys returns NaN.
- **Samples after the analysis window.** `detect_spikes` analyses a sweep up to
  `t_cutoff`, by default 20 ms after `end` (the reach of the peak search and of
  the filter transients, which decay below 1e-20 within it); all features then
  agree with an analysis of the whole sweep. An earlier `t_cutoff` (the user's
  choice) can change spikes near `end`. The short-square `stim_amp`, the largest
  current after `start` (`_short_step_stim_amp`), only covers samples up to
  `t_cutoff`; it differs from Allen if the current is larger afterwards.
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
