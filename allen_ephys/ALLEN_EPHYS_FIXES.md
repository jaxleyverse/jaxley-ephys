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
    (`avg_rate`) for stable alignment with Allen DB rheobase sweep IDs
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

## 3) Unit alignment with Allen DB
- `allen_ephys/ephys_extractor.py`
  - `mean_isi`: s -> ms (`1e3 * isis.mean()`)
  - `tau`: s -> ms (`1e3 / inv_tau`)
  - `membrane_time_constant` docs updated to ms

## 4) DB parity mapping convention used in tests
- `tests/allen_ephys/test_parity.py`
  - long-square adaptation/latency/avg-ISI are mapped from the DB
    `thumbnail_sweep_id` when present
  - long-square first-spike timing/voltage fields are mapped from the first
    spike of `rheobase_sweep`
  - extractor `long_squares.input_resistance` is mapped to DB `ri`
    (Allen DB `input_resistance_mohm` is not a direct output of
    `EphysCellFeatureExtractor`)
  - `threshold_i_short_square` uses a DB-aligned aggregation over
    common-amplitude short-square sweeps:
    - for small common-amplitude sets (<=2 sweeps), use mean first-spike
      `threshold_i`
    - otherwise use mean nonzero first-spike `threshold_i` (fallback to
      common stimulus amplitude when all are zero)
  - short-square `slow_trough_t/v` use first-spike `slow_trough_t/v` with
    fallback to `trough_t/v` when slow-trough values are not present
  - `threshold_i_long_square` is mapped to `rheobase_i`
  - `threshold_i_short_square` uses first-spike `threshold_i` values from
    canonical short-square common-amplitude sweeps; if none are available it
    falls back to short-square stimulus amplitude
  - sweep-level `num_spikes` parity uses NWB `index_range` windows to align
    with Allen DB sweep metadata semantics

## 5) Canonical stimulus selection in preprocessing
- `tests/load_testdata.py`
  - cache only canonical stimulus names: `Long Square`, `Ramp`, `Short Square`
  - exclude non-canonical variants (e.g. `Ramp to Rheobase`,
    `Short Square - Triple`) for stable parity mapping across specimens

## Validation

Allen-native vendored tests:
```bash
uv run python -m pytest tests/allen_ephys/test_allen_features.py tests/allen_ephys/test_allen_extractor.py -q
```

New repository tests:
```bash
uv run python -m pytest tests/allen_ephys/test_integrity.py tests/allen_ephys/test_parity.py -q
```
