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

## 3) Unit alignment with Allen DB
- `allen_ephys/ephys_extractor.py`
  - `mean_isi`: s -> ms (`1e3 * isis.mean()`)
  - `tau`: s -> ms (`1e3 / inv_tau`)
  - `membrane_time_constant` docs updated to ms

## 4) DB parity mapping convention used in tests
- `tests/conftest.py`
  - long-square spike fields are mapped from the first spike of
    `rheobase_sweep` (fallback to hero-sweep when unavailable)
  - `threshold_i_long_square` is mapped to `rheobase_i`
  - `threshold_i_short_square` uses first-spike `threshold_i` values from
    canonical short-square common-amplitude sweeps, excluding zero-current
    threshold events; if all are zero, it falls back to short-square stimulus
    amplitude

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
uv run python -m pytest tests/allen_ephys/test_integrity.py tests/allen_ephys/test_match_data.py -q
```

## Observed DB mismatches and hypotheses

Current persistent mismatch clusters (from `tests/allen_ephys/report_match_data.py`):
- `input_resistance_mohm` 
- `threshold_i_short_square`
- small long-square timing offsets (`threshold_t_long_square`, `peak_t_long_square`,
  `fast_trough_t_long_square`, `trough_t_long_square`)

Working hypotheses:
- **Rheobase tie-break differences**: DB and vendored extractor likely choose
  different sweep IDs when multiple sweeps share rheobase amplitude.
- **Short-square inclusion differences**: DB pipeline likely uses additional
  sweep inclusion/exclusion logic (or protocol grouping) beyond the simple
  canonical subset used here.
- **Subthreshold fit selection differences**: `input_resistance_mohm` appears
  sensitive to which long-square subthreshold sweeps are included for fitting.
