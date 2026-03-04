# Vendored Allen Ephys Fixes

This file lists the **minimal functional changes** needed to make the vendored
Allen ephys tests pass on current dependencies.

## Required fixes

1) Use vendored code in tests (not installed `allensdk`)
- `tests/test_allen_features.py`: import `tests.allen_ephys.ephys_features`
- `tests/test_allen_extractor.py`: import `tests.allen_ephys.ephys_extractor`
- `tests/test_allen_extractor.py`: patch target updated to
  `tests.allen_ephys.ephys_extractor._step_stim_amp`
- `tests/__init__.py`: added so `tests.allen_ephys` is importable

2) pandas compatibility in extractor
- `tests/allen_ephys/ephys_extractor.py`
  - `spikes_df.ix[...]` -> `spikes_df.loc[...]`
  - `six.iteritems(...)` -> `.items()`

Reason: pandas 3 removed `.ix`; extractor fails with `AttributeError` otherwise.

3) avoid invalid divide warning on variable-dt fixture
- `tests/allen_ephys/ephys_features.py` in `calculate_dvdt`
  - before: `dvdt = 1e-3 * dv / dt`
  - now: `valid_dt = dt != 0`; `dvdt = 1e-3 * dv[valid_dt] / dt[valid_dt]`

Reason: `tests/data/spike_test_var_dt.txt` has one repeated timestamp (`dt == 0`),
which triggers `RuntimeWarning: invalid value encountered in divide`.

4) make spike-train metrics robust on edge-case sweeps
- `tests/allen_ephys/ephys_extractor.py`
  - `pause_metrics`: return `(0, 0.)` when no spikes are present
  - `burst_metrics`: return `(0., 0)` when no spikes are present and guard
    against edge-case exceptions in burst detection
  - `delay_metrics`: guard prespike fit with warning-safe/exception-safe path

Reason: on real cached sweeps, these methods were raising errors/warnings for
valid "no-spike" or unstable-fit cases. They now fail soft with neutral values.

## Validation

```bash
uv run python -m pytest tests/test_allen_features.py tests/test_allen_extractor.py -q
uv run python -m pytest tests/test_vendored_computability.py -q
```

Current result:
- `tests/test_allen_features.py tests/test_allen_extractor.py`: `31 passed, 1 skipped`
- `tests/test_vendored_computability.py`: `25 passed`
