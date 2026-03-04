# Implementation Plan

This file is the canonical execution plan for `jaxley-features`.

## Working Rules
- Work on exactly one subgoal at a time.
- Do not start the next subgoal until the user explicitly approves.
- Keep this file updated during execution (checkboxes and short progress notes).
- Keep AllenSDK use confined to `tests/` and dev tooling.

## Status Legend
- `[ ]` not started
- `[-]` in progress
- `[x]` completed
- `[!]` blocked (waiting on user decision or external issue)

## Subgoal 1 - Reproducible Allen testcase loader and cache
Goal: Create reproducible cached testcases from Allen with mixed mouse/human cells.

- [x] Create `tests/utils/load_allen_testcases.py`
- [x] Hardcode a mixed panel of Allen cell IDs at top of loader script
- [x] Download and store per-cell sweep data as:
  - `tests/data/cell_{cell_id}_data.npz`
  - `tests/data/cell_{cell_id}_meta.json`
- [x] Include long-square, ramp, and short-square sweep metadata required for testing
- [x] Include website/database-provided feature values in meta JSON (for later comparison)
- [x] Add clear CLI usage/help text to loader script
- [x] Keep processed testcase artifacts under `tests/data/` and raw AllenSDK cache under `tests/data/allen_cache/`
- [x] Smoke-run loader for at least one cell and verify files are written

Exit criteria:
- Loader runs successfully and creates the expected cache files.
- Cache format is documented in code comments/docstring.

## Subgoal 2 - Get Allen reference pipeline running cleanly
Goal: Ensure vendored Allen feature extraction executes with current environment.

- [x] Ensure environment setup commands work (`uv sync`, dev deps, `pytest`)
- [x] Run Allen vendored tests:
  - `uv run python -m pytest tests/allen_ephys_tests`
- [x] Run same test suite with warnings elevated:
  - `uv run python -m pytest -W error tests/allen_ephys_tests`
- [x] Capture failures/warnings due to NumPy/SciPy/API changes
- [x] Apply minimal compatibility fixes to tests-side Allen code if required
- [x] Re-run targeted failing tests until stable

Exit criteria:
- Allen vendored tests pass in current environment without unexpected warnings/errors.

## Subgoal 3 - Validate Allen extraction against Allen website/database features
Goal: Verify extracted Allen features match stored/served Allen features where applicable.

- [ ] Implement comparison tests in `tests/reference/` for sweep-level features first
- [ ] Prioritize sweep comparisons in order:
  1. long square
  2. ramp
  3. short square
- [ ] Add cell-level comparison tests after sweep-level is stable
- [ ] Define per-feature tolerance policy (`atol`/`rtol`) in one place
- [ ] Produce mismatch reports with cell ID, sweep ID, feature name, values, and delta
- [ ] Document confirmed Allen mismatches/bugs with minimal reproductions

Exit criteria:
- We have a clear list of matching features and any justified mismatches.
- Comparison tests are deterministic and readable.

## Subgoal 4 - Harden Allen baseline and ensure all required features are computable
Goal: Make sure the reference baseline is robust enough to serve as oracle for parity.

- [ ] Enumerate required sweep-level and cell-level features for project scope
- [ ] Add tests for any missing Allen feature computation paths
- [ ] Ensure all required features can be computed on cached testcases
- [ ] If missing/unstable, add minimal test-only wrappers in `tests/utils/allen_reference.py`
- [ ] Freeze Allen reference parameter choices and windows used by parity tests

Exit criteria:
- Every feature required for JAX reimplementation has a stable Allen reference path.

## Subgoal 5 - Implement JAX feature primitives and sweep-level features
Goal: Build clean, functional JAX-first sweep feature functions with parity checks.

- [ ] Create runtime package layout under `jaxley_features/` (underscore naming)
- [ ] Add `jaxley_features/features/primitives/` modules (dV/dt, spike detection, thresholds, troughs, ISIs)
- [ ] Define explicit function signatures:
  - sweep level: `v[T]`, `t[T]`, optional `i[T]`, optional `precomputed`
- [ ] Implement long-square sweep features first
- [ ] Implement ramp sweep features next
- [ ] Implement short-square sweep features last
- [ ] Keep code pure and vmappable-friendly from the start
- [ ] Add parity tests in `tests/parity/` against Allen reference for each sweep feature

Exit criteria:
- Sweep-level JAX features for all three stimulus types pass parity tests.

## Subgoal 6 - Implement JAX cell-level features and parity tests
Goal: Compute cell-level features from sweep-level building blocks.

- [ ] Implement cell-level APIs using sweep-level functions on `[S, T]`
- [ ] Ensure APIs are vmappable across cells `[C, S, T]`
- [ ] Reuse optional `precomputed` where it simplifies dependency reuse
- [ ] Implement parity tests for rheobase and other targeted cell features
- [ ] Validate parity on cached mixed-cell panel

Exit criteria:
- Cell-level JAX features pass parity tests and work on batched inputs.

## Subgoal 7 - Optimize codebase and computational efficiency
Goal: Improve performance/clarity while preserving behavior.

- [ ] Profile hotspots in sweep and cell feature computation
- [ ] Remove redundant computations via reusable intermediate results
- [ ] Keep APIs simple and avoid overengineering
- [ ] Refactor only where it improves maintainability/readability
- [ ] Re-run parity and regression tests after each optimization batch

Exit criteria:
- Performance improves or code is simpler without changing validated outputs.

## Subgoal 8 - Make features jittable
Goal: Ensure core feature paths can be compiled with `jax.jit`.

- [ ] Identify non-jittable control flow and data structures
- [ ] Replace Python-heavy branches with JAX-friendly patterns where needed
- [ ] Add tests that execute selected features under `jax.jit`
- [ ] Confirm output parity between eager and jitted modes

Exit criteria:
- Selected core sweep and cell features run under `jax.jit` and match eager outputs.

## Subgoal 9 - Make features differentiable (or provide differentiable variants)
Goal: Enable gradient-based workflows for feasible features.

- [ ] Audit features for non-differentiable operations (argmax, thresholding, indexing)
- [ ] For each such feature, decide:
  - exact non-diff implementation, or
  - differentiable surrogate variant
- [ ] Add clear naming/docstrings for differentiable variants
- [ ] Add tests for gradient existence/stability where expected
- [ ] Document caveats for features that remain non-differentiable by design

Exit criteria:
- Differentiability status is explicit per feature, with tested behavior.

## Subgoal 10 - Cleanup and release readiness
Goal: Final cleanup, docs alignment, and reliable test workflow.

- [ ] Remove dead code and stale experimental helpers
- [ ] Ensure docs match real commands, paths, and package names
- [ ] Ensure cache precondition is clearly communicated in pytest failures
- [ ] Run complete test suite and report final status
- [ ] Prepare concise implementation summary and known limitations

Exit criteria:
- Repository is consistent, documented, and ready for continued development.

## Progress Notes
- 2026-03-03: Plan initialized and linked from `AGENTS.md`. No implementation subgoal started yet.
- 2026-03-03: Subgoal 1 completed. Added `tests/utils/load_allen_testcases.py`, added cache ignore rule, and smoke-tested cache generation for specimen `525011903`.
- 2026-03-04: Subgoal 1 hardening: loader now supports `--manifest-file` and defaults to `<output-dir>/cell_types_manifest.json` so AllenSDK cache stays under `tests/data`. Added CI workflow at `.github/workflows/ci.yml` with cache generation smoke test and stable Allen feature tests.
- 2026-03-04: Subgoal 1 hardening: loader now skips existing per-specimen cache files by default and fetches only missing specimens; added `--force` to regenerate. Simplified CI to run loader without forcing redownloads.
- 2026-03-03: Subgoal 2 completed. Installed `pytest`, routed Allen tests to vendored modules via `tests/allen_ephys_tests/conftest.py`, and fixed NumPy/SciPy-era warning behavior in `tests/allen_ephys/ephys_features.py` so vendored tests pass cleanly (including `-W error`).
