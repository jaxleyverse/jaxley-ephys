# AGENTS.md

## Purpose
This guide is for autonomous coding agents operating in this repository.
It defines commands, style expectations, and workflow constraints so changes stay
correct, readable, and aligned with project goals.

## Scope And Priorities
- Project: `jaxley-features` (library for electrophysiological feature functions)
- Runtime library should **not** depend on AllenSDK
- AllenSDK is allowed in tests/dev tooling only
- Focus on maintainable, minimal, explicit implementations
- Prefer correctness + clarity over clever compact code

## Critical Workflow Rule
- Work on **one subgoal at a time**
- Do not begin the next subgoal until the user explicitly approves proceeding
- If a task branches, finish or pause current branch and request direction

## Plan Tracking (Required)
- The canonical project plan is in `IMPLEMENTATION_PLAN.md`
- Keep plan checkboxes updated whenever a subgoal/task status changes
- At the end of each completed subgoal:
  1. Mark relevant tasks `[x]` in `IMPLEMENTATION_PLAN.md`
  2. Add a brief dated progress note in that file
  3. Stop and wait for explicit user approval before starting the next subgoal
- Do not execute tasks from a later subgoal even if they seem small or related

## Current Repository Reality
- Package metadata: `pyproject.toml`
- Source package currently minimal: `jaxley-features/__init__.py`
- Allen reference code vendored under `tests/allen_ephys/`
- Allen tests under `tests/allen_ephys_tests/`
- No Cursor/Copilot instruction files detected:
  - no `.cursorrules`
  - no `.cursor/rules/`
  - no `.github/copilot-instructions.md`

## Environment And Setup
Run from repository root.

1. Sync environment
   - `uv sync`
2. Sync dev dependencies
   - `uv sync --group dev`
3. Ensure pytest is available (not currently in `pyproject.toml`)
   - `uv pip install pytest`

Quick checks:
- `uv run python -V`
- `uv run python -c "import jax, numpy, scipy, allensdk; print('deps-ok')"`

## Build / Install Commands
This project currently has no explicit build backend workflow configured.

- Editable install
  - `uv pip install -e .`
- Import smoke test
  - `uv run python -c "import importlib; importlib.import_module('jaxley_features')"`

If a build backend is added later:
- `uv run python -m build`

## Test Commands
Prefer `python -m pytest` invocations.

- Run all tests
  - `uv run python -m pytest`
- Run Allen vendored tests only
  - `uv run python -m pytest tests/allen_ephys_tests`
- Run single test file
  - `uv run python -m pytest tests/allen_ephys_tests/test_features.py`
- Run single test function
  - `uv run python -m pytest tests/allen_ephys_tests/test_features.py::test_detect_two_spikes`
- Run by keyword
  - `uv run python -m pytest -k "threshold or trough"`
- Stop on first failure
  - `uv run python -m pytest -x`
- Verbose mode
  - `uv run python -m pytest -vv`
- Treat warnings as errors
  - `uv run python -m pytest -W error`

Single-test examples expected during development:
- `uv run python -m pytest tests/allen_ephys_tests/test_extractor.py::test_extractor_on_variable_time_step -vv`
- `uv run python -m pytest tests/allen_ephys_tests/test_features.py::test_width_calculation -vv`

## Planned Testcase Cache Workflow
Planned cache location:
- `tests/data/`
- Re-downloadable AllenSDK raw cache should live in `tests/data/allen_cache/`

Planned loader script:
- `tests/utils/load_allen_testcases.py`

Expected usage:
1. Generate cache first:
   - `uv run python tests/utils/load_allen_testcases.py`
2. Run cache-dependent tests:
   - `uv run python -m pytest tests/reference tests/parity`

Tests should fail early with a clear action message if required cache is missing.

## Lint / Format / Types
No lint/format/type tool is currently configured in `pyproject.toml`.
Until tooling is added, follow these conventions:

- PEP 8 style baseline
- 4-space indentation
- Keep line length around 88-100 chars
- Keep functions cohesive and short where practical
- No dense nested comprehensions that hurt readability
- Avoid unnecessary abstractions or class-heavy patterns

When tools are later added, recommended defaults:
- Lint: `ruff`
- Format: `black`
- Types: `mypy` or `pyright`

## Imports
- Order: stdlib, third-party, local
- Separate import groups with one blank line
- Avoid wildcard imports
- Avoid importing heavy optional deps in library import path
- Keep AllenSDK imports out of runtime package modules

## Naming Conventions
- Modules/files: `snake_case.py`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Classes: `PascalCase` (use only when helpful)
- Tests: `test_<behavior>.py`, `test_<scenario>()`

## Typing And API Shape Conventions
- Add type hints for public functions and non-trivial helpers
- Document array shape conventions in docstrings:
  - sweep level: `v[T]`, `t[T]`, optional `i[T]`
  - cell level: `v[S, T]`, `t[S, T]`, optional `i[S, T]`
  - batched cells: `v[C, S, T]`
- Keep feature APIs functional and explicit
- Include optional `precomputed` dict where feature reuse is useful

## Error Handling
- Validate input shape and key assumptions at boundaries
- Raise specific exceptions (`ValueError`, `TypeError`, domain-specific errors)
- Use clear, actionable error messages
- Do not silently swallow exceptions
- Use warnings sparingly and intentionally

## JAX Implementation Guidance
- Prefer `jax.numpy` operations in library code
- Keep functions pure (no hidden mutable global state)
- Write with `vmap` compatibility in mind
- Favor implementations that are JAX-friendly and efficient
- Match Allen reference behavior via tests, but do not copy unnecessary complexity

## Allen Reference Strategy
- Allen reference lives in tests only
- Use Allen outputs as parity target unless a verified bug is documented
- For mismatches:
  1. Reproduce on minimal case
  2. Check units/window/parameter assumptions
  3. Add regression test and note rationale

## Testing Standards
- Use deterministic inputs where possible
- Assert with explicit tolerances for floats (`rtol`, `atol`)
- Cover edge cases:
  - no spikes
  - clipped spikes
  - variable dt
  - threshold/peak boundary conditions
- Add a test for every bug fix

## Git And Change Hygiene
- Keep changes focused to the active subgoal only
- Do not revert unrelated user changes
- Do not add unrelated refactors during task execution
- Keep commit scopes tight when commits are requested
- Use Conventional Commits style for commit messages (e.g., `feat: ...`, `fix: ...`, `test: ...`, `docs: ...`)

## Documentation Expectations
- Public functions should include concise docstrings:
  - what is computed
  - expected units
  - input/output shape
  - important assumptions
- Keep README/usage docs aligned with real commands and paths

## Agent Readiness Checklist
Before implementing code for a subgoal:
1. Confirm exact subgoal and acceptance criteria with user
2. Identify relevant files and tests
3. Run the smallest useful test slice first
4. Implement minimal solution for this subgoal only
5. Re-run targeted tests
6. Report results and wait for explicit approval before next subgoal
