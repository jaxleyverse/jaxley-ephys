from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import pytest

from tests.utils.allen_dataset import cached_specimen_ids, load_specimen_case
from tests.utils.feature_cache import (
    load_cached_allen_features,
    write_cached_allen_features,
)

# Allen parity checks compare sample indices exactly; float64 avoids artificial
# dt quantization drift from float32 that can shift threshold crossings by 1+ samples.
jax.config.update("jax_enable_x64", True)


def _modality_for_extractor_key(extractor_key: str) -> str | None:
    if extractor_key.startswith("long_square"):
        return "long_square"
    if extractor_key == "short_square":
        return "short_square"
    if extractor_key.startswith("ramp"):
        return "ramp"
    return None


def _prepare_sweep_arrays_for_parity(sweep_extractors: dict[str, Any]) -> None:
    max_len_by_modality: dict[str, int] = {}
    modality_by_sweep: dict[int, str] = {}
    sweep_by_key: dict[int, Any] = {}

    for extractor_key, ext in sweep_extractors.items():
        modality = _modality_for_extractor_key(extractor_key)
        if modality is None:
            continue
        for sweep in ext.sweeps():
            sweep_key = id(sweep)
            sweep_by_key[sweep_key] = sweep
            prior = modality_by_sweep.get(sweep_key)
            if prior is not None and prior != modality:
                raise ValueError(
                    f"sweep object reused across modalities: {prior} vs {modality}"
                )
            modality_by_sweep[sweep_key] = modality
            max_len_by_modality[modality] = max(
                max_len_by_modality.get(modality, 0),
                int(len(sweep.t)),
            )

    for sweep_key, sweep in sweep_by_key.items():
        modality = modality_by_sweep[sweep_key]
        t = np.asarray(sweep.t, dtype=float)
        v = np.asarray(sweep.v, dtype=float)
        i = None if sweep.i is None else np.asarray(sweep.i, dtype=float)

        target_len = max_len_by_modality.get(modality, int(len(t)))
        if target_len > len(t):
            pad = target_len - len(t)
            v = np.pad(v, (0, pad), mode="edge")
            if i is not None:
                i = np.pad(i, (0, pad), mode="edge")
            dt = float(t[1] - t[0]) if t.size > 1 else 0.0
            t_tail = t[-1] + dt * np.arange(1, pad + 1, dtype=float)
            t = np.concatenate([t, t_tail])

        sweep.t = t
        sweep.v = v
        if i is not None:
            sweep.i = i


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--num_specimens",
        "--num-specimens",
        dest="num_specimens",
        action="store",
        type=int,
        default=1,
        help="Number of cached specimens to include (default: 1).",
    )
    parser.addoption(
        "--use-cached-features",
        action="store_true",
        default=False,
        help=(
            "Load precomputed Allen extractor features from "
            "tests/testdata/allen_cache/specimen_<id>/ephys_features.{json,npz}."
        ),
    )


def _limited_specimen_ids(num_specimens: int, all_ids: list[int]) -> list[int]:
    if num_specimens <= 0:
        raise pytest.UsageError("--num-specimens must be >= 1")
    return all_ids[: min(num_specimens, len(all_ids))]


def _selected_specimen_ids(config: pytest.Config) -> list[int]:
    all_ids = cached_specimen_ids()
    num_specimens = int(config.getoption("num_specimens"))
    return _limited_specimen_ids(num_specimens, all_ids)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "specimen_id" not in metafunc.fixturenames:
        return

    selected = _selected_specimen_ids(metafunc.config)
    if not selected:
        selected = [-1]

    metafunc.parametrize(
        "specimen_id",
        selected,
        ids=[f"specimen_{specimen_id}" for specimen_id in selected],
        scope="session",
    )


@pytest.fixture(scope="session")
def specimen_ids(request: pytest.FixtureRequest) -> list[int]:
    selected = _selected_specimen_ids(request.config)
    if not selected:
        pytest.fail(
            "No cached specimens found in tests/testdata/allen_cache. Run: "
            "uv run python tests/load_testdata.py"
        )
    return selected


@pytest.fixture(scope="session")
def specimen_cases(specimen_ids: list[int], specimen_context_loader):
    # Lazy compatibility shim; use specimen_cell_context where possible.
    valid_ids = set(specimen_ids)

    class _SpecimenCaseMap(dict):
        def __missing__(self, key):
            if key not in valid_ids:
                raise KeyError(key)
            value = specimen_context_loader(key).specimen_case
            self[key] = value
            return value

    return _SpecimenCaseMap()


@dataclass
class SpecimenCellContext:
    specimen_id: int
    specimen_case: object
    cell_extractor: object
    cell_features: dict
    sweep_extractors: dict[str, object]


@pytest.fixture(scope="session")
def specimen_context_loader(request: pytest.FixtureRequest):
    cache: dict[int, SpecimenCellContext] = {}
    use_cached_features = bool(request.config.getoption("use_cached_features"))

    def _build(specimen_id: int) -> SpecimenCellContext:
        if specimen_id in cache:
            return cache[specimen_id]

        specimen_case = load_specimen_case(specimen_id)
        cell = specimen_case.build_cell_extractor_for_nwb()

        if use_cached_features:
            try:
                features = load_cached_allen_features(specimen_case, cell)
            except FileNotFoundError as exc:
                pytest.fail(
                    f"--use-cached-features set but cached features are missing for "
                    f"specimen {specimen_id}: {exc}. Run pytest once without "
                    f"--use-cached-features to generate ephys_features cache files."
                )
        else:
            cell.process()
            features = cell.as_dict()

        sweep_extractors = {
            # Allen process() mutates internal state and filters short-square and
            # ramp sets (common-amplitude and spiking-only respectively).
            # We intentionally use these processed extractors as parity targets.
            "long_square": cell.long_squares_features(),
            "long_square_spiking": cell.long_squares_features("spiking"),
            "long_square_subthreshold": cell.long_squares_features("subthreshold"),
            "long_square_subthreshold_membrane_property": cell.long_squares_features(
                "subthreshold_membrane_property"
            ),
            "short_square": cell.short_squares_features(),
            "ramp": cell.ramps_features(all=False),
            "ramp_all": cell.ramps_features(all=True),
        }

        if not use_cached_features:
            write_cached_allen_features(specimen_case, features, sweep_extractors)

        # Keep parity inputs float64-stable and shape-aligned by modality.
        _prepare_sweep_arrays_for_parity(sweep_extractors)

        ctx = SpecimenCellContext(
            specimen_id=specimen_id,
            specimen_case=specimen_case,
            cell_extractor=cell,
            cell_features=features,
            sweep_extractors=sweep_extractors,
        )
        cache[specimen_id] = ctx
        return ctx

    return _build


@pytest.fixture(scope="session")
def specimen_cell_context(
    specimen_id: int, specimen_context_loader
) -> SpecimenCellContext:
    if specimen_id == -1:
        pytest.fail(
            "No cached specimens found in tests/testdata/allen_cache. Run: "
            "uv run python tests/load_testdata.py"
        )
    return specimen_context_loader(specimen_id)


@pytest.fixture(scope="session")
def specimen_sweepset_extractors(specimen_cell_context: SpecimenCellContext):
    return {
        "long_square": specimen_cell_context.sweep_extractors["long_square"],
        "short_square": specimen_cell_context.sweep_extractors["short_square"],
        "ramp": specimen_cell_context.sweep_extractors["ramp"],
    }


@pytest.fixture(scope="session")
def specimen_case(specimen_cell_context: SpecimenCellContext):
    return specimen_cell_context.specimen_case
