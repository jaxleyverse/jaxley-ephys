from __future__ import annotations

from dataclasses import dataclass

import jax
import pytest

from tests.utils.allen_dataset import cached_specimen_ids, load_specimen_case

# Allen parity checks compare sample indices exactly; float64 avoids artificial
# dt quantization drift from float32 that can shift threshold crossings by 1+ samples.
jax.config.update("jax_enable_x64", True)


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
def specimen_context_loader():
    cache: dict[int, SpecimenCellContext] = {}

    def _build(specimen_id: int) -> SpecimenCellContext:
        if specimen_id in cache:
            return cache[specimen_id]

        specimen_case = load_specimen_case(specimen_id)
        cell = specimen_case.build_cell_extractor_for_nwb()
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

        # Keep parity inputs float64-stable for JAX filtering/index comparisons.
        seen = set()
        for ext in sweep_extractors.values():
            for sweep in ext.sweeps():
                sweep_key = id(sweep)
                if sweep_key in seen:
                    continue
                seen.add(sweep_key)
                sweep.t = sweep.t.astype(float)
                sweep.v = sweep.v.astype(float)
                if sweep.i is not None:
                    sweep.i = sweep.i.astype(float)

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
def specimen_cell_context(specimen_id: int, specimen_context_loader) -> SpecimenCellContext:
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
