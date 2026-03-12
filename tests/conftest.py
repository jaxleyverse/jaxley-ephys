from __future__ import annotations

import pytest

from tests.utils.allen_dataset import cached_specimen_ids, load_specimen_case


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
        scope="module",
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
def specimen_cases(specimen_ids: list[int]):
    return {
        specimen_id: load_specimen_case(specimen_id) for specimen_id in specimen_ids
    }


@pytest.fixture(scope="module")
def specimen_case(specimen_id: int):
    if specimen_id == -1:
        pytest.fail(
            "No cached specimens found in tests/testdata/allen_cache. Run: "
            "uv run python tests/load_testdata.py"
        )
    return load_specimen_case(specimen_id)
