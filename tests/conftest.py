from __future__ import annotations

import pytest

from tests.utils.allen_dataset import cached_specimen_ids, load_specimen_case


CACHED_SPECIMEN_IDS = cached_specimen_ids()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--specimen-scope",
        action="store",
        default="fast",
        choices=("fast", "slow", "all"),
        help="Specimen scope for Allen tests: fast=1, slow=5, all=all cached.",
    )


def _scoped_specimen_ids(scope: str) -> list[int]:
    if scope == "fast":
        return CACHED_SPECIMEN_IDS[:1]
    if scope == "slow":
        return CACHED_SPECIMEN_IDS[:5]
    return CACHED_SPECIMEN_IDS


@pytest.fixture(scope="session")
def specimen_ids(request: pytest.FixtureRequest) -> list[int]:
    if not CACHED_SPECIMEN_IDS:
        pytest.fail(
            "No cached specimens found in tests/data/allen_cache. Run: "
            "uv run python tests/load_testdata.py"
        )

    scope = str(request.config.getoption("--specimen-scope"))
    selected = _scoped_specimen_ids(scope)
    if not selected:
        pytest.fail(f"No specimens selected for scope={scope}")
    return selected


@pytest.fixture(scope="session")
def specimen_cases(specimen_ids: list[int]):
    return {
        specimen_id: load_specimen_case(specimen_id) for specimen_id in specimen_ids
    }


@pytest.fixture(params=CACHED_SPECIMEN_IDS, scope="module")
def specimen_case(
    request: pytest.FixtureRequest,
    specimen_ids: list[int],
):
    specimen_id = int(request.param)
    if specimen_id not in specimen_ids:
        pytest.skip(f"specimen {specimen_id} skipped by --specimen-scope")
    return load_specimen_case(specimen_id)
