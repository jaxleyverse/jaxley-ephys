from __future__ import annotations

import jax
import pytest

from tests.utils.allen_dataset import cached_specimen_ids, load_specimen_case
from tests.utils.allen_reference import KINDS, allen_sweeps, load_reference
from tests.utils.jax_runner import analyse_sweep

# Allen parity compares sample indices exactly; float64 avoids float32 rounding
# of long time axes shifting threshold crossings by one or more samples.
jax.config.update("jax_enable_x64", True)

MISSING_CACHE = (
    "No cached specimens in tests/testdata/allen_cache. Run: "
    "uv run python tests/load_testdata.py && uv run python tests/build_reference.py"
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--num_specimens",
        "--num-specimens",
        dest="num_specimens",
        type=int,
        default=1,
        help="Number of cached specimens to test against (default: 1).",
    )
    parser.addoption(
        "--allen",
        action="store_true",
        default=False,
        help="Also run the tests of the vendored Allen code in tests/allen_ephys/ "
        "(unit tests, integrity checks, comparison against the Allen DB).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    if config.getoption("allen"):
        return
    skip = pytest.mark.skip(reason="tests of the vendored Allen code; run with --allen")
    for item in items:
        if "allen_ephys" in item.path.parts:
            item.add_marker(skip)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "specimen_id" not in metafunc.fixturenames:
        return
    num_specimens = metafunc.config.getoption("num_specimens")
    if num_specimens < 1:
        raise pytest.UsageError("--num-specimens must be >= 1")
    selected = cached_specimen_ids()[:num_specimens] or [None]
    metafunc.parametrize(
        "specimen_id",
        selected,
        ids=[f"specimen_{specimen_id}" for specimen_id in selected],
        scope="session",
    )


@pytest.fixture(scope="session")
def specimen_case(specimen_id):
    if specimen_id is None:
        pytest.fail(MISSING_CACHE)
    return load_specimen_case(specimen_id)


@pytest.fixture(scope="session")
def reference(specimen_id) -> dict:
    """Vendored-Allen outputs for all sweeps and the cell (see tests/utils/allen_reference.py)."""
    if specimen_id is None:
        pytest.fail(MISSING_CACHE)
    return load_reference(specimen_id)


@pytest.fixture(scope="session")
def results(specimen_id, reference) -> dict[str, list[dict]]:
    """JAX outputs for every sweep, on the same traces and parameters Allen used."""
    _, sweeps = allen_sweeps(specimen_id)
    out = {}
    for kind in KINDS:
        numbers = [record["sweep_number"] for record in reference["sweeps"][kind]]
        if [int(sweep.id) for sweep in sweeps[kind]] != numbers:
            pytest.fail(
                f"Allen reference for specimen {specimen_id} lists different {kind} sweeps "
                "than the cache. Run: uv run python tests/build_reference.py --force"
            )
        out[kind] = [
            analyse_sweep(
                s.t,
                s.v,
                s.i,
                s.start,
                s.end,
                stimulus=kind,
                dv_cutoff=s.dv_cutoff,
                thresh_frac=s.thresh_frac,
            )
            for s in sweeps[kind]
        ]
    return out
