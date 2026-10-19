"""Fixtures for the tests of the vendored Allen code (run with ``--allen``)."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def specimen_sweepset_extractors(specimen_case):
    """Allen's sweep-set extractors after cell processing.

    ``cell.process()`` reduces short squares to the common-amplitude sweeps and
    ramps to the spiking ones.
    """
    cell = specimen_case.build_cell_extractor_for_nwb()
    cell.process()
    return {
        "long_square": cell.long_squares_features(),
        "short_square": cell.short_squares_features(),
        "ramp": cell.ramps_features(),
    }
