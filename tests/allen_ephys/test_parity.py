from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import CACHED_SPECIMEN_IDS
from tests.utils.match_data_utils import (
    DEFAULT_TOLERANCE,
    FEATURE_TOLERANCES,
    expected_db_feature_dict,
    flatten_for_db,
)


def _db_feature_params():
    params = []
    for specimen_id in CACHED_SPECIMEN_IDS:
        meta_path = Path("tests/data") / f"cell_{specimen_id}_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = expected_db_feature_dict(meta)
        for key in sorted(expected):
            params.append(
                pytest.param(
                    specimen_id,
                    key,
                    id=f"specimen_{specimen_id}__{key}",
                )
            )
    return params


@pytest.fixture(scope="session")
def db_parity_payload(specimen_cases):
    payload: dict[int, tuple[dict[str, float], dict[str, float]]] = {}
    for specimen_id, specimen_case in specimen_cases.items():
        expected = expected_db_feature_dict(specimen_case.meta)
        computed = flatten_for_db(specimen_case.compute_cell_features())
        payload[specimen_id] = (expected, computed)
    return payload


@pytest.mark.parametrize(("specimen_id", "feature_key"), _db_feature_params())
def test_database_feature_match_by_specimen(
    specimen_id: int,
    feature_key: str,
    db_parity_payload,
):
    expected, computed = db_parity_payload[specimen_id]
    exp = expected[feature_key]
    comp = computed.get(feature_key, float("nan"))

    if np.isnan(exp) and np.isnan(comp):
        return

    assert np.isnan(exp) == np.isnan(comp), (
        f"specimen={specimen_id} feature={feature_key} expected={exp} computed={comp}"
    )

    tol = FEATURE_TOLERANCES.get(feature_key, DEFAULT_TOLERANCE)
    assert np.isclose(exp, comp, atol=tol.atol, rtol=tol.rtol), (
        f"specimen={specimen_id} feature={feature_key} expected={exp} computed={comp} "
        f"atol={tol.atol} rtol={tol.rtol}"
    )
