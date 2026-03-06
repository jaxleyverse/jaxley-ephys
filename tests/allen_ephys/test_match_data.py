from tests.utils.match_data_utils import compare_with_database


def test_database_comparison_status_values(specimen_case):
    entries = compare_with_database(specimen_case)
    assert entries
    allowed = {
        "match",
        "mismatch",
        "unit_suspect",
        "expected_missing",
        "computed_missing",
    }
    assert all(entry.status in allowed for entry in entries.values())
