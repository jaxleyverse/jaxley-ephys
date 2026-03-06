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


def test_database_features_match_with_tolerance(specimen_case):
    entries = compare_with_database(specimen_case)
    failures = [
        (
            key,
            entry.status,
            entry.expected,
            entry.computed,
            entry.abs_diff,
            entry.rel_diff,
        )
        for key, entry in sorted(entries.items())
        if entry.status != "match"
    ]

    assert not failures, (
        f"Specimen {specimen_case.specimen_id} has {len(failures)} non-matching features: "
        f"{failures}"
    )
