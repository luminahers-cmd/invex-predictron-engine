"""Tests for deduplication logic (Part E)."""

from __future__ import annotations

from predictron_engine.dataset.dedup import find_duplicates, match_record_to_store
from tests.dataset.conftest import make_record


def test_duplicate_by_website() -> None:
    rec1 = make_record(startup_name="Acme Corp", website="https://acme.com", record_id="a")
    rec2 = make_record(
        startup_name="Acme Corporation",
        website="https://www.acme.com",
        record_id="b",
    )
    report = find_duplicates([rec1, rec2])
    assert report.group_count == 1
    assert report.duplicate_record_count == 2
    assert report.method_counts.get("website", 0) == 1


def test_duplicate_by_name() -> None:
    rec1 = make_record(startup_name="Acme Inc", website="https://site1.com", record_id="a")
    rec2 = make_record(startup_name="acme inc.", website="https://site2.com", record_id="b")
    report = find_duplicates([rec1, rec2])
    assert report.group_count == 1
    assert report.method_counts.get("name", 0) == 1


def test_duplicate_by_identifier() -> None:
    rec1 = make_record(record_id="a").model_copy(
        update={"analysis_metadata": {"sec_cik": "000123"}}
    )
    rec2 = make_record(record_id="b").model_copy(
        update={"analysis_metadata": {"sec_cik": "000123"}}
    )
    report = find_duplicates([rec1, rec2])
    assert report.group_count == 1
    assert report.method_counts.get("identifier", 0) == 1


def test_distinct_records_no_group() -> None:
    rec1 = make_record(startup_name="Alpha", website="https://alpha.com", record_id="a")
    rec2 = make_record(startup_name="Beta", website="https://beta.com", record_id="b")
    report = find_duplicates([rec1, rec2])
    assert report.group_count == 0
    assert report.duplicate_record_count == 0


def test_match_record_to_store() -> None:
    existing = [
        make_record(startup_name="Acme", website="https://acme.com", record_id="a")
    ]
    candidate = make_record(
        startup_name="Acme Inc", website="https://www.acme.com", record_id="z"
    )
    matched, method = match_record_to_store(candidate, existing)
    assert matched is not None
    assert method == "website"

    unrelated = make_record(startup_name="Other", website="https://other.com", record_id="y")
    matched, method = match_record_to_store(unrelated, existing)
    assert matched is None
    assert method == "none"
