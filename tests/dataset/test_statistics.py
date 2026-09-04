"""Tests for dataset statistics generation (Part F)."""

from __future__ import annotations

from predictron_engine.dataset.statistics import compute_dataset_stats
from predictron_engine.dataset.store import DatasetStore
from tests.dataset.conftest import make_record


def test_empty_store_statistics(tmp_path) -> None:
    store = DatasetStore(tmp_path)
    store.initialize()
    stats = compute_dataset_stats(store)
    assert stats.record_count == 0
    assert stats.startup_count == 0
    assert stats.sectors == {}
    assert stats.missing_fields.counts == {}


def test_statistics_populated(tmp_path) -> None:
    store = DatasetStore(tmp_path)
    store.initialize()

    rec1 = make_record(
        startup_name="Acme", website="https://acme.com", record_id="a"
    ).model_copy(
        update={
            "analysis_metadata": {
                "sector": "SaaS",
                "country": "US",
                "funding_stage_at_analysis": "seed",
            }
        }
    )
    rec2 = make_record(
        startup_name="Acme", website="https://acme2.com", record_id="b"
    ).model_copy(
        update={
            "analysis_metadata": {
                "sector": "SaaS",
                "country": "US",
                "funding_stage_at_analysis": "series_a",
            }
        }
    )
    store.save_record(rec1)
    store.save_record(rec2)

    stats = compute_dataset_stats(store)
    assert stats.record_count == 2
    assert stats.startup_count == 1
    assert stats.sectors == {"SaaS": 2}
    assert stats.years[str(rec1.analysis_date.year)] == 2
    assert stats.stages == {"seed": 1, "series_a": 1}
    # Countries: both records have metadata country US
    assert stats.countries == {"US": 2}


def test_missing_field_report(tmp_path) -> None:
    store = DatasetStore(tmp_path)
    store.initialize()
    record = make_record(record_id="a")  # no investment_readiness_score
    store.save_record(record)
    stats = compute_dataset_stats(store)
    report = stats.missing_fields
    assert report.total_records == 1
    assert "prediction.investment_readiness_score" in report.counts
    assert report.counts["prediction.investment_readiness_score"] == 1


def test_duplicate_report_in_stats(tmp_path) -> None:
    store = DatasetStore(tmp_path)
    store.initialize()
    store.save_record(
        make_record(startup_name="Acme", website="https://acme.com", record_id="a")
    )
    store.save_record(
        make_record(startup_name="Acme", website="https://www.acme.com", record_id="b")
    )
    stats = compute_dataset_stats(store)
    assert stats.duplicate_group_count == 1
    assert stats.duplicate_record_count == 2


def test_stats_to_dict(tmp_path) -> None:
    store = DatasetStore(tmp_path)
    store.initialize()
    stats = compute_dataset_stats(store)
    d = stats.to_dict()
    assert "record_count" in d
    assert "sectors" in d
    assert "duplicate_report" in d
    assert "missing_fields" in d
