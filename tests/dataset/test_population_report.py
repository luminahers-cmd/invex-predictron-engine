"""Tests for population run reports (Project V4)."""

from __future__ import annotations

from datetime import UTC, datetime

from predictron_engine.dataset.acquisition.pipeline import (
    AcquisitionMetrics,
    AcquisitionResult,
)
from predictron_engine.dataset.population_report import (
    FailureReason,
    build_population_report,
)
from predictron_engine.dataset.store import DatasetStore


def _result(source: str, imported: int = 0) -> AcquisitionResult:
    metrics = AcquisitionMetrics(
        total_raw_records=imported,
        imported=imported,
        failed=0,
        skipped=0,
    )
    return AcquisitionResult(metrics=metrics, source_name=source)


def test_report_is_deterministic(dataset_store: DatasetStore) -> None:
    pinned = datetime(2026, 1, 1, tzinfo=UTC)
    kwargs: dict = dict(
        results=[_result("csv_export", imported=2)],
        store=dataset_store,
        start_record_count=0,
        elapsed_seconds=1.0,
        run_id="run-1",
        generated_at=pinned,
    )
    r1 = build_population_report(**kwargs)
    r2 = build_population_report(**kwargs)
    assert r1.to_json() == r2.to_json()


def test_report_counts_and_sources(dataset_store: DatasetStore) -> None:
    report = build_population_report(
        [_result("csv_export", imported=2)],
        store=dataset_store,
        start_record_count=0,
        elapsed_seconds=2.0,
        run_id="run-1",
    )
    assert report.to_dict()["report_type"] == "population_report"
    assert report.counts["imported"] == 2
    assert report.source_names == ["csv_export"]
    assert report.dry_run is False


def test_report_failure_reasons(dataset_store: DatasetStore) -> None:
    def err_result(source: str, error: str) -> AcquisitionResult:
        metrics = AcquisitionMetrics(errors=[error])
        return AcquisitionResult(metrics=metrics, source_name=source)

    results = [
        err_result("a", "parse error"),
        err_result("b", "network timeout"),
        err_result("c", "parse error"),
    ]
    report = build_population_report(
        results,
        store=dataset_store,
        start_record_count=0,
        elapsed_seconds=0.0,
        run_id="run-1",
    )
    reasons = report.failure_reasons
    assert reasons[0] == {"reason": "parse error", "count": 2}
    assert {r["reason"] for r in reasons} == {"parse error", "network timeout"}


def test_report_write_roundtrip(dataset_store: DatasetStore, tmp_path) -> None:
    report = build_population_report(
        [_result("csv_export", imported=1)],
        store=dataset_store,
        start_record_count=0,
        elapsed_seconds=0.5,
        run_id="run-1",
    )
    import json

    path = report.write(tmp_path / "nested" / "report.json")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == "run-1"
    assert data["counts"]["imported"] == 1


def test_failure_reason_to_dict() -> None:
    assert FailureReason("boom", count=3).to_dict() == {
        "reason": "boom",
        "count": 3,
    }


def test_quality_embedded_in_report(dataset_store: DatasetStore) -> None:
    report = build_population_report(
        [_result("csv_export")],
        store=dataset_store,
        start_record_count=0,
        elapsed_seconds=0.1,
        run_id="run-1",
    )
    assert "finding_count" in report.quality
    assert report.quality["records_checked"] == 0
