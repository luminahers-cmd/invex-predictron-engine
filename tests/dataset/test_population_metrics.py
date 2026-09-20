"""Tests for population run metrics (Project V4)."""

from __future__ import annotations

from predictron_engine.dataset.acquisition.pipeline import (
    AcquisitionMetrics,
    AcquisitionResult,
)
from predictron_engine.dataset.population_metrics import (
    PopulationRunMetrics,
    compute_population_metrics,
)


def _result(source: str, imported: int = 0, failed: int = 0) -> AcquisitionResult:
    metrics = AcquisitionMetrics(
        total_raw_records=imported + failed,
        imported=imported,
        failed=failed,
        skipped=0,
    )
    return AcquisitionResult(metrics=metrics, source_name=source)


def test_compute_aggregates_counts_and_sources() -> None:
    results = [
        _result("csv_export", imported=4),
        _result("crawler", imported=3),
    ]
    metrics = compute_population_metrics(
        results,
        start_record_count=0,
        end_record_count=7,
        elapsed_seconds=2.0,
    )
    assert metrics.total_imported == 7
    assert metrics.total_processed == 7
    assert metrics.sources["csv_export"]["imported"] == 4
    assert metrics.sources["crawler"]["imported"] == 3
    assert metrics.import_rate() == 3.5
    assert metrics.records_per_hour() == 12600
    assert metrics.dataset_growth() == 7


def test_duplicate_rate_and_validation_success_rate() -> None:
    results = [_result("csv_export", imported=2)]
    metrics = compute_population_metrics(
        results,
        start_record_count=1,
        end_record_count=3,
        elapsed_seconds=1.0,
        duplicate_count=1,
    )
    assert metrics.total_duplicates == 1
    assert metrics.duplicate_rate == 0.5
    # Duplicates count as successes; with no duplicates the rate is 1.0.
    assert compute_population_metrics(
        results,
        start_record_count=1,
        end_record_count=3,
        elapsed_seconds=1.0,
    ).validation_success_rate == 1.0


def test_zero_elapsed_yields_none_rates() -> None:
    metrics = compute_population_metrics(
        [],
        start_record_count=0,
        end_record_count=0,
        elapsed_seconds=0.0,
    )
    assert metrics.import_rate() is None
    assert metrics.records_per_hour() is None
    assert metrics.total_imported == 0


def test_sources_override() -> None:
    metrics = compute_population_metrics(
        [_result("a", imported=1)],
        start_record_count=0,
        end_record_count=1,
        elapsed_seconds=1.0,
        sources={"override": {"processed": 5, "imported": 5, "skipped": 0, "failed": 0}},
    )
    expected = {"override": {"processed": 5, "imported": 5, "skipped": 0, "failed": 0}}
    assert metrics.sources == expected


def test_history_passthrough_and_to_dict() -> None:
    history = [{"batch_id": "b1", "record_count": 3}]
    metrics = compute_population_metrics(
        [_result("a", imported=2)],
        start_record_count=0,
        end_record_count=2,
        elapsed_seconds=0.5,
        history=history,
    )
    assert metrics.history == history
    data = metrics.to_dict()
    assert data["import_history"] == history
    assert data["dataset_growth"] == 2


def test_rate_helpers_guard_division() -> None:
    m = PopulationRunMetrics(total_imported=5, elapsed_seconds=0.0)
    assert m.import_rate() is None
