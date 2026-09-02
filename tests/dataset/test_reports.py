"""Tests for dataset JSON reports (Part E)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from predictron_engine.dataset.evaluation import PredictionEvaluation
from predictron_engine.dataset.models import DecisionLabel
from predictron_engine.dataset.reports import DatasetReportBuilder
from predictron_engine.dataset.store import DatasetStore
from tests.dataset.conftest import make_outcome, make_record


def _populate_store(store: DatasetStore) -> None:
    """Populate a store with records, outcomes, and evaluations."""
    store.save_record(make_record(startup_name="Alpha", record_id="r-alpha"))
    store.save_record(make_record(startup_name="Beta", record_id="r-beta"))

    # Alpha: invested and acquired (success)
    outcome_alpha = make_outcome("r-alpha", acquisition="Acq")
    outcome_alpha.outcome.acquisition_price_usd = 5_000_000
    store.save_outcome(outcome_alpha)
    ev_alpha = PredictionEvaluation.from_records(
        make_record(record_id="r-alpha", decision=DecisionLabel.INVEST),
        outcome_alpha,
    )
    # Use the record that has the same record_id to keep consistency
    store.save_evaluation(ev_alpha)


class TestReportBuilder:
    def test_empty_report_structure(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        builder = DatasetReportBuilder(store, engine_version="0.12.1")
        report = builder.build(generated_at=datetime(2024, 1, 1, tzinfo=UTC))
        assert report["dataset_summary"]["record_count"] == 0
        assert report["coverage"]["outcome_coverage"] is None
        assert report["version_metadata"]["engine_version"] == "0.12.1"
        assert report["generated_at"] == "2024-01-01T00:00:00+00:00"
        assert report["outcome_distribution"]["unknown"] == 0

    def test_report_contains_all_sections(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        _populate_store(store)
        builder = DatasetReportBuilder(
            store,
            engine_version="0.12.1",
            benchmark_version="v0.13.0",
        )
        report = builder.build(generated_at=datetime(2024, 1, 1, tzinfo=UTC))
        assert set(report.keys()) == {
            "dataset_summary",
            "coverage",
            "outcome_distribution",
            "evaluation_metrics",
            "version_metadata",
            "generated_at",
        }
        assert report["version_metadata"]["benchmark_version"] == "v0.13.0"
        assert report["dataset_summary"]["record_count"] == 2

    def test_outcome_distribution_counts(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        store.save_record(make_record(record_id="r1"))
        store.save_outcome(make_outcome("r1", shutdown=False))
        builder = DatasetReportBuilder(store, engine_version="0.12.1")
        report = builder.build(generated_at=datetime(2024, 1, 1, tzinfo=UTC))
        # shutdown=False with fully_verified status and no other signal -> unknown verdict
        assert report["outcome_distribution"]["unknown"] == 1

    def test_metrics_passthrough(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        _populate_store(store)
        builder = DatasetReportBuilder(store, engine_version="0.12.1")
        report = builder.build(generated_at=datetime(2024, 1, 1, tzinfo=UTC))
        assert "evaluation_metrics" in report
        metrics = report["evaluation_metrics"]
        assert "accuracy" in metrics
        assert "coverage" in metrics
        assert "insufficient_ground_truth_rate" in metrics

    def test_to_json_serializes(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        builder = DatasetReportBuilder(store, engine_version="0.12.1")
        s = builder.to_json(generated_at=datetime(2024, 1, 1, tzinfo=UTC))
        parsed = json.loads(s)
        assert parsed["version_metadata"]["engine_version"] == "0.12.1"

    def test_deterministic_apart_from_timestamp(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        _populate_store(store)
        builder = DatasetReportBuilder(store, engine_version="0.12.1")
        r1 = builder.build(generated_at=datetime(2024, 1, 1, tzinfo=UTC))
        r2 = builder.build(generated_at=datetime(2024, 1, 1, tzinfo=UTC))
        assert r1 == r2
