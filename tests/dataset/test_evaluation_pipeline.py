"""Tests for the evaluation pipeline (Part C)."""

from __future__ import annotations

from predictron_engine.dataset.evaluation import EvaluationVerdict
from predictron_engine.dataset.evaluation_pipeline import (
    EvaluationBatchResult,
    EvaluationPipeline,
    evaluate_record_pair,
)
from predictron_engine.dataset.models import DecisionLabel
from predictron_engine.dataset.store import DatasetStore
from tests.dataset.conftest import make_outcome, make_record


class TestEvaluationPipeline:
    def test_evaluate_record_creates_and_stores(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        record = make_record(decision=DecisionLabel.INVEST)
        outcome = make_outcome(record.record_id, acquisition="Acq")
        outcome.outcome.acquisition_price_usd = 1_000_000
        store.save_record(record)
        store.save_outcome(outcome)

        pipeline = EvaluationPipeline(store)
        evaluation = pipeline.evaluate_record(record, outcome)
        assert evaluation.verdict == EvaluationVerdict.CORRECT
        assert store.find_evaluation_by_record(record.record_id) is not None

    def test_evaluate_all_skips_records_without_outcome(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        with_outcome = make_record(decision=DecisionLabel.PASS)
        without_outcome = make_record(startup_name="No Outcome")
        store.save_record(with_outcome)
        store.save_record(without_outcome)
        outcome = make_outcome(with_outcome.record_id, shutdown=True)
        store.save_outcome(outcome)

        pipeline = EvaluationPipeline(store)
        result = pipeline.evaluate_all()
        assert isinstance(result, EvaluationBatchResult)
        assert result.evaluated_count == 1
        assert result.skipped_count == 1
        assert result.missing_records == []

    def test_evaluate_all_missing_records(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        # Write a record that cannot be parsed (schema mismatch) so it is
        # still listed by list_records but fails to load.

        (store._records_dir / "bad-id.json").write_text(
            "{ not valid json", encoding="utf-8"
        )
        pipeline = EvaluationPipeline(store)
        result = pipeline.evaluate_all()
        assert result.missing_records == ["bad-id"]

    def test_compute_metrics_no_evaluations(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        pipeline = EvaluationPipeline(store)
        metrics = pipeline.compute_metrics()
        assert metrics.total == 0

    def test_evaluate_record_pair_no_persist(self, tmp_path) -> None:
        store = DatasetStore(tmp_path)
        store.initialize()
        record = make_record(decision=DecisionLabel.PASS)
        outcome = make_outcome(record.record_id, shutdown=True)
        ev = evaluate_record_pair(record, outcome)
        assert ev.verdict == EvaluationVerdict.CORRECT
        assert store.count_evaluations() == 0
