"""Tests for aggregate evaluation metrics (Part D)."""

from __future__ import annotations

from predictron_engine.dataset.evaluation import (
    EvaluationVerdict,
    PredictionEvaluation,
)
from predictron_engine.dataset.metrics import (
    BinaryLabel,
    compute_evaluation_metrics,
)
from predictron_engine.dataset.models import DecisionLabel
from predictron_engine.dataset.outcomes import OutcomeRecord
from tests.dataset.conftest import make_outcome, make_record


def _evaluation(
    decision: DecisionLabel,
    outcome_status="fully_verified",
    shutdown: bool | None = None,
    acquisition: str | None = None,
) -> PredictionEvaluation:
    record = make_record(decision=decision)
    outcome = make_outcome(
        record.record_id,
        shutdown=shutdown,
        acquisition=acquisition,
    )
    # acquire price to make acquisition a verified success
    if acquisition:
        outcome.outcome.acquisition_price_usd = 1_000_000
    return PredictionEvaluation.from_records(record, outcome)


class TestBinaryLabel:
    def test_positive_prediction(self) -> None:
        record = make_record(decision=DecisionLabel.INVEST)
        outcome = make_outcome(record.record_id, acquisition="X")
        outcome.outcome.acquisition_price_usd = 1_000_000
        ev = PredictionEvaluation.from_records(record, outcome)
        label = BinaryLabel(
            predicted_positive=ev.prediction.decision in (
                DecisionLabel.STRONG_INVEST,
                DecisionLabel.INVEST,
            ),
            actual_positive=ev.outcome_record.derive_verdict().value == "success",
        )
        assert label.predicted_positive is True
        assert label.actual_positive is True

    def test_neutral_decision_is_not_binary(self) -> None:
        record = make_record(decision=DecisionLabel.WATCH)
        outcome = make_outcome(record.record_id, acquisition="X")
        outcome.outcome.acquisition_price_usd = 1_000_000
        ev = PredictionEvaluation.from_records(record, outcome)
        # WATCH is not crisply positive or negative
        assert ev.prediction.decision == DecisionLabel.WATCH


class TestComputeMetrics:
    def test_empty(self) -> None:
        metrics = compute_evaluation_metrics([])
        assert metrics.total == 0
        assert metrics.scoreable == 0
        assert metrics.accuracy is None
        assert metrics.insufficient_ground_truth_rate is None

    def test_all_success_and_correct(self) -> None:
        evs = [
            _evaluation(DecisionLabel.INVEST, acquisition="A"),
            _evaluation(DecisionLabel.INVEST, acquisition="B"),
        ]
        metrics = compute_evaluation_metrics(evs)
        assert metrics.total == 2
        assert metrics.scoreable == 2
        assert metrics.true_positives == 2
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.coverage == 1.0
        assert metrics.insufficient_ground_truth_rate == 0.0

    def test_all_pass_on_failures_are_true_negatives(self) -> None:
        evs = [
            _evaluation(DecisionLabel.PASS, shutdown=True),
            _evaluation(DecisionLabel.PASS, shutdown=True),
        ]
        metrics = compute_evaluation_metrics(evs)
        assert metrics.scoreable == 2
        assert metrics.true_negatives == 2
        assert metrics.true_positives == 0
        assert metrics.accuracy == 1.0
        # precision/recall undefined with no positives
        assert metrics.precision is None
        assert metrics.recall is None

    def test_mixed_accuracy(self) -> None:
        # TP: invest+success, TN: pass+failure, FP: invest+failure, FN: pass+success
        evs = [
            _evaluation(DecisionLabel.INVEST, acquisition="A"),   # TP
            _evaluation(DecisionLabel.PASS, shutdown=True),        # TN
            _evaluation(DecisionLabel.INVEST, shutdown=True),      # FP
            _evaluation(DecisionLabel.PASS, acquisition="B"),      # FN
        ]
        metrics = compute_evaluation_metrics(evs)
        assert metrics.true_positives == 1
        assert metrics.true_negatives == 1
        assert metrics.false_positives == 1
        assert metrics.false_negatives == 1
        assert metrics.accuracy == 0.5
        assert metrics.precision == 0.5  # TP/(TP+FP)=1/2
        assert metrics.recall == 0.5  # TP/(TP+FN)=1/2
        assert metrics.specificity == 0.5  # TN/(TN+FP)=1/2
        assert metrics.f1 == 0.5
        assert metrics.balanced_accuracy == 0.5

    def test_unknown_outcomes_are_insufficient(self) -> None:
        from predictron_engine.dataset.outcomes import OutcomeStatus, StartupOutcome

        rec = make_record(decision=DecisionLabel.INVEST)
        outcome = OutcomeRecord(
            record_id=rec.record_id,
            outcome=StartupOutcome(status=OutcomeStatus.UNKNOWN),
        )
        ev = PredictionEvaluation.from_records(rec, outcome)
        metrics = compute_evaluation_metrics([ev])
        assert metrics.total == 1
        assert metrics.scoreable == 0
        assert metrics.insufficient_ground_truth == 1
        assert metrics.coverage == 0.0
        assert metrics.insufficient_ground_truth_rate == 1.0
        assert metrics.accuracy is None

    def test_neutral_decision_is_insufficient(self) -> None:
        ev = _evaluation(DecisionLabel.WATCH, shutdown=True)
        metrics = compute_evaluation_metrics([ev])
        assert metrics.scoreable == 0
        assert metrics.insufficient_ground_truth == 1

    def test_verdict_counts(self) -> None:
        evs = [
            _evaluation(DecisionLabel.INVEST, acquisition="A"),
            _evaluation(DecisionLabel.PASS, shutdown=True),
        ]
        metrics = compute_evaluation_metrics(evs)
        assert metrics.verdict_counts[EvaluationVerdict.CORRECT.value] == 2

    def test_summary_rounds_and_returns_none_for_undefined(self) -> None:
        metrics = compute_evaluation_metrics([])
        summary = metrics.summary()
        assert summary["accuracy"] is None
        assert summary["total"] == 0
