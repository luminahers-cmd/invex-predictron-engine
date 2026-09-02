"""Evaluation pipeline (Part C).

Builds the pipeline that:
  - loads a DatasetRecord and its OutcomeRecord,
  - creates a PredictionEvaluation,
  - stores the evaluation,
  - computes aggregate metrics.

Uses the existing :class:`PredictionEvaluation` model and
``PredictionEvaluation.from_records`` for verdict derivation.  No
prediction is ever modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from predictron_engine.dataset.evaluation import PredictionEvaluation
from predictron_engine.dataset.metrics import EvaluationMetrics
from predictron_engine.dataset.models import DatasetRecord
from predictron_engine.dataset.outcomes import OutcomeRecord


class _StoreLike(Protocol):
    """Minimal store interface required by the evaluation pipeline."""

    def find_outcome_by_record(self, record_id: str) -> OutcomeRecord | None: ...

    def save_evaluation(self, evaluation: PredictionEvaluation) -> object: ...

    def list_records(self) -> list[str]: ...

    def load_record(self, record_id: str) -> DatasetRecord | None: ...

    def load_evaluation(self, evaluation_id: str) -> PredictionEvaluation | None: ...

    def list_evaluations(self) -> list[str]: ...


class EvaluationPipeline:
    """Orchestrates evaluation of predictions against outcomes.

    The pipeline consumes records already present in the store and the
    matching outcome records, and persists evaluations.  Aggregated
    metrics are delegated to :func:`predictron_engine.dataset.metrics.
    compute_metrics`.
    """

    def __init__(self, store: _StoreLike) -> None:
        self._store = store

    def evaluate_record(
        self, record: DatasetRecord, outcome: OutcomeRecord
    ) -> PredictionEvaluation:
        """Create and store an evaluation for a record/outcome pair."""
        evaluation = PredictionEvaluation.from_records(record, outcome)
        self._store.save_evaluation(evaluation)
        return evaluation

    def evaluate_all(self) -> EvaluationBatchResult:
        """Evaluate every record that has an outcome, storing evaluations.

        Records without a linked outcome are counted as skipped (they
        have no ground truth to evaluate against) and are never
        fabricated into an evaluation.
        """
        result = EvaluationBatchResult()
        for record_id in self._store.list_records():
            try:
                record = self._store.load_record(record_id)
            except Exception:  # noqa: BLE001 - schema mismatch on load
                result.missing_records.append(record_id)
                continue
            if record is None:
                result.missing_records.append(record_id)
                continue
            outcome = self._store.find_outcome_by_record(record_id)
            if outcome is None:
                result.skipped_no_outcome.append(record_id)
                continue
            evaluation = self.evaluate_record(record, outcome)
            result.evaluations.append(evaluation)
        return result

    def compute_metrics(self) -> EvaluationMetrics:
        """Compute aggregate metrics over all stored evaluations."""
        from predictron_engine.dataset.metrics import compute_evaluation_metrics

        evaluations: list[PredictionEvaluation] = []
        for eid in self._store.list_evaluations():
            try:
                evaluation = self._store.load_evaluation(eid)
            except Exception:  # noqa: BLE001 - skip unreadable evaluations
                continue
            if evaluation is not None:
                evaluations.append(evaluation)
        return compute_evaluation_metrics(evaluations)


@dataclass
class EvaluationBatchResult:
    """Result of evaluating all records in a store."""

    evaluations: list[PredictionEvaluation] = field(default_factory=list)
    skipped_no_outcome: list[str] = field(default_factory=list)
    missing_records: list[str] = field(default_factory=list)

    @property
    def evaluated_count(self) -> int:
        return len(self.evaluations)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_no_outcome)


def evaluate_record_pair(
    record: DatasetRecord, outcome: OutcomeRecord
) -> PredictionEvaluation:
    """Create a PredictionEvaluation without persisting.

    Thin wrapper over ``PredictionEvaluation.from_records`` so the
    evaluation pipeline exposes a single evaluation entry point.
    """
    return PredictionEvaluation.from_records(record, outcome)
