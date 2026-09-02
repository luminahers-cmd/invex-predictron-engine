"""Immutable prediction evaluation storage (Part D).

Stores the relationship between original predictions and later outcomes
without modifying the prediction.  The prediction must remain immutable
once recorded.  Evaluation metadata captures when and how the comparison
was made.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from predictron_engine.dataset.models import DatasetRecord, PredictionSummary
from predictron_engine.dataset.outcomes import OutcomeRecord, OutcomeVerdict


class EvaluationVerdict(str, Enum):
    """Verdict on prediction accuracy.

    Compares the original decision against the actual outcome.
    """

    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    INCONCLUSIVE = "inconclusive"
    UNABLE_TO_EVALUATE = "unable_to_evaluate"


class PredictionOutcomeAlignment(str, Enum):
    """How well the prediction direction matches the outcome direction."""

    STRONG_MATCH = "strong_match"
    PARTIAL_MATCH = "partial_match"
    MISMATCH = "mismatch"
    NEUTRAL = "neutral"


class EvaluationMetadata(BaseModel):
    """Metadata about the evaluation process.

    Records when and how the evaluation was performed, by whom,
    and any configuration used.
    """

    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this evaluation was performed",
    )
    evaluator_version: str = Field(
        default="1.0.0",
        description="Version of the evaluation logic",
    )
    evaluation_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration used for this evaluation",
    )
    notes: str = Field(
        default="", description="Free-text evaluation notes"
    )


class PredictionEvaluation(BaseModel):
    """Immutable evaluation comparing a prediction against an outcome.

    The prediction field is a frozen copy — it is never modified after
    creation.  The evaluation records the comparison between the
    prediction and the actual outcome, along with metadata about
    the evaluation process.
    """

    evaluation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique evaluation identifier (UUIDv4)",
    )
    record_id: str = Field(
        ..., description="Dataset record this evaluation belongs to"
    )
    prediction: PredictionSummary = Field(
        ..., description="Frozen copy of the original prediction (immutable)"
    )
    outcome_record: OutcomeRecord = Field(
        ..., description="The outcome data used for comparison"
    )
    verdict: EvaluationVerdict = Field(
        default=EvaluationVerdict.UNABLE_TO_EVALUATE,
        description="Verdict on prediction accuracy",
    )
    alignment: PredictionOutcomeAlignment = Field(
        default=PredictionOutcomeAlignment.NEUTRAL,
        description="Direction alignment between prediction and outcome",
    )
    decision_match: bool | None = Field(
        default=None,
        description=(
            "True if the engine's decision was directionally correct"
        ),
    )
    confidence_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="How well the confidence calibrated to reality",
    )
    score_delta: float | None = Field(
        default=None,
        description=(
            "Difference between predicted composite_score and "
            "a derived score from outcomes"
        ),
    )
    evaluation_metadata: EvaluationMetadata = Field(
        default_factory=EvaluationMetadata,
        description="Metadata about the evaluation process",
    )
    dimension_evaluations: dict[str, str] = Field(
        default_factory=dict,
        description="Per-dimension evaluation notes",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this evaluation was created",
    )

    @classmethod
    def from_records(
        cls,
        dataset_record: DatasetRecord,
        outcome_record: OutcomeRecord,
    ) -> PredictionEvaluation:
        """Create an evaluation from a dataset record and outcome record.

        The prediction is copied (frozen) from the dataset record.
        The verdict is derived deterministically from the comparison.
        """
        prediction_copy = PredictionSummary(
            decision=dataset_record.prediction.decision,
            confidence=dataset_record.prediction.confidence,
            composite_score=dataset_record.prediction.composite_score,
            dimension_scores=dict(dataset_record.prediction.dimension_scores),
            investment_readiness_score=dataset_record.prediction.investment_readiness_score,
            recommendation_count=dataset_record.prediction.recommendation_count,
        )
        verdict = _derive_evaluation_verdict(
            dataset_record.prediction, outcome_record
        )
        alignment = _derive_alignment(
            dataset_record.prediction, outcome_record
        )
        return cls(
            record_id=dataset_record.record_id,
            prediction=prediction_copy,
            outcome_record=outcome_record,
            verdict=verdict,
            alignment=alignment,
        )


def _derive_evaluation_verdict(
    prediction: PredictionSummary,
    outcome_record: OutcomeRecord,
) -> EvaluationVerdict:
    """Derive an evaluation verdict from prediction and outcome.

    Compares the engine's decision against the observed outcome
    to determine prediction accuracy.
    """
    from predictron_engine.dataset.models import DecisionLabel

    if outcome_record.outcome.status.value == "unknown":
        return EvaluationVerdict.UNABLE_TO_EVALUATE

    outcome_verdict = outcome_record.derive_verdict()
    if outcome_verdict == OutcomeVerdict.UNKNOWN:
        return EvaluationVerdict.INCONCLUSIVE

    decision = prediction.decision

    if decision in (DecisionLabel.STRONG_INVEST, DecisionLabel.INVEST):
        if outcome_verdict == OutcomeVerdict.SUCCESS:
            return EvaluationVerdict.CORRECT
        if outcome_verdict == OutcomeVerdict.PARTIAL_SUCCESS:
            return EvaluationVerdict.PARTIALLY_CORRECT
        if outcome_verdict == OutcomeVerdict.FAILURE:
            return EvaluationVerdict.INCORRECT

    if decision == DecisionLabel.PASS:
        if outcome_verdict == OutcomeVerdict.FAILURE:
            return EvaluationVerdict.CORRECT
        if outcome_verdict == OutcomeVerdict.SUCCESS:
            return EvaluationVerdict.INCORRECT
        return EvaluationVerdict.PARTIALLY_CORRECT

    if decision in (DecisionLabel.WATCH, DecisionLabel.INVESTIGATE_FURTHER):
        if outcome_verdict == OutcomeVerdict.FAILURE:
            return EvaluationVerdict.PARTIALLY_CORRECT
        if outcome_verdict == OutcomeVerdict.SUCCESS:
            return EvaluationVerdict.PARTIALLY_CORRECT
        return EvaluationVerdict.INCONCLUSIVE

    return EvaluationVerdict.UNABLE_TO_EVALUATE


def _derive_alignment(
    prediction: PredictionSummary,
    outcome_record: OutcomeRecord,
) -> PredictionOutcomeAlignment:
    """Derive alignment between prediction direction and outcome direction."""
    from predictron_engine.dataset.models import DecisionLabel

    if outcome_record.outcome.status.value == "unknown":
        return PredictionOutcomeAlignment.NEUTRAL

    outcome_verdict = outcome_record.derive_verdict()
    decision = prediction.decision

    positive_decisions = {DecisionLabel.STRONG_INVEST, DecisionLabel.INVEST}
    neutral_decisions = {DecisionLabel.WATCH, DecisionLabel.INVESTIGATE_FURTHER}
    negative_decisions = {DecisionLabel.PASS}

    positive_outcomes = {OutcomeVerdict.SUCCESS}
    negative_outcomes = {OutcomeVerdict.FAILURE}

    if decision in positive_decisions and outcome_verdict in positive_outcomes:
        return PredictionOutcomeAlignment.STRONG_MATCH
    if decision in negative_decisions and outcome_verdict in negative_outcomes:
        return PredictionOutcomeAlignment.STRONG_MATCH
    if decision in positive_decisions and outcome_verdict in negative_outcomes:
        return PredictionOutcomeAlignment.MISMATCH
    if decision in negative_decisions and outcome_verdict in positive_outcomes:
        return PredictionOutcomeAlignment.MISMATCH
    if decision in neutral_decisions:
        return PredictionOutcomeAlignment.NEUTRAL
    if outcome_verdict == OutcomeVerdict.PARTIAL_SUCCESS:
        return PredictionOutcomeAlignment.PARTIAL_MATCH

    return PredictionOutcomeAlignment.NEUTRAL
