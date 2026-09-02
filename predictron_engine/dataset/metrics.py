"""Aggregate evaluation metrics (Part D).

Computes deterministic aggregate metrics over a collection of
:class:`PredictionEvaluation` records:

  - accuracy
  - precision
  - recall
  - specificity
  - F1
  - balanced accuracy
  - coverage
  - insufficient-ground-truth rate

Respects the V1 ground-truth design: outcomes that are unknown (not
verified) or predictions that are not binary-classifiable are treated as
insufficient ground truth and excluded from scoreable metrics, never
fabricated.  Metrics that are undefined (e.g. precision with no
predicted positives) are reported as ``None`` rather than NaN or zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from predictron_engine.dataset.evaluation import (
    EvaluationVerdict,
    PredictionEvaluation,
)
from predictron_engine.dataset.models import DecisionLabel
from predictron_engine.dataset.outcomes import OutcomeVerdict


class BinaryLabel(NamedTuple):
    """Binary classifiable view of a prediction/outcome pair.

    ``predicted_positive``: True when the engine decided INVEST/STRONG_INVEST,
    False when it decided PASS.  Neither value is set (None) when the
    decision is a neutral watch/investigate label and cannot be classified
    as a binary positive/negative call.

    ``actual_positive``: True on a verified SUCCESS, False on a verified
    FAILURE, None when the outcome is unknown/insufficient.
    """

    predicted_positive: bool | None
    actual_positive: bool | None


@dataclass
class EvaluationMetrics:
    """Aggregate metrics over a set of evaluations.

    ``total`` is the number of evaluations examined.  ``scoreable`` is
    the number with sufficient binary ground truth.  Every rate-style
    metric is ``None`` when undefined.
    """

    total: int = 0
    scoreable: int = 0
    insufficient_ground_truth: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    specificity: float | None = None
    f1: float | None = None
    balanced_accuracy: float | None = None
    coverage: float | None = None
    insufficient_ground_truth_rate: float | None = None
    verdict_counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        """Dict view for reports and CLI."""
        return {
            "total": self.total,
            "scoreable": self.scoreable,
            "insufficient_ground_truth": self.insufficient_ground_truth,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "accuracy": _round_or_none(self.accuracy),
            "precision": _round_or_none(self.precision),
            "recall": _round_or_none(self.recall),
            "specificity": _round_or_none(self.specificity),
            "f1": _round_or_none(self.f1),
            "balanced_accuracy": _round_or_none(self.balanced_accuracy),
            "coverage": _round_or_none(self.coverage),
            "insufficient_ground_truth_rate": _round_or_none(
                self.insufficient_ground_truth_rate
            ),
            "verdict_counts": self.verdict_counts,
        }


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def compute_evaluation_metrics(
    evaluations: list[PredictionEvaluation],
) -> EvaluationMetrics:
    """Compute aggregate metrics over a list of evaluations.

    Deterministic: identical inputs always produce identical metrics.
    Never fabricates labels — unknown outcomes remain un-scoreable.
    """
    metrics = EvaluationMetrics(total=len(evaluations))
    tp = tn = fp = fn = 0
    scoreable = 0
    insufficient = 0

    for ev in evaluations:
        key = ev.verdict.value
        metrics.verdict_counts[key] = metrics.verdict_counts.get(key, 0) + 1

        label = binary_label(ev)
        # A record is only binary-scoreable when both the prediction and
        # the outcome can be crisply classified AND the verdict is a
        # crisp correct/incorrect call.  Neutral decisions, unverified
        # outcomes, and partially-correct/inconclusive verdicts are never
        # coerced into fabricated binary labels.
        if label.predicted_positive is None or label.actual_positive is None:
            insufficient += 1
            continue
        if ev.verdict not in (
            EvaluationVerdict.CORRECT,
            EvaluationVerdict.INCORRECT,
        ):
            insufficient += 1
            continue

        scoreable += 1
        if ev.verdict == EvaluationVerdict.CORRECT:
            if label.actual_positive:
                tp += 1
            else:
                tn += 1
        else:  # INCORRECT
            if label.actual_positive:
                # predicted False but outcome success -> false negative
                fn += 1
            else:
                # predicted True but outcome failure -> false positive
                fp += 1

    metrics.true_positives = tp
    metrics.true_negatives = tn
    metrics.false_positives = fp
    metrics.false_negatives = fn
    metrics.scoreable = scoreable
    metrics.insufficient_ground_truth = insufficient

    denom = tp + tn + fp + fn
    if denom > 0:
        metrics.accuracy = (tp + tn) / denom

    if tp + fp > 0:
        metrics.precision = tp / (tp + fp)
    if tp + fn > 0:
        metrics.recall = tp / (tp + fn)
    if tn + fp > 0:
        metrics.specificity = tn / (tn + fp)

    if (
        metrics.precision is not None
        and metrics.recall is not None
        and (metrics.precision + metrics.recall) > 0
    ):
        metrics.f1 = (
            2
            * metrics.precision
            * metrics.recall
            / (metrics.precision + metrics.recall)
        )

    if (
        metrics.recall is not None
        and metrics.specificity is not None
    ):
        metrics.balanced_accuracy = (
            metrics.recall + metrics.specificity
        ) / 2

    if metrics.total > 0:
        metrics.coverage = metrics.scoreable / metrics.total
        metrics.insufficient_ground_truth_rate = (
            metrics.insufficient_ground_truth / metrics.total
        )

    return metrics


def binary_label(ev: PredictionEvaluation) -> BinaryLabel:
    """Return the binary classifiable view of an evaluation.

    Neither component is fabricated: a neutral decision or an unverified
    outcome yields ``None`` for the corresponding side.
    """
    predicted_positive = _predicted_positive(ev)
    actual_positive = _actual_positive(ev)
    return BinaryLabel(
        predicted_positive=predicted_positive,
        actual_positive=actual_positive,
    )


def _predicted_positive(ev: PredictionEvaluation) -> bool | None:
    decision = ev.prediction.decision
    if decision in (DecisionLabel.STRONG_INVEST, DecisionLabel.INVEST):
        return True
    if decision == DecisionLabel.PASS:
        return False
    return None


def _actual_positive(ev: PredictionEvaluation) -> bool | None:
    verdict = ev.outcome_record.derive_verdict()
    if verdict == OutcomeVerdict.SUCCESS:
        return True
    if verdict == OutcomeVerdict.FAILURE:
        return False
    return None
