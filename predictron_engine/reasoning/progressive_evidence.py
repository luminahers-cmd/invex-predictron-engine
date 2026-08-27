"""Progressive Evidence Evaluation — early-stop and adaptive evidence gathering (Sprint 8).

Implements progressive evaluation of evidence with early-stop when
evidence becomes overwhelmingly conclusive, and continued gathering
when uncertainty remains high.

Key principles:
  - Early-stop when conclusive threshold is reached
  - Continue gathering when uncertainty remains high
  - Average computation saved is measurable
  - Fully deterministic: identical evidence order produces identical decisions
  - No external dependencies

Public API:
  - :func:`evaluate_evidence_progressively` — progressive evaluation entry point
  - :class:`ProgressiveEvaluation` — result with early-stop metadata
  - :class:`EvidenceCheckpoint` — snapshot at an evaluation checkpoint
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.report import Observation


# ---------------------------------------------------------------------------
# Thresholds (deterministic constants)
# ---------------------------------------------------------------------------

# Evidence is conclusive when agreement ratio exceeds this threshold
_CONCLUSIVE_AGREEMENT: float = 0.85

# Evidence is conclusive when the proportion of supporting vs conflicting
# evidence exceeds this ratio
_CONCLUSIVE_SUPPORT_RATIO: float = 0.80

# Early-stop after this many consecutive supporting evidence items
_CONSECUTIVE_SUPPORT_LIMIT: int = 3

# Minimum evidence items before early-stop is allowed
_MIN_EVIDENCE_FOR_EARLY_STOP: int = 3

# Maximum evidence items to evaluate (budget cap)
_MAX_EVALUATION_ITEMS: int = 20

# Confidence ceiling: stop gathering if cumulative confidence exceeds this
_CONFIDENCE_CEILING: float = 0.92


@dataclass(frozen=True)
class EvidenceCheckpoint:
    """Snapshot of evaluation state at a checkpoint.

    Captured after each evidence item is evaluated to enable
    progressive early-stop decisions.
    """

    items_evaluated: int
    supporting_count: int
    conflicting_count: int
    cumulative_confidence: float
    agreement_ratio: float
    support_ratio: float
    consecutive_support: int
    should_continue: bool
    early_stop_reason: str = ""


@dataclass
class ProgressiveEvaluation:
    """Result of progressive evidence evaluation.

    Contains the final observations, which evidence items were evaluated,
    evaluation checkpoints, and whether early-stop was triggered.
    """

    observations: list[Observation] = field(default_factory=list)
    evaluated_evidence: list[EvidenceItem] = field(default_factory=list)
    skipped_evidence: list[EvidenceItem] = field(default_factory=list)
    checkpoints: list[EvidenceCheckpoint] = field(default_factory=list)
    early_stopped: bool = False
    early_stop_checkpoint: EvidenceCheckpoint | None = None
    total_evidence_count: int = 0
    evaluated_count: int = 0
    skipped_count: int = 0

    @property
    def computation_saved_fraction(self) -> float:
        """Fraction of evidence evaluation skipped."""
        if self.total_evidence_count == 0:
            return 0.0
        return round(self.skipped_count / self.total_evidence_count, 4)

    @property
    def evaluation_efficiency(self) -> float:
        """Ratio of observations produced per evidence item evaluated."""
        if self.evaluated_count == 0:
            return 0.0
        return round(len(self.observations) / self.evaluated_count, 4)

    def to_dict(self) -> dict:
        """Serialize for benchmark reporting."""
        return {
            "total_evidence": self.total_evidence_count,
            "evaluated": self.evaluated_count,
            "skipped": self.skipped_count,
            "early_stopped": self.early_stopped,
            "computation_saved_fraction": self.computation_saved_fraction,
            "observation_count": len(self.observations),
            "evaluation_efficiency": self.evaluation_efficiency,
            "checkpoints": len(self.checkpoints),
        }


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------


def _classify_evidence_item(
    item: EvidenceItem,
    existing_observations: list[Observation],
) -> str:
    """Classify whether an evidence item supports or conflicts with existing observations.

    Returns 'supporting', 'conflicting', or 'neutral'.
    """
    if not existing_observations:
        return "neutral"

    domain_match = any(
        obs.dimension.lower() in item.domain.lower()
        or item.domain.lower() in obs.dimension.lower()
        for obs in existing_observations
    )

    if not domain_match:
        return "neutral"

    category_support = sum(
        1 for obs in existing_observations
        if obs.dimension.lower() in item.domain.lower()
        and obs.confidence > 0.5
    )

    if category_support > 0:
        return "supporting"

    category_conflict = sum(
        1 for obs in existing_observations
        if obs.dimension.lower() in item.domain.lower()
        and obs.confidence < 0.3
    )

    if category_conflict > 0:
        return "conflicting"

    return "neutral"


def _compute_checkpoint(
    evaluated: list[EvidenceItem],
    observations: list[Observation],
) -> EvidenceCheckpoint:
    """Compute a deterministic checkpoint from current state."""
    supporting = sum(
        1 for item in evaluated
        if _classify_evidence_item(item, observations) == "supporting"
    )
    conflicting = sum(
        1 for item in evaluated
        if _classify_evidence_item(item, observations) == "conflicting"
    )
    total = len(evaluated)
    agreement = (
        supporting / total if total > 0 else 0.0
    )
    support_ratio = (
        supporting / (supporting + conflicting)
        if (supporting + conflicting) > 0
        else 0.0
    )

    consecutive_support = _count_consecutive_support(evaluated, observations)
    cumulative_conf = _compute_cumulative_confidence(observations)

    should_continue = _should_continue_evaluation(
        total, agreement, support_ratio, consecutive_support, cumulative_conf,
    )
    early_stop_reason = ""
    if not should_continue:
        early_stop_reason = _early_stop_reason(
            total, agreement, support_ratio, consecutive_support, cumulative_conf,
        )

    return EvidenceCheckpoint(
        items_evaluated=total,
        supporting_count=supporting,
        conflicting_count=conflicting,
        cumulative_confidence=round(cumulative_conf, 4),
        agreement_ratio=round(agreement, 4),
        support_ratio=round(support_ratio, 4),
        consecutive_support=consecutive_support,
        should_continue=should_continue,
        early_stop_reason=early_stop_reason,
    )


def _count_consecutive_support(
    evaluated: list[EvidenceItem],
    observations: list[Observation],
) -> int:
    """Count consecutive supporting evidence from the end of the evaluated list."""
    count = 0
    for item in reversed(evaluated):
        if _classify_evidence_item(item, observations) == "supporting":
            count += 1
        else:
            break
    return count


def _compute_cumulative_confidence(observations: list[Observation]) -> float:
    """Compute cumulative confidence from observations."""
    if not observations:
        return 0.0
    return sum(o.confidence for o in observations) / len(observations)


def _should_continue_evaluation(
    total_evaluated: int,
    agreement: float,
    support_ratio: float,
    consecutive_support: int,
    cumulative_conf: float,
) -> bool:
    """Deterministic decision on whether to continue evidence evaluation.

    Returns False (stop) when evidence is conclusive.
    """
    if total_evaluated < _MIN_EVIDENCE_FOR_EARLY_STOP:
        return True

    if total_evaluated >= _MAX_EVALUATION_ITEMS:
        return False

    if cumulative_conf >= _CONFIDENCE_CEILING:
        return False

    if agreement >= _CONCLUSIVE_AGREEMENT and total_evaluated >= _MIN_EVIDENCE_FOR_EARLY_STOP:
        return False

    if support_ratio >= _CONCLUSIVE_SUPPORT_RATIO and total_evaluated >= _MIN_EVIDENCE_FOR_EARLY_STOP:
        return False

    if consecutive_support >= _CONSECUTIVE_SUPPORT_LIMIT:
        return False

    return True


def _early_stop_reason(
    total_evaluated: int,
    agreement: float,
    support_ratio: float,
    consecutive_support: int,
    cumulative_conf: float,
) -> str:
    """Build deterministic early-stop reason string."""
    if cumulative_conf >= _CONFIDENCE_CEILING:
        return f"Cumulative confidence ({cumulative_conf:.2f}) exceeded ceiling ({_CONFIDENCE_CEILING})"
    if agreement >= _CONCLUSIVE_AGREEMENT:
        return f"Evidence agreement ({agreement:.2f}) exceeded conclusive threshold ({_CONCLUSIVE_AGREEMENT})"
    if support_ratio >= _CONCLUSIVE_SUPPORT_RATIO:
        return f"Support ratio ({support_ratio:.2f}) exceeded threshold ({_CONCLUSIVE_SUPPORT_RATIO})"
    if consecutive_support >= _CONSECUTIVE_SUPPORT_LIMIT:
        return f"{consecutive_support} consecutive supporting evidence items"
    return f"Maximum evaluation items ({_MAX_EVALUATION_ITEMS}) reached"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_evidence_progressively(
    evidence: list[EvidenceItem],
    initial_observations: list[Observation],
    *,
    observation_factory: object | None = None,
) -> ProgressiveEvaluation:
    """Evaluate evidence progressively with early-stop capability.

    Processes evidence items one at a time, checking after each whether
    the evidence is conclusive enough to stop early.

    Parameters
    ----------
    evidence:
        Evidence items to evaluate (order matters for determinism).
    initial_observations:
        Observations from the reasoning layer (used as context).
    observation_factory:
        Optional callable that produces new observations from evidence.
        When None, only early-stop analysis is performed without
        generating new observations.

    Returns
    -------
    ProgressiveEvaluation
        Complete evaluation result with checkpoints and metadata.
    """
    result = ProgressiveEvaluation(
        total_evidence_count=len(evidence),
    )

    if not evidence:
        return result

    evaluated: list[EvidenceItem] = []
    observations = list(initial_observations)
    checkpoints: list[EvidenceCheckpoint] = []

    for item in evidence:
        if len(evaluated) >= _MAX_EVALUATION_ITEMS:
            result.skipped_evidence = evidence[len(evaluated):]
            break

        evaluated.append(item)
        result.evaluated_count = len(evaluated)

        if observation_factory is not None:
            try:
                new_obs = observation_factory(item, observations)
                if new_obs:
                    observations.extend(new_obs)
            except Exception:
                pass

        checkpoint = _compute_checkpoint(evaluated, observations)
        checkpoints.append(checkpoint)

        if not checkpoint.should_continue:
            result.early_stopped = True
            result.early_stop_checkpoint = checkpoint
            remaining = evidence[len(evaluated):]
            result.skipped_evidence = remaining
            result.skipped_count = len(remaining)
            break

    result.observations = observations
    result.evaluated_evidence = evaluated
    result.checkpoints = checkpoints

    if not result.early_stopped:
        result.skipped_evidence = []
        result.skipped_count = 0

    return result
