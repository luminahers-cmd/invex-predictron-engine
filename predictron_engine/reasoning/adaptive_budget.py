"""Adaptive Reasoning Budget — dynamic allocation of computational effort (Sprint 8).

Allocates deeper reasoning only when confidence is low or decision impact
is high. Skips unnecessary computation on straightforward cases while
remaining fully deterministic.

Key principles:
  - Budget allocation is deterministic given the same inputs
  - Simpler cases use less computation (early-stop)
  - Complex/uncertain cases receive deeper analysis
  - Every optimization is measurable via the budget report
  - No external dependencies

Public API:
  - :func:`compute_reasoning_budget` — compute adaptive budget for a run
  - :class:`ReasoningBudget` — deterministic budget allocation
  - :class:`BudgetReport` — measurable budget utilization report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures


# ---------------------------------------------------------------------------
# Budget thresholds and weights — all deterministic constants
# ---------------------------------------------------------------------------

# Data completeness thresholds
_HIGH_COMPLETENESS: float = 0.80
_LOW_COMPLETENESS: float = 0.40

# Evidence quality thresholds
_SUFFICIENT_EVIDENCE_COUNT: int = 3
_STRONG_EVIDENCE_THRESHOLD: float = 0.70

# Confidence thresholds
_HIGH_CONFIDENCE: float = 0.75
_LOW_CONFIDENCE: float = 0.35

# Budget tiers (fractions of max rules to run)
_BUDGET_FULL: float = 1.0
_BUDGET_ELEVATED: float = 0.85
_BUDGET_STANDARD: float = 0.70
_BUDGET_REDUCED: float = 0.50
_BUDGET_MINIMAL: float = 0.35

# Maximum number of rules that can be skipped
_MAX_SKIPPABLE_RULES: int = 4


@dataclass(frozen=True)
class ReasoningBudget:
    """Deterministic budget allocation for one reasoning pass.

    Parameters
    ----------
    budget_fraction:
        Fraction of full reasoning rules to execute (0.0-1.0).
    max_rules:
        Maximum number of rules that will be executed.
    skipped_rules:
        Names of rules that will be skipped.
    confidence_estimate:
        Pre-reasoning confidence estimate used for budget allocation.
    impact_score:
        Decision impact score (0.0-1.0) used for budget allocation.
    rationale:
        Deterministic explanation of why this budget was allocated.
    """

    budget_fraction: float = 1.0
    max_rules: int = 11
    skipped_rules: tuple[str, ...] = ()
    confidence_estimate: float = 0.5
    impact_score: float = 0.5
    rationale: str = ""

    @property
    def savings_fraction(self) -> float:
        """Fraction of computation saved relative to full budget."""
        return round(1.0 - self.budget_fraction, 4)

    @property
    def rules_executed(self) -> int:
        """Number of rules that will be executed."""
        return self.max_rules - len(self.skipped_rules)


@dataclass
class BudgetReport:
    """Measurable budget utilization report for one reasoning pass.

    Captures actual execution metrics so budget savings can be
    validated and benchmarked.
    """

    budget: ReasoningBudget = field(default_factory=ReasoningBudget)
    total_rules: int = 0
    rules_executed: int = 0
    rules_skipped: int = 0
    estimated_time_saved_ms: float = 0.0
    actual_time_ms: float = 0.0

    @property
    def actual_savings_fraction(self) -> float:
        """Actual fraction of rules skipped."""
        if self.total_rules == 0:
            return 0.0
        return round(self.rules_skipped / self.total_rules, 4)

    @property
    def budget_adherence(self) -> float:
        """How closely actual execution matched the budget plan (0-1)."""
        if self.total_rules == 0:
            return 1.0
        planned = self.budget.budget_fraction
        actual = self.rules_executed / self.total_rules
        return round(1.0 - abs(planned - actual), 4)

    def to_dict(self) -> dict[str, object]:
        """Serialize for benchmark reporting."""
        return {
            "budget_fraction": self.budget.budget_fraction,
            "savings_fraction": self.budget.savings_fraction,
            "actual_savings_fraction": self.actual_savings_fraction,
            "rules_total": self.total_rules,
            "rules_executed": self.rules_executed,
            "rules_skipped": self.rules_skipped,
            "estimated_time_saved_ms": self.estimated_time_saved_ms,
            "budget_adherence": self.budget_adherence,
            "rationale": self.budget.rationale,
        }


# ---------------------------------------------------------------------------
# Pre-reasoning heuristic estimators (deterministic, pure functions)
# ---------------------------------------------------------------------------


def _estimate_pre_reasoning_confidence(
    features: ExtractedFeatures,
    evidence: list[EvidenceItem],
) -> float:
    """Estimate confidence before reasoning runs, based on features and evidence.

    Uses data completeness, evidence volume, and evidence quality signals
    as proxies for expected reasoning confidence. This estimate is used to
    decide the reasoning budget — it is NOT the final confidence.
    """
    completeness = features.data_completeness
    evidence_count = len(evidence)

    if evidence_count >= _SUFFICIENT_EVIDENCE_COUNT:
        evidence_quality = min(evidence_count / 8.0, 1.0)
    elif evidence_count > 0:
        evidence_quality = evidence_count / _SUFFICIENT_EVIDENCE_COUNT * 0.6
    else:
        evidence_quality = 0.0

    feature_signals = 0.0
    total_checks = 0
    if features.industry:
        feature_signals += 1.0
    total_checks += 1
    if features.business_model:
        feature_signals += 1.0
    total_checks += 1
    if features.funding_stage:
        feature_signals += 1.0
    total_checks += 1
    if features.founder_profile_count > 0:
        feature_signals += 1.0
    total_checks += 1
    if features.technology_stack:
        feature_signals += 1.0
    total_checks += 1
    if features.has_revenue is not None:
        feature_signals += 1.0
    total_checks += 1

    feature_quality = feature_signals / total_checks if total_checks > 0 else 0.0

    confidence = (
        completeness * 0.35
        + evidence_quality * 0.35
        + feature_quality * 0.30
    )
    return round(max(0.0, min(1.0, confidence)), 4)


def _estimate_decision_impact(
    features: ExtractedFeatures,
    evidence: list[EvidenceItem],
) -> float:
    """Estimate how impactful the decision is, based on available signals.

    High impact cases: early stage, limited data, high-stakes industry.
    Low impact cases: mature companies with abundant data.
    """
    completeness = features.data_completeness
    early_stage = features.funding_stage in (
        "pre_seed", "idea", "pre-seed", "preseed", "seed",
    ) if features.funding_stage else False

    data_gap = 1.0 - completeness
    stage_factor = 0.7 if early_stage else 0.3

    evidence_scarcity = 0.0
    if len(evidence) < 2:
        evidence_scarcity = 0.8
    elif len(evidence) < 5:
        evidence_scarcity = 0.4

    impact = (
        data_gap * 0.4
        + stage_factor * 0.3
        + evidence_scarcity * 0.3
    )
    return round(max(0.0, min(1.0, impact)), 4)


# ---------------------------------------------------------------------------
# Budget computation
# ---------------------------------------------------------------------------

# Rule names in canonical order (must match DEFAULT_RULES order)
_RULE_NAMES: tuple[str, ...] = (
    "MarketContextRule",
    "BusinessModelContextRule",
    "StageExpectationRule",
    "TechnologyContextRule",
    "TeamAssessmentRule",
    "DataQualityRule",
    "RiskIndicatorRule",
    "CompetitionAssessmentRule",
    "QuantitativeSignalsRule",
    "CrossSignalReasoningRule",
    "QuantitativeCrossSignalRule",
)

# Rules that are most skippable when budget is tight (least critical)
_SKIPPABLE_PRIORITY: tuple[str, ...] = (
    "QuantitativeCrossSignalRule",
    "CrossSignalReasoningRule",
    "QuantitativeSignalsRule",
    "CompetitionAssessmentRule",
    "RiskIndicatorRule",
    "DataQualityRule",
    "StageExpectationRule",
    "BusinessModelContextRule",
    "TechnologyContextRule",
    "TeamAssessmentRule",
    "MarketContextRule",
)


def compute_reasoning_budget(
    features: ExtractedFeatures,
    evidence: list[EvidenceItem],
    *,
    total_rules: int | None = None,
) -> ReasoningBudget:
    """Compute adaptive reasoning budget for one analysis pass.

    The budget is deterministic: identical inputs always produce
    identical budget allocations.

    Parameters
    ----------
    features:
        Extracted features (used for pre-reasoning confidence estimate).
    evidence:
        Available evidence items (used for quality estimation).
    total_rules:
        Total number of rules available. Defaults to the canonical count.
    """
    count = total_rules or len(_RULE_NAMES)
    confidence = _estimate_pre_reasoning_confidence(features, evidence)
    impact = _estimate_decision_impact(features, evidence)

    budget_fraction = _compute_budget_fraction(confidence, impact)
    max_rules = max(1, round(count * budget_fraction))
    max_rules = min(max_rules, count)

    skipped_count = count - max_rules
    skipped_rules = _SKIPPABLE_PRIORITY[:skipped_count]

    rationale = _build_rationale(confidence, impact, budget_fraction, skipped_count)

    return ReasoningBudget(
        budget_fraction=round(budget_fraction, 4),
        max_rules=count,
        skipped_rules=tuple(skipped_rules),
        confidence_estimate=confidence,
        impact_score=impact,
        rationale=rationale,
    )


def _compute_budget_fraction(confidence: float, impact: float) -> float:
    """Deterministic budget fraction from confidence and impact.

    High confidence + low impact => minimal budget (skip many rules).
    Low confidence + high impact => full budget (run all rules).
    """
    uncertainty = 1.0 - confidence
    complexity = uncertainty * 0.6 + impact * 0.4

    if complexity >= 0.75:
        return _BUDGET_FULL
    elif complexity >= 0.55:
        return _BUDGET_ELEVATED
    elif complexity >= 0.35:
        return _BUDGET_STANDARD
    elif complexity >= 0.20:
        return _BUDGET_REDUCED
    else:
        return _BUDGET_MINIMAL


def _build_rationale(
    confidence: float,
    impact: float,
    budget_fraction: float,
    skipped_count: int,
) -> str:
    """Build a deterministic rationale string for the budget allocation."""
    if skipped_count == 0:
        return (
            f"Full reasoning budget (confidence={confidence:.2f}, "
            f"impact={impact:.2f}): all rules executed"
        )
    if confidence >= _HIGH_CONFIDENCE and impact < 0.3:
        return (
            f"Reduced budget (confidence={confidence:.2f} >= {_HIGH_CONFIDENCE}, "
            f"impact={impact:.2f} < 0.30): straightforward case, "
            f"skipping {skipped_count} low-priority rules"
        )
    if confidence < _LOW_CONFIDENCE:
        return (
            f"Elevated budget (confidence={confidence:.2f} < {_LOW_CONFIDENCE}): "
            f"uncertain case, running {skipped_count + 1}+ rules"
        )
    return (
        f"Adaptive budget (confidence={confidence:.2f}, impact={impact:.2f}, "
        f"fraction={budget_fraction:.2f}): {skipped_count} rule(s) skipped"
    )


def should_skip_rule(
    rule_name: str,
    budget: ReasoningBudget,
) -> bool:
    """Deterministically decide whether a rule should be skipped.

    A rule is skipped if it appears in the budget's skipped_rules tuple.
    """
    return rule_name in budget.skipped_rules
