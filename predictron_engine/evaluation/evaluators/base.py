"""Base evaluator protocol and shared utilities.

This module defines the DimensionEvaluator protocol and provides
shared utilities for all evaluator implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from predictron_engine.evaluation.evaluation_models import DimensionAssessment
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation


@runtime_checkable
class DimensionEvaluator(Protocol):
    """Protocol for dimension-specific evaluators.

    Each evaluator translates observations and evidence for a specific
    analysis dimension into a structured DimensionAssessment.
    """

    @property
    def dimension(self) -> str:
        """The analysis dimension this evaluator handles."""
        ...

    def evaluate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        evidence: list[EvidenceItem],
    ) -> DimensionAssessment:
        """Evaluate and return a structured assessment for this dimension."""
        ...


def filter_observations(
    observations: list[Observation],
    dimension: str,
) -> list[Observation]:
    """Return observations matching the given dimension."""
    return [o for o in observations if o.dimension == dimension]


def filter_evidence_by_domain(
    evidence: list[EvidenceItem],
    domain: str,
) -> list[EvidenceItem]:
    """Return evidence items matching the given domain."""
    return [e for e in evidence if e.domain == domain]


def calculate_average_confidence(observations: list[Observation]) -> float:
    """Calculate average confidence from a list of observations."""
    if not observations:
        return 0.0
    total = sum(o.confidence for o in observations)
    return total / len(observations)


def calculate_weighted_importance(observations: list[Observation]) -> float:
    """Calculate importance-weighted average confidence from observations.

    Higher-importance observations contribute more to the result,
    producing a signal-weighted confidence rather than a simple average.
    """
    if not observations:
        return 0.0
    total_weight = sum(o.importance for o in observations)
    if total_weight == 0.0:
        return calculate_average_confidence(observations)
    weighted = sum(o.confidence * o.importance for o in observations)
    return weighted / total_weight


def count_category_coverage(observations: list[Observation]) -> int:
    """Count the number of distinct observation categories present."""
    return len({o.category for o in observations})


def identify_strengths(observations: list[Observation]) -> list[str]:
    """Extract strength statements from high-confidence observations."""
    strengths: list[str] = []
    for obs in sorted(observations, key=lambda o: o.confidence * o.importance, reverse=True):
        if obs.confidence >= 0.6 and obs.importance >= 0.5:
            strengths.append(obs.statement)
    return strengths[:3]


def identify_concerns(observations: list[Observation]) -> list[str]:
    """Extract concern statements from observations with risk/conflict signals."""
    concerns: list[str] = []
    for obs in observations:
        if obs.category in ("risk_indicator", "signal_conflict", "entry_difficulty"):
            concerns.append(obs.statement)
    for obs in sorted(observations, key=lambda o: o.confidence * o.importance, reverse=True):
        if obs.confidence >= 0.6 and obs.importance >= 0.6:
            risk_keywords = (
                "limited", "weak", "risk", "conflict", "concern", "no ",
            )
            if any(kw in obs.statement.lower() for kw in risk_keywords):
                if obs.statement not in concerns:
                    concerns.append(obs.statement)
    return concerns[:3]


def generate_summary(
    dimension: str,
    observations: list[Observation],
    evidence: list[EvidenceItem],
) -> str:
    """Generate a high-level summary for a dimension assessment."""
    if not observations:
        return f"No observations available for {dimension}."

    obs_count = len(observations)
    evidence_count = len(evidence)
    avg_confidence = calculate_average_confidence(observations)
    category_count = count_category_coverage(observations)

    strengths = identify_strengths(observations)
    concerns = identify_concerns(observations)

    parts = [
        f"Assessed {dimension} based on {obs_count} observations "
        f"({category_count} categories) and {evidence_count} evidence items "
        f"(average confidence: {avg_confidence:.2f})."
    ]

    if strengths:
        parts.append(f"Strengths: {'; '.join(strengths[:2])}.")
    if concerns:
        parts.append(f"Concerns: {'; '.join(concerns[:2])}.")

    return " ".join(parts)


def generate_rationale(
    dimension: str,
    observations: list[Observation],
    evidence: list[EvidenceItem],
) -> str:
    """Generate a detailed rationale for a dimension assessment."""
    if not observations:
        return f"Insufficient data to assess {dimension}."

    ranked = sorted(observations, key=lambda o: o.confidence * o.importance, reverse=True)
    top_statements = [o.statement for o in ranked[:3]]
    rationale_parts = [f"Key findings: {'; '.join(top_statements)}"]

    if evidence:
        evidence_categories = {e.category for e in evidence[:5]}
        rationale_parts.append(
            f"Supported by evidence categories: {', '.join(sorted(evidence_categories))}"
        )

    importance_range = [o.importance for o in observations]
    avg_importance = sum(importance_range) / len(importance_range)
    high_importance = sum(1 for i in importance_range if i >= 0.7)
    rationale_parts.append(
        f"Observation importance: avg {avg_importance:.2f}, "
        f"{high_importance} high-importance signals"
    )

    return ". ".join(rationale_parts) + "."


def generate_cross_signal_context(
    observations: list[Observation],
    target_dimension: str,
) -> str:
    """Extract cross-signal observations relevant to a given dimension.

    Identifies investment_thesis observations that reference the target
    dimension or contain relevant reinforcement/conflict signals.
    """
    cross_obs = [
        o for o in observations
        if o.dimension == "investment_thesis"
    ]
    if not cross_obs:
        return ""

    relevant: list[str] = []
    for obs in cross_obs:
        category = obs.category
        if "fit" in category or "alignment" in category:
            relevant.append(obs.statement)
        elif "conflict" in category:
            relevant.append(obs.statement)
        elif "consistency" in category and target_dimension in (
            "traction_signals", "business_model_viability",
        ):
            relevant.append(obs.statement)

    if not relevant:
        return ""
    return " Cross-signal context: " + " ".join(relevant[:2])
