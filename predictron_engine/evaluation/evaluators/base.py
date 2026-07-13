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

    return (
        f"Assessed {dimension} based on {obs_count} observations "
        f"and {evidence_count} evidence items "
        f"(average confidence: {avg_confidence:.2f})."
    )


def generate_rationale(
    dimension: str,
    observations: list[Observation],
    evidence: list[EvidenceItem],
) -> str:
    """Generate a detailed rationale for a dimension assessment."""
    if not observations:
        return f"Insufficient data to assess {dimension}."

    statements = [o.statement for o in observations[:3]]
    rationale_parts = [f"Key findings: {'; '.join(statements)}"]

    if evidence:
        evidence_categories = {e.category for e in evidence[:5]}
        rationale_parts.append(
            f"Supported by evidence categories: {', '.join(sorted(evidence_categories))}"
        )

    return ". ".join(rationale_parts) + "."
