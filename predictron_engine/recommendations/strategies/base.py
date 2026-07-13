"""Base protocol and shared utilities for recommendation strategies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    DimensionAssessment,
    Observation,
    Recommendation,
)


@runtime_checkable
class DomainRecommendationStrategy(Protocol):
    """Protocol for domain-specific recommendation strategies.

    Each strategy owns exactly one decision domain (market, team,
    technology, etc.) and produces recommendations based on the
    analysis context relevant to that domain.
    """

    @property
    def domain(self) -> str:
        """The domain this strategy handles."""
        ...

    def generate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        assessments: list[DimensionAssessment],
    ) -> list[Recommendation]:
        """Generate recommendations for this domain."""
        ...


def filter_observations(
    observations: list[Observation], dimension: str
) -> list[Observation]:
    """Return observations matching the given dimension."""
    return [o for o in observations if o.dimension == dimension]


def filter_assessments(
    assessments: list[DimensionAssessment], dimension: str
) -> list[DimensionAssessment]:
    """Return assessments matching the given dimension."""
    return [a for a in assessments if a.dimension == dimension]


def observation_confidence(observations: list[Observation]) -> float:
    """Calculate average confidence from observations."""
    if not observations:
        return 0.0
    return sum(o.confidence for o in observations) / len(observations)


def assessment_confidence(assessments: list[DimensionAssessment]) -> float:
    """Calculate average confidence from assessments."""
    if not assessments:
        return 0.0
    return sum(a.confidence for a in assessments) / len(assessments)
