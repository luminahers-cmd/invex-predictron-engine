"""Base protocol and shared utilities for recommendation strategies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    DimensionAssessment,
    InvestmentReadiness,
    Observation,
    Recommendation,
    ScoreResult,
)


@runtime_checkable
class DomainRecommendationStrategy(Protocol):
    """Protocol for domain-specific recommendation strategies.

    Each strategy owns exactly one decision domain (market, team,
    technology, etc.) and produces recommendations based on the
    analysis context relevant to that domain.

    ``scores`` (ScoreResult collection) and ``readiness``
    (InvestmentReadiness) are optional, deterministic pipeline outputs
    that precede recommendation generation. They are exposed so
    strategies can prioritize the largest weighted gaps, the weakest
    readiness contributors, and the highest-impact improvements. Both
    default to None so existing strategies and callers stay compatible.
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
        scores: list[ScoreResult] | None = None,
        readiness: InvestmentReadiness | None = None,
    ) -> list[Recommendation]:
        """Generate recommendations for this domain."""
        ...


def filter_observations(observations: list[Observation], dimension: str) -> list[Observation]:
    """Return observations matching the given dimension."""
    return [o for o in observations if o.dimension == dimension]


def filter_assessments(
    assessments: list[DimensionAssessment], dimension: str
) -> list[DimensionAssessment]:
    """Return assessments matching the given dimension."""
    return [a for a in assessments if a.dimension == dimension]


def filter_scores(scores: list[ScoreResult], dimension: str) -> ScoreResult | None:
    """Return the first ScoreResult matching the given dimension.

    Returns None when no score exists for the dimension (or when the
    scores collection is empty), preserving downstream decision logic.
    """
    for result in scores:
        if result.dimension == dimension:
            return result
    return None


def dimension_score(scores: list[ScoreResult], dimension: str) -> float | None:
    """Return the score value for a dimension, or None when absent."""
    result = filter_scores(scores, dimension)
    return None if result is None else result.score


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
