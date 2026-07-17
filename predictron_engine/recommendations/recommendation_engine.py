"""Recommendation Engine — generates actionable investor recommendations.

The recommendation layer produces structured recommendations based on
features, observations, scores, and dimension assessments. Each recommendation
includes an actionable item, a category, a priority, and a rationale.

Key principles:
  - All recommendations must be actionable
  - Each recommendation must have a clear rationale
  - Strategies are injectable and independently testable
  - Duplicate recommendations are merged or eliminated
  - Recommendations are prioritized by importance and confidence
  - Default implementation produces placeholder recommendations
  - Real recommendation logic will be developed as separate strategies

Extensibility:
  - Implement RecommendationStrategy for different recommendation types
  - Inject strategy sets for different investor profiles
  - Strategies can be enabled/disabled per analysis context
"""

import logging
from typing import Protocol, runtime_checkable

from predictron_engine.evaluation.evaluation_models import DimensionAssessment
from predictron_engine.knowledge.concepts import Priority, RecommendationCategory
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import Observation, Recommendation, ScoreResult

logger = logging.getLogger(__name__)

_PRIORITY_ORDER: dict[str, int] = {
    Priority.HIGH.value: 3,
    Priority.MEDIUM.value: 2,
    Priority.LOW.value: 1,
}


@runtime_checkable
class RecommendationStrategy(Protocol):
    """Protocol for individual recommendation strategies.

    Each strategy evaluates the full analysis context and produces
    zero or more recommendations.
    """

    def generate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        scores: list[ScoreResult],
        assessments: list[DimensionAssessment] | None = None,
    ) -> list[Recommendation]:
        """Generate recommendations based on the analysis context."""
        ...


class DefaultRecommendationStrategy:
    """Placeholder strategy that produces basic recommendations.

    This strategy exists as a template for future recommendation logic.
    It produces recommendations based on data completeness signals.
    """

    def generate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        scores: list[ScoreResult],
        assessments: list[DimensionAssessment] | None = None,
    ) -> list[Recommendation]:
        recommendations: list[Recommendation] = []

        if not features.has_pitch_deck:
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.DUE_DILIGENCE.value,
                    action="Request the pitch deck for deeper analysis.",
                    priority=Priority.HIGH.value,
                    rationale="No pitch deck was provided, limiting visual analysis.",
                )
            )

        if features.founder_profile_count == 0:
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.FOLLOW_UP.value,
                    action="Request founder LinkedIn profiles for team assessment.",
                    priority=Priority.MEDIUM.value,
                    rationale="No founder profiles were provided for team evaluation.",
                )
            )

        if not recommendations:
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.FOLLOW_UP.value,
                    action="Conduct full due diligence review.",
                    priority=Priority.MEDIUM.value,
                    rationale="Baseline data is sufficient for preliminary review.",
                )
            )

        return recommendations


def _build_default_strategies() -> list[RecommendationStrategy]:
    """Create the default set of recommendation strategies."""
    return [DefaultRecommendationStrategy()]


def _deduplicate_recommendations(
    recommendations: list[Recommendation],
) -> list[Recommendation]:
    """Remove duplicate or near-duplicate recommendations.

    When multiple strategies produce similar recommendations,
    this keeps the one with the highest priority and confidence.
    """
    if not recommendations:
        return []

    seen: dict[str, Recommendation] = {}
    for rec in recommendations:
        key = _recommendation_fingerprint(rec)
        if key in seen:
            existing = seen[key]
            if _recommendation_rank(rec) > _recommendation_rank(existing):
                seen[key] = rec
        else:
            seen[key] = rec

    return list(seen.values())


def _recommendation_fingerprint(rec: Recommendation) -> str:
    """Create a deduplication key for a recommendation.

    Uses the action text (lowercased, stripped) as the primary key
    since the action is the most semantically meaningful field.
    """
    return rec.action.strip().lower()


def _recommendation_rank(rec: Recommendation) -> int:
    """Compute a numeric rank for prioritization.

    Higher rank means higher priority.
    """
    priority_rank = _PRIORITY_ORDER.get(rec.priority, 0)
    confidence_bonus = 1 if rec.confidence > 0.5 else 0
    return priority_rank * 10 + confidence_bonus


def _prioritize_recommendations(
    recommendations: list[Recommendation],
) -> list[Recommendation]:
    """Sort recommendations by priority and confidence.

    High-priority, high-confidence recommendations appear first.
    """
    return sorted(
        recommendations,
        key=lambda r: (_recommendation_rank(r), r.confidence),
        reverse=True,
    )


class DefaultRecommendationEngine:
    """Standard implementation of the RecommendationEngine protocol.

    Delegates recommendation generation to a set of injectable
    RecommendationStrategy components. Deduplicates and prioritizes
    the combined output.
    """

    def __init__(
        self, strategies: list[RecommendationStrategy] | None = None
    ) -> None:
        self._strategies = (
            strategies if strategies is not None else _build_default_strategies()
        )

    def recommend(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        scores: list[ScoreResult],
        assessments: list[DimensionAssessment] | None = None,
    ) -> list[Recommendation]:
        """Generate recommendations from all active strategies."""
        logger.info(
            "Running recommendation engine with %d strategies",
            len(self._strategies),
        )

        recommendations: list[Recommendation] = []
        for strategy in self._strategies:
            try:
                recommendations.extend(
                    strategy.generate(features, observations, scores, assessments)
                )
            except TypeError:
                try:
                    recommendations.extend(
                        strategy.generate(features, observations, scores)
                    )
                except Exception:
                    logger.warning(
                        "Strategy %s failed, skipping", type(strategy).__name__
                    )
            except Exception:
                logger.warning(
                    "Strategy %s failed, skipping", type(strategy).__name__
                )

        deduplicated = _deduplicate_recommendations(recommendations)
        prioritized = _prioritize_recommendations(deduplicated)

        logger.info(
            "Generated %d recommendations (%d after dedup)",
            len(recommendations),
            len(prioritized),
        )
        return prioritized
