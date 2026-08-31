"""Composite recommendation engine — orchestrates domain strategies.

This module provides the CompositeRecommendationEngine, which orchestrates
all domain-specific recommendation strategies using protocol-based
dependency injection.

Recommendations are ordered by a deterministic, gap-aware rank so the
output prioritizes the largest weighted readiness gaps, the weakest
readiness contributors, and the highest-impact improvements (Sprint P8C,
audit H4). The rank reuses existing pipeline outputs only — the weighted
readiness dimension contributions and the per-dimension score results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from predictron_engine.evaluation.investment_readiness import (
    weighted_gap_magnitude,
)
from predictron_engine.knowledge.concepts import AnalysisDimension

if TYPE_CHECKING:
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import (
        DimensionAssessment,
        InvestmentReadiness,
        Observation,
        Recommendation,
        ScoreResult,
    )
    from predictron_engine.recommendations.strategies.base import (
        DomainRecommendationStrategy,
    )

logger = logging.getLogger(__name__)

# Analysis dimensions are the canonical readiness dimensions.
_KNOWN_DIMENSIONS: frozenset[str] = frozenset(dim.value for dim in AnalysisDimension)

_PRIORITY_RANK: dict[str, float] = {
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
}


class CompositeRecommendationEngine:
    """Orchestrates all domain-specific recommendation strategies.

    This class composes independent recommendation strategies into a
    cohesive recommendation flow. All dependencies are injected via
    the constructor.
    """

    def __init__(
        self,
        strategies: list[DomainRecommendationStrategy] | None = None,
    ) -> None:
        if strategies is None:
            strategies = self._create_default_strategies()
        self._strategies = strategies
        logger.info(
            "CompositeRecommendationEngine initialized with %d strategies",
            len(self._strategies),
        )

    def recommend(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        scores: list[ScoreResult],
        assessments: list[DimensionAssessment] | None = None,
        readiness: InvestmentReadiness | None = None,
    ) -> list[Recommendation]:
        """Generate recommendations from all active domain strategies.

        ``scores`` and ``readiness`` are the already-produced pipeline
        outputs (Sprint P8C, H4). They are exposed to each strategy and
        used to rank the combined output by weighted readiness gap.
        """
        logger.info(
            "Running CompositeRecommendationEngine with %d strategies",
            len(self._strategies),
        )

        collected: list[tuple[str, Recommendation]] = []
        for strategy in self._strategies:
            generated = self._call_strategy(strategy, features, observations, scores, assessments)
            domain = getattr(strategy, "domain", "")
            for rec in generated:
                collected.append((domain, rec))

        logger.info(
            "Generated %d total recommendations",
            len(collected),
        )
        return self._order_gap_aware(collected, scores, readiness)

    @staticmethod
    def _call_strategy(
        strategy: DomainRecommendationStrategy,
        features: ExtractedFeatures,
        observations: list[Observation],
        scores: list[ScoreResult],
        assessments: list[DimensionAssessment] | None,
    ) -> list[Recommendation]:
        """Invoke a strategy, falling back to the legacy 3-arg signature.

        Strategies built before Sprint P8C do not accept the optional
        ``scores`` / ``readiness`` keyword arguments; ``TypeError`` from an
        unexpected keyword argument triggers the legacy call so no
        customized strategy is dropped.
        """
        try:
            return strategy.generate(
                features,
                observations,
                assessments or [],
                scores=scores,
                readiness=None,
            )
        except TypeError:
            pass
        except Exception:
            logger.warning(
                "Strategy %s failed, skipping",
                getattr(strategy, "domain", type(strategy).__name__),
                exc_info=True,
            )
            return []
        try:
            return strategy.generate(features, observations, assessments or [])
        except Exception:
            logger.warning(
                "Strategy %s failed, skipping",
                getattr(strategy, "domain", type(strategy).__name__),
                exc_info=True,
            )
            return []

    @staticmethod
    def _order_gap_aware(
        collected: list[tuple[str, Recommendation]],
        scores: list[ScoreResult],
        readiness: InvestmentReadiness | None,
    ) -> list[Recommendation]:
        """Order recommendations by weighted readiness gap magnitude.

        The primary sort key is the largest normalized weighted readiness
        gap across a recommendation's analysis dimensions; the static
        priority label is demoted to a final tie-break so gaps dominate
        (Sprint P8C, audit H4). The sort is stable and deterministic.
        """
        ranked = sorted(
            collected,
            key=lambda item: _gap_aware_rank(item[0], item[1], scores, readiness),
            reverse=True,
        )
        return [rec for _, rec in ranked]

    @staticmethod
    def _create_default_strategies() -> list[DomainRecommendationStrategy]:
        """Create the default set of domain strategies."""
        from predictron_engine.recommendations.strategies.business_model import (
            BusinessModelStrategy,
        )
        from predictron_engine.recommendations.strategies.fundraising import (
            FundraisingStrategy,
        )
        from predictron_engine.recommendations.strategies.market import (
            MarketStrategy,
        )
        from predictron_engine.recommendations.strategies.risk import RiskStrategy
        from predictron_engine.recommendations.strategies.team import TeamStrategy
        from predictron_engine.recommendations.strategies.technology import (
            TechnologyStrategy,
        )
        from predictron_engine.recommendations.strategies.traction import (
            TractionStrategy,
        )

        return [
            MarketStrategy(),
            TeamStrategy(),
            TechnologyStrategy(),
            BusinessModelStrategy(),
            TractionStrategy(),
            RiskStrategy(),
            FundraisingStrategy(),
        ]


def _recommendation_dimensions(
    strategy_domain: str,
    rec: Recommendation,
) -> list[str]:
    """Return the analysis dimensions a recommendation is tied to.

    Reuses the producing strategy's domain (when it is a canonical
    readiness dimension) plus the dimensions of any attached
    assessments, so no new dimension model is introduced.
    """
    dimensions: list[str] = []
    if strategy_domain in _KNOWN_DIMENSIONS:
        dimensions.append(strategy_domain)
    for assessment in rec.supporting_assessments:
        if assessment.dimension not in dimensions:
            dimensions.append(assessment.dimension)
    return dimensions


def _gap_aware_rank(
    strategy_domain: str,
    rec: Recommendation,
    scores: list[ScoreResult],
    readiness: InvestmentReadiness | None,
) -> tuple[float, float, float, float]:
    """Deterministic descending sort key for gap-aware ordering.

    Key components, in priority order:
      1. Largest normalized weighted readiness gap across the tied
         dimensions (reused from readiness / scoring outputs).
      2. Recommendation confidence.
      3. Calibrated expected confidence.
      4. Static priority label (demoted so gaps dominate).
    """
    dimensions = _recommendation_dimensions(strategy_domain, rec)
    if dimensions:
        gap = max(weighted_gap_magnitude(dim, scores, readiness) for dim in dimensions)
    elif readiness is not None:
        gap = max(0.0, (100.0 - readiness.readiness_score) / 100.0)
    else:
        gap = 0.0
    priority = _PRIORITY_RANK.get(rec.priority.strip().lower(), 0.0)
    return (gap, rec.confidence, rec.expected_confidence or 0.0, priority)
