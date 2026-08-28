"""Composite recommendation engine — orchestrates domain strategies.

This module provides the CompositeRecommendationEngine, which orchestrates
all domain-specific recommendation strategies using protocol-based
dependency injection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import (
        DimensionAssessment,
        Observation,
        Recommendation,
        ScoreResult,
    )
    from predictron_engine.recommendations.strategies.base import (
        DomainRecommendationStrategy,
    )

logger = logging.getLogger(__name__)


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
    ) -> list[Recommendation]:
        """Generate recommendations from all active domain strategies."""
        logger.info(
            "Running CompositeRecommendationEngine with %d strategies",
            len(self._strategies),
        )

        recommendations: list[Recommendation] = []
        for strategy in self._strategies:
            try:
                recommendations.extend(
                    strategy.generate(features, observations, assessments or [])
                )
                logger.debug(
                    "Strategy %s generated %d recommendations",
                    strategy.domain,
                    len(recommendations),
                )
            except Exception:
                logger.warning(
                    "Strategy %s failed, skipping", strategy.domain, exc_info=True
                )

        logger.info("Generated %d total recommendations", len(recommendations))
        return recommendations

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
