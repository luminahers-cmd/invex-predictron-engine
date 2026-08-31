"""Market recommendation strategy.

Generates recommendations based on market opportunity observations
and assessments. Focuses on market validation, competitive landscape,
and market entry considerations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from predictron_engine.knowledge.concepts import Priority, RecommendationCategory
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    DimensionAssessment,
    Observation,
    Recommendation,
)
from predictron_engine.recommendations.strategies.base import (
    assessment_confidence,
    filter_assessments,
    filter_observations,
    observation_confidence,
)

if TYPE_CHECKING:
    from predictron_engine.models.report import InvestmentReadiness, ScoreResult


class MarketStrategy:
    """Generates market-related recommendations.

    Focuses on market opportunity validation, competitive landscape
    analysis, and market entry considerations.
    """

    @property
    def domain(self) -> str:
        return "market_opportunity"

    def generate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        assessments: list[DimensionAssessment],
        scores: list[ScoreResult] | None = None,
        readiness: InvestmentReadiness | None = None,
    ) -> list[Recommendation]:
        recommendations: list[Recommendation] = []
        market_obs = filter_observations(observations, self.domain)
        market_assess = filter_assessments(assessments, self.domain)

        conf = observation_confidence(market_obs)
        assess_conf = assessment_confidence(market_assess)

        recommendations.extend(self._recommend_industry(features, conf))
        recommendations.extend(self._recommend_geography(features, conf))
        recommendations.extend(
            self._recommend_market_analysis(market_obs, market_assess, conf, assess_conf)
        )
        recommendations.extend(self._recommend_network_effects(features))
        recommendations.extend(self._recommend_fragmentation(features))

        return recommendations

    @staticmethod
    def _recommend_industry(features: ExtractedFeatures, conf: float) -> list[Recommendation]:
        if features.industry is not None:
            return []
        return [
            Recommendation(
                category=RecommendationCategory.DUE_DILIGENCE.value,
                action="Research and confirm the target industry classification.",
                priority=Priority.HIGH.value,
                rationale="Industry could not be determined from available data.",
                title="Confirm Industry Classification",
                description=(
                    "The industry classification was not determined. "
                    "Manual verification is needed to ensure accurate "
                    "market analysis."
                ),
                confidence=max(conf, 0.3),
                expected_impact="Enables accurate market sizing and competitive analysis.",
                action_items=[
                    "Review the company website and product materials",
                    "Identify the primary industry vertical",
                    "Validate against standard industry taxonomies",
                ],
                metadata={"strategy": "market", "missing_field": "industry"},
            )
        ]

    @staticmethod
    def _recommend_geography(features: ExtractedFeatures, conf: float) -> list[Recommendation]:
        if features.geography is not None or features.headquarters_region is not None:
            return []
        return [
            Recommendation(
                category=RecommendationCategory.FOLLOW_UP.value,
                action="Request geographic market information.",
                priority=Priority.MEDIUM.value,
                rationale="No geographic data was available for market analysis.",
                title="Clarify Geographic Focus",
                description=(
                    "Geographic market information was not provided. "
                    "Understanding the target market geography is essential "
                    "for assessing market size and regulatory considerations."
                ),
                confidence=max(conf, 0.3),
                expected_impact="Enables geographic market sizing and regulatory assessment.",
                action_items=[
                    "Ask about headquarters location",
                    "Identify primary target markets",
                    "Assess geographic expansion plans",
                ],
                metadata={"strategy": "market", "missing_field": "geography"},
            )
        ]

    @staticmethod
    def _recommend_market_analysis(
        market_obs: list[Observation],
        market_assess: list[DimensionAssessment],
        conf: float,
        assess_conf: float,
    ) -> list[Recommendation]:
        if not market_obs:
            return []
        return [
            Recommendation(
                category=RecommendationCategory.OPPORTUNITY.value,
                action="Conduct detailed market sizing analysis.",
                priority=Priority.MEDIUM.value,
                rationale=(
                    f"Market observations indicate potential with "
                    f"{len(market_obs)} data points and {conf:.0%} confidence."
                ),
                title="Deep-Dive Market Analysis",
                description=(
                    "Available market data suggests opportunity. A detailed "
                    "market sizing exercise would quantify the addressable "
                    "market and growth trajectory."
                ),
                supporting_observations=market_obs,
                supporting_assessments=market_assess,
                confidence=(conf + assess_conf) / 2 if assess_conf else conf,
                expected_impact="Quantifies total addressable market and growth potential.",
                action_items=[
                    "Request TAM/SAM/SOM estimates from the team",
                    "Validate market size with third-party research",
                    "Assess market growth rate and trends",
                ],
                metadata={"strategy": "market", "observation_count": len(market_obs)},
            )
        ]

    @staticmethod
    def _recommend_network_effects(features: ExtractedFeatures) -> list[Recommendation]:
        if features.network_effect_competition != "strong_network_effects":
            return []
        return [
            Recommendation(
                category=RecommendationCategory.OPPORTUNITY.value,
                action="Leverage network effects for competitive advantage.",
                priority=Priority.MEDIUM.value,
                rationale=(
                    "Strong network effects detected. First-mover or "
                    "fast-follower strategies can create winner-take-most "
                    "dynamics."
                ),
                title="Network Effects Strategy",
                description=(
                    "The market exhibits strong network effects. "
                    "Develop a strategy to rapidly build critical mass "
                    "and establish defensible network advantages."
                ),
                confidence=0.65,
                expected_impact="Enables winner-take-most positioning through network effects.",
                action_items=[
                    "Design user acquisition strategy for rapid scale",
                    "Identify network effect triggers in the product",
                    "Plan for multi-sided marketplace dynamics if applicable",
                ],
                metadata={
                    "strategy": "market",
                    "network_effect": features.network_effect_competition,
                },
            )
        ]

    @staticmethod
    def _recommend_fragmentation(features: ExtractedFeatures) -> list[Recommendation]:
        if features.market_concentration != "fragmented":
            return []
        return [
            Recommendation(
                category=RecommendationCategory.OPPORTUNITY.value,
                action="Evaluate consolidation and differentiation opportunities.",
                priority=Priority.LOW.value,
                rationale=(
                    "Fragmented market detected. Consolidation strategies "
                    "or strong differentiation can capture market share."
                ),
                title="Fragmented Market Opportunity",
                description=(
                    "The market is fragmented with many participants. "
                    "Consider consolidation plays or differentiated "
                    "positioning to capture outsized market share."
                ),
                confidence=0.55,
                expected_impact=(
                    "Identifies consolidation or differentiation strategies in fragmented markets."
                ),
                action_items=[
                    "Map competitive landscape and key players",
                    "Identify consolidation or partnership opportunities",
                    "Assess differentiation requirements",
                ],
                metadata={
                    "strategy": "market",
                    "market_concentration": features.market_concentration,
                },
            )
        ]
