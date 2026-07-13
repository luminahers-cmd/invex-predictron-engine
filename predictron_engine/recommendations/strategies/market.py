"""Market recommendation strategy.

Generates recommendations based on market opportunity observations
and assessments. Focuses on market validation, competitive landscape,
and market entry considerations.
"""

from __future__ import annotations

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
    ) -> list[Recommendation]:
        recommendations: list[Recommendation] = []
        market_obs = filter_observations(observations, self.domain)
        market_assess = filter_assessments(assessments, self.domain)

        conf = observation_confidence(market_obs)
        assess_conf = assessment_confidence(market_assess)

        if features.industry is None:
            recommendations.append(
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
            )

        if features.geography is None and features.headquarters_region is None:
            recommendations.append(
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
            )

        if market_obs and not recommendations:
            recommendations.append(
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
            )

        return recommendations
