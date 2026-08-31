"""Technology recommendation strategy.

Generates recommendations based on technology and product observations
and assessments. Focuses on technology stack, product strength,
and technical differentiation.
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


class TechnologyStrategy:
    """Generates technology-related recommendations.

    Focuses on technology stack assessment, product strength
    evaluation, and technical differentiation analysis.
    """

    @property
    def domain(self) -> str:
        return "product_strength"

    def generate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        assessments: list[DimensionAssessment],
        scores: list[ScoreResult] | None = None,
        readiness: InvestmentReadiness | None = None,
    ) -> list[Recommendation]:
        recommendations: list[Recommendation] = []
        tech_obs = filter_observations(observations, self.domain)
        tech_assess = filter_assessments(assessments, self.domain)

        conf = observation_confidence(tech_obs)
        assess_conf = assessment_confidence(tech_assess)

        if not features.technology_stack:
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.DUE_DILIGENCE.value,
                    action="Request technology stack details.",
                    priority=Priority.MEDIUM.value,
                    rationale="No technology stack information was identified.",
                    title="Identify Technology Stack",
                    description=(
                        "The technology stack could not be determined from "
                        "available data. Understanding the technical foundation "
                        "is important for assessing product capability and "
                        "technical risk."
                    ),
                    confidence=0.4,
                    expected_impact="Enables technical capability and risk assessment.",
                    action_items=[
                        "Ask about core technologies and frameworks",
                        "Inquire about infrastructure and hosting",
                        "Identify key technical dependencies",
                    ],
                    metadata={"strategy": "technology", "missing_field": "technology_stack"},
                )
            )

        if not features.has_pitch_deck:
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.DUE_DILIGENCE.value,
                    action="Request the pitch deck for product analysis.",
                    priority=Priority.HIGH.value,
                    rationale="No pitch deck was provided, limiting product analysis.",
                    title="Obtain Pitch Deck",
                    description=(
                        "A pitch deck typically contains product screenshots, "
                        "architecture diagrams, and technical differentiation "
                        "details that are essential for product assessment."
                    ),
                    confidence=0.6,
                    expected_impact="Enables visual product assessment and architecture review.",
                    action_items=[
                        "Request the latest pitch deck",
                        "Ask for product demo or screenshots",
                        "Review any technical documentation",
                    ],
                    metadata={"strategy": "technology", "missing_field": "pitch_deck"},
                )
            )

        if tech_obs and not recommendations:
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.OPPORTUNITY.value,
                    action="Evaluate technical differentiation and moat.",
                    priority=Priority.MEDIUM.value,
                    rationale=(
                        f"Technology observations indicate potential with "
                        f"{len(tech_obs)} data points and {conf:.0%} confidence."
                    ),
                    title="Assess Technical Differentiation",
                    description=(
                        "Available technology data suggests capability. A deeper "
                        "assessment of technical differentiation and defensibility "
                        "is recommended."
                    ),
                    supporting_observations=tech_obs,
                    supporting_assessments=tech_assess,
                    confidence=(conf + assess_conf) / 2 if assess_conf else conf,
                    expected_impact="Identifies technical moat and competitive advantages.",
                    action_items=[
                        "Review technical architecture",
                        "Assess IP or proprietary technology",
                        "Evaluate scalability characteristics",
                    ],
                    metadata={"strategy": "technology", "observation_count": len(tech_obs)},
                )
            )

        return recommendations
