"""Business model recommendation strategy.

Generates recommendations based on business model observations
and assessments. Focuses on revenue model, unit economics,
and business model viability.
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


class BusinessModelStrategy:
    """Generates business model recommendations.

    Focuses on revenue model validation, unit economics assessment,
    and business model viability analysis.
    """

    @property
    def domain(self) -> str:
        return "business_model_viability"

    def generate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        assessments: list[DimensionAssessment],
        scores: list[ScoreResult] | None = None,
        readiness: InvestmentReadiness | None = None,
    ) -> list[Recommendation]:
        recommendations: list[Recommendation] = []
        bm_obs = filter_observations(observations, self.domain)
        bm_assess = filter_assessments(assessments, self.domain)

        conf = observation_confidence(bm_obs)
        assess_conf = assessment_confidence(bm_assess)

        if features.business_model is None:
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.DUE_DILIGENCE.value,
                    action="Identify and validate the business model.",
                    priority=Priority.HIGH.value,
                    rationale="Business model could not be determined from available data.",
                    title="Clarify Business Model",
                    description=(
                        "The business model was not determined. Understanding "
                        "how the company generates revenue is fundamental to "
                        "assessing viability and scalability."
                    ),
                    confidence=0.4,
                    expected_impact="Enables revenue model and unit economics assessment.",
                    action_items=[
                        "Ask about primary revenue streams",
                        "Understand pricing model and structure",
                        "Identify recurring vs. one-time revenue",
                    ],
                    metadata={"strategy": "business_model", "missing_field": "business_model"},
                )
            )

        if features.has_revenue is not True:
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.FOLLOW_UP.value,
                    action="Request revenue and traction metrics.",
                    priority=Priority.MEDIUM.value,
                    rationale="Revenue status could not be confirmed.",
                    title="Validate Revenue Status",
                    description=(
                        "Revenue generation was not confirmed from available "
                        "data. Understanding current revenue status is important "
                        "for assessing business model maturity."
                    ),
                    confidence=0.3,
                    expected_impact="Clarifies business model maturity stage.",
                    action_items=[
                        "Ask about current MRR/ARR",
                        "Inquire about revenue growth rate",
                        "Understand customer payment terms",
                    ],
                    metadata={
                        "strategy": "business_model",
                        "has_revenue": str(features.has_revenue),
                    },
                )
            )

        if bm_obs and not recommendations:
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.OPPORTUNITY.value,
                    action="Analyze unit economics and scalability.",
                    priority=Priority.MEDIUM.value,
                    rationale=(
                        f"Business model observations indicate potential with "
                        f"{len(bm_obs)} data points and {conf:.0%} confidence."
                    ),
                    title="Unit Economics Deep-Dive",
                    description=(
                        "Available business model data suggests viability. "
                        "A detailed unit economics analysis would quantify "
                        "margins, CAC/LTV ratios, and scalability potential."
                    ),
                    supporting_observations=bm_obs,
                    supporting_assessments=bm_assess,
                    confidence=(conf + assess_conf) / 2 if assess_conf else conf,
                    expected_impact="Quantifies unit economics and scalability indicators.",
                    action_items=[
                        "Request CAC and LTV data",
                        "Analyze gross margin structure",
                        "Assess scalability of the revenue model",
                    ],
                    metadata={"strategy": "business_model", "observation_count": len(bm_obs)},
                )
            )

        return recommendations
