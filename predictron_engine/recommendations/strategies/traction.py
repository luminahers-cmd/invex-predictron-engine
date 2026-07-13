"""Traction recommendation strategy.

Generates recommendations based on traction observations
and assessments. Focuses on market validation, growth signals,
and stage-appropriate expectations.
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


class TractionStrategy:
    """Generates traction-related recommendations.

    Focuses on market validation signals, growth metrics,
    and stage-appropriate expectations.
    """

    @property
    def domain(self) -> str:
        return "traction_signals"

    def generate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        assessments: list[DimensionAssessment],
    ) -> list[Recommendation]:
        recommendations: list[Recommendation] = []
        traction_obs = filter_observations(observations, self.domain)
        traction_assess = filter_assessments(assessments, self.domain)

        conf = observation_confidence(traction_obs)
        assess_conf = assessment_confidence(traction_assess)

        if features.funding_stage is None:
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.DUE_DILIGENCE.value,
                    action="Clarify current funding stage and capital requirements.",
                    priority=Priority.MEDIUM.value,
                    rationale="Funding stage could not be determined.",
                    title="Identify Funding Stage",
                    description=(
                        "The current funding stage was not determined. "
                        "Understanding the stage helps set appropriate "
                        "expectations and benchmarks."
                    ),
                    confidence=0.3,
                    expected_impact="Enables stage-appropriate benchmarking.",
                    action_items=[
                        "Ask about current and previous funding rounds",
                        "Understand capital raised to date",
                        "Identify runway and future capital needs",
                    ],
                    metadata={"strategy": "traction", "missing_field": "funding_stage"},
                )
            )

        if features.founded_year is not None:
            from datetime import datetime

            company_age = datetime.now().year - features.founded_year
            if company_age > 3 and features.has_revenue is not True:
                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.RISK_MITIGATION.value,
                        action="Investigate revenue generation status for mature company.",
                        priority=Priority.HIGH.value,
                        rationale=(
                            f"Company is {company_age} years old but revenue "
                            f"status is unconfirmed."
                        ),
                        title="Revenue Validation for Mature Startup",
                        description=(
                            f"At {company_age} years old, the company should "
                            f"have clear revenue signals. The absence of revenue "
                            f"data for a company of this age warrants investigation."
                        ),
                        confidence=0.5,
                        expected_impact="Identifies potential viability risks early.",
                        action_items=[
                            "Request revenue metrics and growth data",
                            "Understand path to profitability",
                            "Assess burn rate and runway",
                        ],
                        metadata={
                            "strategy": "traction",
                            "company_age_years": company_age,
                        },
                    )
                )

        if traction_obs and not recommendations:
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.OPPORTUNITY.value,
                    action="Validate traction metrics with supporting data.",
                    priority=Priority.MEDIUM.value,
                    rationale=(
                        f"Traction observations indicate signals with "
                        f"{len(traction_obs)} data points and {conf:.0%} confidence."
                    ),
                    title="Traction Validation",
                    description=(
                        "Available traction data suggests market validation. "
                        "Requesting concrete metrics would strengthen the "
                        "traction assessment."
                    ),
                    supporting_observations=traction_obs,
                    supporting_assessments=traction_assess,
                    confidence=(conf + assess_conf) / 2 if assess_conf else conf,
                    expected_impact="Quantifies market validation and growth trajectory.",
                    action_items=[
                        "Request customer growth metrics",
                        "Ask about retention and churn rates",
                        "Review pipeline and forward commitments",
                    ],
                    metadata={"strategy": "traction", "observation_count": len(traction_obs)},
                )
            )

        return recommendations
