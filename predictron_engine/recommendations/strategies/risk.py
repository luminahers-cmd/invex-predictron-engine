"""Risk recommendation strategy.

Generates recommendations based on risk observations
and assessments. Focuses on competitive position, risk factors,
and mitigation opportunities.
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


class RiskStrategy:
    """Generates risk-related recommendations.

    Focuses on competitive position assessment, risk factor
    identification, and mitigation strategies.
    """

    @property
    def domain(self) -> str:
        return "competitive_position"

    def generate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        assessments: list[DimensionAssessment],
    ) -> list[Recommendation]:
        recommendations: list[Recommendation] = []
        risk_obs = filter_observations(observations, self.domain)
        risk_assess = filter_assessments(assessments, self.domain)

        conf = observation_confidence(risk_obs)
        assess_conf = assessment_confidence(risk_assess)

        # Check data completeness as a risk signal
        if features.data_completeness < 0.3:
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.RISK_MITIGATION.value,
                    action="Request additional data to reduce analysis uncertainty.",
                    priority=Priority.HIGH.value,
                    rationale=(
                        f"Data completeness is only {features.data_completeness:.0%}, "
                        f"limiting analysis confidence."
                    ),
                    title="Improve Data Coverage",
                    description=(
                        f"With only {features.data_completeness:.0%} data completeness, "
                        f"the analysis has significant blind spots. Requesting "
                        f"additional information would improve assessment accuracy."
                    ),
                    confidence=0.7,
                    expected_impact="Reduces analysis uncertainty and improves assessment quality.",
                    action_items=[
                        "Complete the company profile with missing information",
                        "Provide additional documentation",
                        "Share key metrics and performance data",
                    ],
                    metadata={
                        "strategy": "risk",
                        "data_completeness": features.data_completeness,
                    },
                )
            )

        if risk_obs:
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.RISK_MITIGATION.value,
                    action="Review identified risk factors in detail.",
                    priority=Priority.MEDIUM.value,
                    rationale=(
                        f"{len(risk_obs)} risk-related observations were identified "
                        f"with {conf:.0%} average confidence."
                    ),
                    title="Risk Factor Review",
                    description=(
                        f"The analysis identified {len(risk_obs)} risk-related "
                        f"observations. A focused review of these factors would "
                        f"help assess their materiality and mitigation options."
                    ),
                    supporting_observations=risk_obs,
                    supporting_assessments=risk_assess,
                    confidence=(conf + assess_conf) / 2 if assess_conf else conf,
                    expected_impact="Enables informed risk assessment and mitigation planning.",
                    action_items=[
                        "Review each identified risk factor",
                        "Assess risk materiality and likelihood",
                        "Identify mitigation strategies",
                    ],
                    metadata={"strategy": "risk", "observation_count": len(risk_obs)},
                )
            )

        return recommendations
