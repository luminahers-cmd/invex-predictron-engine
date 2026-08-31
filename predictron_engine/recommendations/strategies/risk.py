"""Risk recommendation strategy.

Generates recommendations based on risk observations
and assessments. Focuses on competitive position, risk factors,
and mitigation opportunities.
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
        scores: list[ScoreResult] | None = None,
        readiness: InvestmentReadiness | None = None,
    ) -> list[Recommendation]:
        recommendations: list[Recommendation] = []
        risk_obs = filter_observations(observations, self.domain)
        risk_assess = filter_assessments(assessments, self.domain)

        conf = observation_confidence(risk_obs)
        assess_conf = assessment_confidence(risk_assess)

        recommendations.extend(self._recommend_data_coverage(features, conf))
        recommendations.extend(
            self._recommend_risk_review(risk_obs, risk_assess, conf, assess_conf)
        )
        recommendations.extend(self._recommend_concentration(features))
        recommendations.extend(self._recommend_open_source(features))
        recommendations.extend(self._recommend_moat_development(features))

        return recommendations

    @staticmethod
    def _recommend_data_coverage(features: ExtractedFeatures, conf: float) -> list[Recommendation]:
        if features.data_completeness >= 0.3:
            return []
        return [
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
        ]

    @staticmethod
    def _recommend_risk_review(
        risk_obs: list[Observation],
        risk_assess: list[DimensionAssessment],
        conf: float,
        assess_conf: float,
    ) -> list[Recommendation]:
        if not risk_obs:
            return []
        return [
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
        ]

    @staticmethod
    def _recommend_concentration(features: ExtractedFeatures) -> list[Recommendation]:
        if features.market_concentration not in ("concentrated", "dominated"):
            return []
        return [
            Recommendation(
                category=RecommendationCategory.RISK_MITIGATION.value,
                action="Evaluate defensibility against dominant incumbents.",
                priority=Priority.HIGH.value,
                rationale=(
                    f"Market is {features.market_concentration}. "
                    f"Incumbent dominance creates significant entry barriers."
                ),
                title="Concentrated Market Risk",
                description=(
                    f"The market is {features.market_concentration}. "
                    f"Assess specific defensibility strategies against "
                    f"established players and identify viable niche positions."
                ),
                confidence=0.75,
                expected_impact="Identifies viable positioning strategies in competitive markets.",
                action_items=[
                    "Map incumbent strengths and weaknesses",
                    "Identify underserved market segments",
                    "Assess differentiation requirements",
                ],
                metadata={
                    "strategy": "risk",
                    "market_concentration": features.market_concentration,
                },
            )
        ]

    @staticmethod
    def _recommend_open_source(features: ExtractedFeatures) -> list[Recommendation]:
        if not features.open_source_competition:
            return []
        count = len(features.open_source_competition)
        return [
            Recommendation(
                category=RecommendationCategory.RISK_MITIGATION.value,
                action="Assess open-source competitive threat and response strategy.",
                priority=Priority.MEDIUM.value,
                rationale=(
                    f"{count} open-source competition signals detected. "
                    f"Open-source alternatives may compress margins."
                ),
                title="Open-Source Competition Response",
                description=(
                    f"Open-source competition ({count} signals) may impact "
                    f"pricing and defensibility. Develop a strategy that "
                    f"leverages proprietary advantages over open alternatives."
                ),
                confidence=0.6,
                expected_impact=(
                    "Clarifies competitive positioning against open-source alternatives."
                ),
                action_items=[
                    "Audit open-source alternatives in the space",
                    "Identify proprietary value-adds",
                    "Assess pricing strategy relative to free alternatives",
                ],
                metadata={
                    "strategy": "risk",
                    "open_source_count": count,
                },
            )
        ]

    @staticmethod
    def _recommend_moat_development(features: ExtractedFeatures) -> list[Recommendation]:
        if features.competitive_moat_indicators:
            return []
        has_moat_signals = bool(
            features.switching_cost_signals
            or features.network_effect_competition == "strong_network_effects"
        )
        if has_moat_signals:
            return []
        return [
            Recommendation(
                category=RecommendationCategory.RISK_MITIGATION.value,
                action="Develop or strengthen competitive moats.",
                priority=Priority.MEDIUM.value,
                rationale=(
                    "No strong competitive moat indicators detected. "
                    "Without moats, the business is vulnerable to competition."
                ),
                title="Strengthen Competitive Moats",
                description=(
                    "The analysis did not detect strong moat indicators. "
                    "Focus on building defensible advantages through data, "
                    "network effects, switching costs, or proprietary technology."
                ),
                confidence=0.5,
                expected_impact=(
                    "Improves long-term defensibility and reduces competitive vulnerability."
                ),
                action_items=[
                    "Identify potential moat sources in the business model",
                    "Assess network effect or data advantages",
                    "Plan investments in defensibility",
                ],
                metadata={
                    "strategy": "risk",
                    "has_moat_indicators": False,
                },
            )
        ]
