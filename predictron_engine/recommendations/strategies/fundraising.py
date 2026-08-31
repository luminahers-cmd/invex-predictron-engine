"""Fundraising recommendation strategy.

Generates recommendations based on fundraising context from features
and assessments. Focuses on funding stage alignment, capital needs,
and fundraising readiness.
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


class FundraisingStrategy:
    """Generates fundraising-related recommendations.

    Focuses on funding stage alignment, capital requirements assessment,
    and fundraising readiness evaluation.
    """

    @property
    def domain(self) -> str:
        return "fundraising"

    def generate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        assessments: list[DimensionAssessment],
        scores: list[ScoreResult] | None = None,
        readiness: InvestmentReadiness | None = None,
    ) -> list[Recommendation]:
        recommendations: list[Recommendation] = []
        if features.funding_stage is not None:
            stage_obs = filter_observations(observations, "traction_signals")
            stage_assess = filter_assessments(assessments, "traction_signals")

            conf = observation_confidence(stage_obs)
            assess_conf = assessment_confidence(stage_assess)

            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.FOLLOW_UP.value,
                    action=f"Validate {features.funding_stage} stage benchmarks.",
                    priority=Priority.MEDIUM.value,
                    rationale=(
                        f"Company is at {features.funding_stage} stage. "
                        f"Stage-appropriate benchmarks should be validated."
                    ),
                    title=f"Stage Benchmark Validation ({features.funding_stage})",
                    description=(
                        f"At the {features.funding_stage} stage, specific "
                        f"benchmarks and expectations apply. Validating the "
                        f"company against these benchmarks helps assess "
                        f"investment readiness."
                    ),
                    supporting_observations=stage_obs,
                    supporting_assessments=stage_assess,
                    confidence=(conf + assess_conf) / 2 if assess_conf else conf,
                    expected_impact="Validates stage-appropriate expectations and readiness.",
                    action_items=[
                        "Review stage-specific benchmarks",
                        "Compare against peer companies at same stage",
                        "Assess readiness for next funding round",
                    ],
                    metadata={
                        "strategy": "fundraising",
                        "funding_stage": features.funding_stage,
                    },
                )
            )

        if features.founder_profile_count > 0 and features.has_pitch_deck:
            all_assess = filter_assessments(assessments, "market_opportunity") + filter_assessments(
                assessments, "product_strength"
            )
            avg_conf = assessment_confidence(all_assess) if all_assess else 0.0

            if avg_conf > 0.5:
                recommendations.append(
                    Recommendation(
                        category=RecommendationCategory.OPPORTUNITY.value,
                        action="Consider scheduling an investment deep-dive.",
                        priority=Priority.MEDIUM.value,
                        rationale=(
                            f"Key data points are available with {avg_conf:.0%} "
                            f"assessment confidence. Sufficient data exists for "
                            f"a deeper evaluation."
                        ),
                        title="Investment Deep-Dive Readiness",
                        description=(
                            "The available data suggests sufficient coverage "
                            "for a more detailed investment evaluation. Key "
                            "indicators are within assessable ranges."
                        ),
                        supporting_assessments=all_assess,
                        confidence=avg_conf,
                        expected_impact="Progresses from screening to detailed evaluation.",
                        action_items=[
                            "Schedule detailed management presentation",
                            "Request financial model and projections",
                            "Prepare investment committee materials",
                        ],
                        metadata={"strategy": "fundraising", "assessment_confidence": avg_conf},
                    )
                )

        return recommendations
