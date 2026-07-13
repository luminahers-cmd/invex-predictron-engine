"""Team recommendation strategy.

Generates recommendations based on team and founder observations
and assessments. Focuses on founder quality, team composition,
and execution capability.
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


class TeamStrategy:
    """Generates team-related recommendations.

    Focuses on founder quality, team composition, and execution
    capability assessment.
    """

    @property
    def domain(self) -> str:
        return "founder_quality"

    def generate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        assessments: list[DimensionAssessment],
    ) -> list[Recommendation]:
        recommendations: list[Recommendation] = []
        team_obs = filter_observations(observations, self.domain)
        team_assess = filter_assessments(assessments, self.domain)

        conf = observation_confidence(team_obs)
        assess_conf = assessment_confidence(team_assess)

        if features.founder_profile_count == 0:
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.DUE_DILIGENCE.value,
                    action="Request founder profiles and background information.",
                    priority=Priority.HIGH.value,
                    rationale="No founder profiles were provided for team evaluation.",
                    title="Obtain Founder Information",
                    description=(
                        "No founder profiles were provided. Understanding the "
                        "founding team's background, experience, and track record "
                        "is critical for assessing execution capability."
                    ),
                    confidence=0.5,
                    expected_impact="Enables assessment of founder quality and team strength.",
                    action_items=[
                        "Request LinkedIn profiles for all founders",
                        "Ask about previous startup experience",
                        "Inquire about domain expertise relevant to the business",
                    ],
                    metadata={"strategy": "team", "founder_count": 0},
                )
            )
        elif features.founder_profile_count == 1:
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.FOLLOW_UP.value,
                    action="Inquire about the full founding team composition.",
                    priority=Priority.MEDIUM.value,
                    rationale=(
                        "Only one founder profile was provided. Assessing team "
                        "breadth requires understanding the complete founding team."
                    ),
                    title="Expand Team Visibility",
                    description=(
                        "A single founder profile was provided. Understanding "
                        "the full team composition helps assess complementary "
                        "skills and execution capacity."
                    ),
                    confidence=0.4,
                    expected_impact="Provides fuller picture of team capabilities.",
                    action_items=[
                        "Ask about co-founders and key team members",
                        "Request team org chart or bios",
                        "Identify key hires planned",
                    ],
                    metadata={"strategy": "team", "founder_count": 1},
                )
            )

        team_dimension_obs = filter_observations(observations, "team_execution")
        if team_dimension_obs and not team_obs:
            team_obs = team_dimension_obs

        if team_obs and not recommendations:
            recommendations.append(
                Recommendation(
                    category=RecommendationCategory.OPPORTUNITY.value,
                    action="Conduct in-depth team due diligence.",
                    priority=Priority.MEDIUM.value,
                    rationale=(
                        f"Team observations indicate potential with "
                        f"{len(team_obs)} data points and {conf:.0%} confidence."
                    ),
                    title="Team Deep-Dive Assessment",
                    description=(
                        "Available team data suggests competence. A deeper "
                        "assessment of team dynamics, complementary skills, "
                        "and execution track record is recommended."
                    ),
                    supporting_observations=team_obs,
                    supporting_assessments=team_assess,
                    confidence=(conf + assess_conf) / 2 if assess_conf else conf,
                    expected_impact="Validates team capability and identifies gaps.",
                    action_items=[
                        "Schedule founder interviews",
                        "Review past project outcomes",
                        "Assess technical and business skill coverage",
                    ],
                    metadata={"strategy": "team", "observation_count": len(team_obs)},
                )
            )

        return recommendations
