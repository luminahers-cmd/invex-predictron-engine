"""Team assessment reasoning rule.

Evaluates founder and team signals to produce observations
about the team's composition and presence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from predictron_engine.reasoning.rules.base import feature_ref

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation


class TeamAssessmentRule:
    """Produces observations about the founding team.

    Evaluates founder profile count and team size indicators to
    describe what is known about the team composition.
    """

    @property
    def name(self) -> str:
        return "team_assessment"

    def evaluate(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
    ) -> list[Observation]:
        from predictron_engine.knowledge.concepts import AnalysisDimension
        from predictron_engine.models.report import Observation as Obs

        refs: list[str] = []
        observations: list[Obs] = []

        if features.founder_profile_count > 0:
            refs.append(
                feature_ref("founder_profile_count", features.founder_profile_count)
            )
            observations.append(
                Obs(
                    dimension=AnalysisDimension.FOUNDER_QUALITY.value,
                    category="team_assessment",
                    statement=(
                        f"Identified {features.founder_profile_count} founder "
                        f"profile(s) associated with this startup."
                    ),
                    evidence=list(refs),
                    confidence=min(
                        0.4 + features.founder_profile_count * 0.15, 1.0
                    ),
                    importance=0.6,
                    source_rule="TeamAssessmentRule",
                )
            )

        if features.team_size_indicator:
            refs.append(feature_ref("team_size_indicator", features.team_size_indicator))
            observations.append(
                Obs(
                    dimension=AnalysisDimension.TEAM_EXECUTION.value,
                    category="team_assessment",
                    statement=(
                        f"Team size indicator suggests a "
                        f"{features.team_size_indicator} team."
                    ),
                    evidence=[feature_ref(
                        "team_size_indicator", features.team_size_indicator
                    )],
                    confidence=0.5,
                    importance=0.5,
                    source_rule="TeamAssessmentRule",
                )
            )

        return observations
