"""Team assessment reasoning rule.

Evaluates founder and team intelligence signals to produce observations
about the team's composition, capability, and readiness.
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

    Evaluates founder profile count, team size, founder type,
    domain expertise, serial founder indicators, leadership roles,
    engineering/product strength, execution signals, and founder-market fit
    to describe what is known about the team composition and capability.
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

        # --- Founder profile count ---
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

        # --- Team size indicator ---
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

        # --- Founder team type ---
        if features.founder_team_type:
            observations.append(
                Obs(
                    dimension=AnalysisDimension.FOUNDER_QUALITY.value,
                    category="team_composition",
                    statement=(
                        f"Founding team composition leans "
                        f"{features.founder_team_type}."
                    ),
                    evidence=[feature_ref("founder_team_type", features.founder_team_type)],
                    confidence=0.55,
                    importance=0.55,
                    source_rule="TeamAssessmentRule",
                )
            )

        # --- Domain expertise ---
        if features.domain_expertise_signals:
            expertise_str = ", ".join(features.domain_expertise_signals[:3])
            observations.append(
                Obs(
                    dimension=AnalysisDimension.FOUNDER_QUALITY.value,
                    category="domain_expertise",
                    statement=(
                        f"Detected domain expertise signals: {expertise_str}."
                    ),
                    evidence=[
                        feature_ref("domain_expertise_signals", s)
                        for s in features.domain_expertise_signals[:3]
                    ],
                    confidence=0.6,
                    importance=0.65,
                    source_rule="TeamAssessmentRule",
                )
            )

        # --- Serial founder indicators ---
        if features.serial_founder_indicators:
            observations.append(
                Obs(
                    dimension=AnalysisDimension.FOUNDER_QUALITY.value,
                    category="serial_founder",
                    statement=(
                        f"Detected {len(features.serial_founder_indicators)} "
                        f"serial founder indicator(s) suggesting prior startup "
                        f"experience."
                    ),
                    evidence=[
                        feature_ref("serial_founder_indicators", ind)
                        for ind in features.serial_founder_indicators[:2]
                    ],
                    confidence=0.55,
                    importance=0.6,
                    source_rule="TeamAssessmentRule",
                )
            )

        # --- Leadership roles ---
        if features.leadership_roles:
            roles_str = ", ".join(features.leadership_roles[:4])
            observations.append(
                Obs(
                    dimension=AnalysisDimension.FOUNDER_QUALITY.value,
                    category="leadership_roles",
                    statement=(
                        f"Detected leadership roles: {roles_str}."
                    ),
                    evidence=[
                        feature_ref("leadership_roles", r)
                        for r in features.leadership_roles[:4]
                    ],
                    confidence=0.6,
                    importance=0.55,
                    source_rule="TeamAssessmentRule",
                )
            )

        # --- Engineering strength ---
        if features.engineering_strength:
            observations.append(
                Obs(
                    dimension=AnalysisDimension.TEAM_EXECUTION.value,
                    category="engineering_strength",
                    statement=(
                        f"Engineering strength assessed as "
                        f"{features.engineering_strength}."
                    ),
                    evidence=[
                        feature_ref("engineering_strength", features.engineering_strength)
                    ],
                    confidence=0.55,
                    importance=0.6,
                    source_rule="TeamAssessmentRule",
                )
            )

        # --- Product strength ---
        if features.product_strength:
            observations.append(
                Obs(
                    dimension=AnalysisDimension.TEAM_EXECUTION.value,
                    category="product_strength",
                    statement=(
                        f"Product strength assessed as "
                        f"{features.product_strength}."
                    ),
                    evidence=[
                        feature_ref("product_strength", features.product_strength)
                    ],
                    confidence=0.55,
                    importance=0.55,
                    source_rule="TeamAssessmentRule",
                )
            )

        # --- Founder-market fit ---
        if features.founder_market_fit_signals:
            observations.append(
                Obs(
                    dimension=AnalysisDimension.FOUNDER_QUALITY.value,
                    category="founder_market_fit",
                    statement=(
                        f"Detected {len(features.founder_market_fit_signals)} "
                        f"founder-market fit signal(s) indicating relevant "
                        f"background and domain alignment."
                    ),
                    evidence=[
                        feature_ref("founder_market_fit_signals", s)
                        for s in features.founder_market_fit_signals[:2]
                    ],
                    confidence=0.6,
                    importance=0.7,
                    source_rule="TeamAssessmentRule",
                )
            )

        # --- Execution signals ---
        if features.execution_signals:
            observations.append(
                Obs(
                    dimension=AnalysisDimension.TEAM_EXECUTION.value,
                    category="execution_signals",
                    statement=(
                        f"Detected {len(features.execution_signals)} "
                        f"execution and traction signal(s) indicating "
                        f"operational progress."
                    ),
                    evidence=[
                        feature_ref("execution_signals", s)
                        for s in features.execution_signals[:3]
                    ],
                    confidence=min(
                        0.4 + len(features.execution_signals) * 0.05, 0.8
                    ),
                    importance=0.65,
                    source_rule="TeamAssessmentRule",
                )
            )

        # --- Hiring signals ---
        if features.hiring_signals:
            observations.append(
                Obs(
                    dimension=AnalysisDimension.TEAM_EXECUTION.value,
                    category="hiring_signals",
                    statement=(
                        f"Detected {len(features.hiring_signals)} hiring "
                        f"and team growth signal(s)."
                    ),
                    evidence=[
                        feature_ref("hiring_signals", s)
                        for s in features.hiring_signals[:2]
                    ],
                    confidence=0.5,
                    importance=0.4,
                    source_rule="TeamAssessmentRule",
                )
            )

        # --- Advisor mentions ---
        if features.advisor_mentions:
            observations.append(
                Obs(
                    dimension=AnalysisDimension.FOUNDER_QUALITY.value,
                    category="advisor_mentions",
                    statement=(
                        f"Detected {len(features.advisor_mentions)} advisor "
                        f"or advisory board mention(s)."
                    ),
                    evidence=[
                        feature_ref("advisor_mentions", m)
                        for m in features.advisor_mentions[:2]
                    ],
                    confidence=0.5,
                    importance=0.4,
                    source_rule="TeamAssessmentRule",
                )
            )

        # --- Founder confidence composite ---
        if features.founder_confidence > 0:
            observations.append(
                Obs(
                    dimension=AnalysisDimension.FOUNDER_QUALITY.value,
                    category="founder_confidence",
                    statement=(
                        f"Composite founder confidence score: "
                        f"{features.founder_confidence:.2f}."
                    ),
                    evidence=[
                        feature_ref("founder_confidence", features.founder_confidence)
                    ],
                    confidence=features.founder_confidence,
                    importance=0.7,
                    source_rule="TeamAssessmentRule",
                )
            )

        return observations
