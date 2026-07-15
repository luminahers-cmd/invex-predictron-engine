"""Team dimension evaluator.

Translates observations and evidence about founder quality and team
execution into a structured DimensionAssessment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from predictron_engine.evaluation.evaluation_models import DimensionAssessment
from predictron_engine.evaluation.evaluators.base import (
    calculate_average_confidence,
    filter_evidence_by_domain,
    filter_observations,
    generate_rationale,
    generate_summary,
)

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation


class TeamEvaluator:
    """Evaluates the team dimension.

    Translates team-related observations and evidence into a structured
    assessment explaining founder quality and team execution capability.
    """

    @property
    def dimension(self) -> str:
        return "founder_quality"

    def evaluate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        evidence: list[EvidenceItem],
    ) -> DimensionAssessment:
        """Evaluate team quality based on observations and evidence."""
        team_obs = filter_observations(observations, self.dimension)
        team_evidence = filter_evidence_by_domain(evidence, "team")

        summary = generate_summary(self.dimension, team_obs, team_evidence)
        rationale = generate_rationale(self.dimension, team_obs, team_evidence)
        confidence = calculate_average_confidence(team_obs)

        metadata: dict[str, object] = {
            "founder_count": features.founder_profile_count,
            "team_size": features.team_size_indicator or "unknown",
            "founder_team_type": features.founder_team_type or "unknown",
            "domain_expertise_count": len(features.domain_expertise_signals),
            "serial_founder_signals": len(features.serial_founder_indicators),
            "leadership_roles": features.leadership_roles,
            "engineering_strength": features.engineering_strength or "unknown",
            "product_strength": features.product_strength or "unknown",
            "founder_market_fit_signals": len(features.founder_market_fit_signals),
            "execution_signals_count": len(features.execution_signals),
            "hiring_signals_count": len(features.hiring_signals),
            "advisor_mentions_count": len(features.advisor_mentions),
            "founder_confidence": features.founder_confidence,
        }

        return DimensionAssessment(
            dimension=self.dimension,
            summary=summary,
            rationale=rationale,
            confidence=confidence,
            supporting_observations=team_obs,
            supporting_evidence=team_evidence,
            metadata=metadata,
        )
