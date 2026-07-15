"""Risk dimension evaluator.

Translates observations and evidence about competitive position and risk
factors into a structured DimensionAssessment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from predictron_engine.evaluation.evaluation_models import DimensionAssessment
from predictron_engine.evaluation.evaluators.base import (
    calculate_average_confidence,
    calculate_weighted_importance,
    filter_evidence_by_domain,
    filter_observations,
    generate_rationale,
    generate_summary,
)

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation


class RiskEvaluator:
    """Evaluates the risk dimension.

    Translates risk-related observations and evidence into a structured
    assessment explaining competitive position and risk factors.
    """

    @property
    def dimension(self) -> str:
        return "competitive_position"

    def evaluate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        evidence: list[EvidenceItem],
    ) -> DimensionAssessment:
        """Evaluate competitive position and risk based on observations and evidence."""
        risk_obs = filter_observations(observations, self.dimension)
        risk_evidence = filter_evidence_by_domain(evidence, "competition")

        summary = generate_summary(self.dimension, risk_obs, risk_evidence)
        rationale = generate_rationale(self.dimension, risk_obs, risk_evidence)
        confidence = calculate_average_confidence(risk_obs)
        weighted_conf = calculate_weighted_importance(risk_obs)

        return DimensionAssessment(
            dimension=self.dimension,
            summary=summary,
            rationale=rationale,
            confidence=confidence,
            supporting_observations=risk_obs,
            supporting_evidence=risk_evidence,
            metadata={
                "industry": features.industry or "unknown",
                "geography": features.geography or "unknown",
                "market_concentration": features.market_concentration or "unknown",
                "competitive_density": features.competitive_density or "unknown",
                "network_effect_competition": features.network_effect_competition or "unknown",
                "moat_count": len(features.competitive_moat_indicators),
                "switching_cost_count": len(features.switching_cost_signals),
                "barrier_count": len(features.barriers_to_entry),
                "open_source_competition_count": len(features.open_source_competition),
                "weighted_confidence": round(weighted_conf, 4),
            },
        )
