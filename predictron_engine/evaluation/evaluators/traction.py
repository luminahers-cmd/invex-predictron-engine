"""Traction dimension evaluator.

Translates observations and evidence about traction signals into a
structured DimensionAssessment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from predictron_engine.evaluation.evaluation_models import DimensionAssessment
from predictron_engine.evaluation.evaluators.base import (
    calculate_average_confidence,
    calculate_weighted_importance,
    filter_evidence_by_domain,
    filter_observations,
    generate_cross_signal_context,
    generate_rationale,
    generate_summary,
)

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation


class TractionEvaluator:
    """Evaluates the traction dimension.

    Translates traction-related observations and evidence into a structured
    assessment explaining market validation and growth signals.
    """

    @property
    def dimension(self) -> str:
        return "traction_signals"

    def evaluate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        evidence: list[EvidenceItem],
    ) -> DimensionAssessment:
        """Evaluate traction signals based on observations and evidence."""
        traction_obs = filter_observations(observations, self.dimension)
        traction_evidence = filter_evidence_by_domain(evidence, "traction")

        summary = generate_summary(self.dimension, traction_obs, traction_evidence)
        rationale = generate_rationale(self.dimension, traction_obs, traction_evidence)
        confidence = calculate_average_confidence(traction_obs)
        weighted_conf = calculate_weighted_importance(traction_obs)

        cross_ctx = generate_cross_signal_context(observations, self.dimension)
        if cross_ctx:
            rationale += cross_ctx

        return DimensionAssessment(
            dimension=self.dimension,
            summary=summary,
            rationale=rationale,
            confidence=confidence,
            supporting_observations=traction_obs,
            supporting_evidence=traction_evidence,
            metadata={
                "has_revenue": features.has_revenue or False,
                "funding_stage": features.funding_stage or "unknown",
                "weighted_confidence": round(weighted_conf, 4),
                "cross_signal_available": bool(cross_ctx),
            },
        )
