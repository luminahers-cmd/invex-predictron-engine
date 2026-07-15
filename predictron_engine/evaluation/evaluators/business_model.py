"""Business model dimension evaluator.

Translates observations and evidence about business model viability into a
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


class BusinessModelEvaluator:
    """Evaluates the business model dimension.

    Translates business model-related observations and evidence into a
    structured assessment explaining revenue model and unit economics.
    """

    @property
    def dimension(self) -> str:
        return "business_model_viability"

    def evaluate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        evidence: list[EvidenceItem],
    ) -> DimensionAssessment:
        """Evaluate business model viability based on observations and evidence."""
        bm_obs = filter_observations(observations, self.dimension)
        bm_evidence = filter_evidence_by_domain(evidence, "business_model")

        summary = generate_summary(self.dimension, bm_obs, bm_evidence)
        rationale = generate_rationale(self.dimension, bm_obs, bm_evidence)
        confidence = calculate_average_confidence(bm_obs)
        weighted_conf = calculate_weighted_importance(bm_obs)

        cross_ctx = generate_cross_signal_context(observations, self.dimension)
        if cross_ctx:
            rationale += cross_ctx

        return DimensionAssessment(
            dimension=self.dimension,
            summary=summary,
            rationale=rationale,
            confidence=confidence,
            supporting_observations=bm_obs,
            supporting_evidence=bm_evidence,
            metadata={
                "business_model": features.business_model or "unknown",
                "customer_type": features.customer_type or "unknown",
                "weighted_confidence": round(weighted_conf, 4),
                "cross_signal_available": bool(cross_ctx),
            },
        )
