"""Market dimension evaluator.

Translates observations and evidence about market opportunity into a
structured DimensionAssessment with summary, rationale, and supporting evidence.
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


class MarketEvaluator:
    """Evaluates the market opportunity dimension.

    Translates market-related observations and evidence into a structured
    assessment explaining the market's size, growth, competition, and fit.
    """

    @property
    def dimension(self) -> str:
        return "market_opportunity"

    def evaluate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        evidence: list[EvidenceItem],
    ) -> DimensionAssessment:
        """Evaluate market opportunity based on observations and evidence."""
        market_obs = filter_observations(observations, self.dimension)
        market_evidence = filter_evidence_by_domain(evidence, "industry")

        summary = generate_summary(self.dimension, market_obs, market_evidence)
        rationale = generate_rationale(self.dimension, market_obs, market_evidence)
        confidence = calculate_average_confidence(market_obs)
        weighted_conf = calculate_weighted_importance(market_obs)

        cross_ctx = generate_cross_signal_context(observations, self.dimension)
        if cross_ctx:
            rationale += cross_ctx

        return DimensionAssessment(
            dimension=self.dimension,
            summary=summary,
            rationale=rationale,
            confidence=confidence,
            supporting_observations=market_obs,
            supporting_evidence=market_evidence,
            metadata={
                "industry": features.industry or "unknown",
                "geography": features.geography or "unknown",
                "weighted_confidence": round(weighted_conf, 4),
                "cross_signal_available": bool(cross_ctx),
            },
        )
