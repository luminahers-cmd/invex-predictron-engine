"""Data quality dimension evaluator.

Translates observations and evidence about data completeness and quality
into a structured DimensionAssessment.
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


class DataQualityEvaluator:
    """Evaluates the data quality dimension.

    Translates data quality-related observations and evidence into a
    structured assessment explaining data completeness and reliability.
    """

    @property
    def dimension(self) -> str:
        return "team_execution"

    def evaluate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        evidence: list[EvidenceItem],
    ) -> DimensionAssessment:
        """Evaluate data quality based on observations and evidence."""
        dq_obs = filter_observations(observations, self.dimension)
        dq_evidence = filter_evidence_by_domain(evidence, "data_quality")

        summary = generate_summary(self.dimension, dq_obs, dq_evidence)
        rationale = generate_rationale(self.dimension, dq_obs, dq_evidence)
        confidence = calculate_average_confidence(dq_obs)

        return DimensionAssessment(
            dimension=self.dimension,
            summary=summary,
            rationale=rationale,
            confidence=confidence,
            supporting_observations=dq_obs,
            supporting_evidence=dq_evidence,
            metadata={
                "data_completeness": features.data_completeness,
                "description_length": features.description_length,
            },
        )
