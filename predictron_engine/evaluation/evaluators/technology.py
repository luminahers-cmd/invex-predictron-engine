"""Technology dimension evaluator.

Translates observations and evidence about product strength and technology
stack into a structured DimensionAssessment.
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


class TechnologyEvaluator:
    """Evaluates the technology dimension.

    Translates technology-related observations and evidence into a structured
    assessment explaining product strength and technical capabilities.
    """

    @property
    def dimension(self) -> str:
        return "product_strength"

    def evaluate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        evidence: list[EvidenceItem],
    ) -> DimensionAssessment:
        """Evaluate technology and product strength based on observations and evidence."""
        tech_obs = filter_observations(observations, self.dimension)
        tech_evidence = filter_evidence_by_domain(evidence, "technology")

        summary = generate_summary(self.dimension, tech_obs, tech_evidence)
        rationale = generate_rationale(self.dimension, tech_obs, tech_evidence)
        confidence = calculate_average_confidence(tech_obs)

        return DimensionAssessment(
            dimension=self.dimension,
            summary=summary,
            rationale=rationale,
            confidence=confidence,
            supporting_observations=tech_obs,
            supporting_evidence=tech_evidence,
            metadata={
                "technology_stack": features.technology_stack,
                "has_pitch_deck": features.has_pitch_deck,
            },
        )
