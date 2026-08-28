"""Composite evaluator — orchestrates all dimension evaluators.

This module provides the CompositeEvaluator, which orchestrates all
dimension evaluators using protocol-based dependency injection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from predictron_engine.evaluation.evaluation_models import EvaluationResult

if TYPE_CHECKING:
    from predictron_engine.evaluation.evaluators.base import DimensionEvaluator
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import DimensionAssessment, Observation

logger = logging.getLogger(__name__)


class CompositeEvaluator:
    """Orchestrates all dimension evaluators.

    This class composes independent dimension evaluators into a cohesive
    evaluation flow. All dependencies are injected via the constructor.
    """

    def __init__(self, evaluators: list[DimensionEvaluator] | None = None) -> None:
        if evaluators is None:
            evaluators = self._create_default_evaluators()
        self._evaluators = evaluators
        logger.info(
            "CompositeEvaluator initialized with %d evaluators",
            len(self._evaluators),
        )

    def evaluate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        evidence: list[EvidenceItem],
    ) -> EvaluationResult:
        """Execute all evaluators and aggregate results."""
        assessments = []
        for evaluator in self._evaluators:
            try:
                assessment = evaluator.evaluate(features, observations, evidence)
                assessments.append(assessment)
                logger.debug(
                    "Evaluated dimension %s with confidence %.2f",
                    assessment.dimension,
                    assessment.confidence,
                )
            except Exception:
                logger.exception(
                    "Failed to evaluate dimension %s",
                    evaluator.dimension,
                )

        # Calculate aggregate metrics
        total_confidence = sum(a.confidence for a in assessments)
        avg_confidence = total_confidence / len(assessments) if assessments else 0.0

        return EvaluationResult(
            assessments=assessments,
            overall_summary=self._generate_overall_summary(assessments),
            overall_confidence=avg_confidence,
            dimensions_assessed=len(assessments),
            metadata={"evaluator_count": len(self._evaluators)},
        )

    def _generate_overall_summary(
        self, assessments: list[DimensionAssessment]
    ) -> str:
        """Generate a high-level summary across all dimensions."""
        if not assessments:
            return "No dimensions were assessed."

        assessed_dims = [a.dimension for a in assessments]
        return (
            f"Completed assessment of {len(assessments)} dimensions: "
            f"{', '.join(assessed_dims)}."
        )

    @staticmethod
    def _create_default_evaluators() -> list[DimensionEvaluator]:
        """Create the default set of evaluators."""
        from predictron_engine.evaluation.evaluators.business_model import (
            BusinessModelEvaluator,
        )
        from predictron_engine.evaluation.evaluators.data_quality import (
            DataQualityEvaluator,
        )
        from predictron_engine.evaluation.evaluators.market import MarketEvaluator
        from predictron_engine.evaluation.evaluators.risk import RiskEvaluator
        from predictron_engine.evaluation.evaluators.team import TeamEvaluator
        from predictron_engine.evaluation.evaluators.technology import (
            TechnologyEvaluator,
        )
        from predictron_engine.evaluation.evaluators.traction import (
            TractionEvaluator,
        )

        return [
            MarketEvaluator(),
            TeamEvaluator(),
            TechnologyEvaluator(),
            TractionEvaluator(),
            BusinessModelEvaluator(),
            RiskEvaluator(),
            DataQualityEvaluator(),
        ]
