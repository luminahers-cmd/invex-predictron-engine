"""Confidence Engine — assesses reliability of analysis conclusions.

The confidence layer evaluates how reliable each scoring dimension's
conclusion is, based on data completeness, observation strength,
available evidence, and dimension assessments.

Key principles:
  - Confidence is separate from the score itself
  - Low confidence does not mean low score — it means uncertainty
  - Data completeness is a first-class confidence factor
  - Default implementation uses data completeness as a baseline
  - Real confidence models will be developed and injected over time

Extensibility:
  - Inject statistical models for confidence calculation
  - Add evidence-quality assessors as independent components
  - Confidence thresholds can be configured per analysis context
"""

import logging

from predictron_engine.evaluation.evaluation_models import DimensionAssessment
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    ConfidenceAssessment,
    Observation,
    ScoreResult,
)

logger = logging.getLogger(__name__)


class DefaultConfidenceEngine:
    """Standard implementation of the ConfidenceEngine protocol.

    Produces a confidence assessment for each scoring dimension.
    The default implementation uses data completeness, observation
    coverage, and dimension assessment confidence as factors.
    """

    def assess(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        scores: list[ScoreResult],
        assessments: list[DimensionAssessment] | None = None,
    ) -> list[ConfidenceAssessment]:
        """Assess confidence for each scored dimension."""
        logger.info("Running confidence assessment for %d dimensions", len(scores))

        completeness = features.data_completeness
        assessment_map = {a.dimension: a for a in (assessments or [])}

        confidence_assessments: list[ConfidenceAssessment] = []
        for score in scores:
            obs_count = sum(
                1 for o in observations if o.dimension == score.dimension
            )
            obs_confidence = (
                sum(
                    o.confidence
                    for o in observations
                    if o.dimension == score.dimension
                )
                / max(obs_count, 1)
            )

            # Incorporate dimension assessment confidence if available
            assessment_confidence = 0.0
            if score.dimension in assessment_map:
                assessment_confidence = assessment_map[score.dimension].confidence

            overall = (
                (completeness * 0.3)
                + (obs_confidence * 0.3)
                + (assessment_confidence * 0.2)
                + 0.2
            )
            overall = max(0.0, min(1.0, overall))

            factors: list[str] = [
                f"data_completeness={completeness:.2f}",
                f"observation_count={obs_count}",
                f"avg_observation_confidence={obs_confidence:.2f}",
                f"assessment_confidence={assessment_confidence:.2f}",
            ]

            confidence_assessments.append(
                ConfidenceAssessment(
                    dimension=score.dimension,
                    confidence=round(overall, 4),
                    factors=factors,
                    data_completeness=completeness,
                )
            )

        logger.info("Produced %d confidence assessments", len(confidence_assessments))
        return confidence_assessments
