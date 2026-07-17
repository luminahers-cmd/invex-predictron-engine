"""Confidence Engine — assesses reliability of analysis conclusions.

The confidence layer evaluates how reliable each scoring dimension's
conclusion is, based on data completeness, observation strength,
available evidence, and dimension assessments.

Key principles:
  - Confidence is separate from the score itself
  - Low confidence does not mean low score — it means uncertainty
  - Data completeness is a first-class confidence factor
  - Evidence quality and diversity improve confidence
  - Sparse or conflicting evidence reduces confidence
  - Confidence is penalized when observations are sparse or inconsistent

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

_MIN_OBSERVATIONS_FOR_FULL_CONFIDENCE = 3


class DefaultConfidenceEngine:
    """Standard implementation of the ConfidenceEngine protocol.

    Produces a confidence assessment for each scoring dimension.
    The implementation uses data completeness, observation coverage,
    observation diversity, evidence quality, and dimension assessment
    confidence as factors. Confidence is reduced when observations
    are sparse, conflicting, or when evidence is insufficient.
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
            dim_observations = [
                o for o in observations if o.dimension == score.dimension
            ]
            obs_count = len(dim_observations)

            obs_confidence = (
                sum(o.confidence for o in dim_observations) / obs_count
                if obs_count > 0
                else 0.0
            )

            observation_diversity = self._compute_observation_diversity(
                dim_observations
            )

            conflict_penalty = self._compute_conflict_penalty(dim_observations)

            assessment_confidence = 0.0
            if score.dimension in assessment_map:
                assessment_confidence = assessment_map[score.dimension].confidence

            coverage_factor = self._compute_coverage_factor(obs_count)

            overall = (
                (completeness * 0.25)
                + (obs_confidence * 0.25)
                + (assessment_confidence * 0.20)
                + (observation_diversity * 0.10)
                + (coverage_factor * 0.10)
                + 0.15
            )

            overall -= conflict_penalty

            overall = max(0.0, min(1.0, overall))

            factors: list[str] = [
                f"data_completeness={completeness:.2f}",
                f"observation_count={obs_count}",
                f"avg_observation_confidence={obs_confidence:.2f}",
                f"observation_diversity={observation_diversity:.2f}",
                f"coverage_factor={coverage_factor:.2f}",
                f"assessment_confidence={assessment_confidence:.2f}",
                f"conflict_penalty={conflict_penalty:.2f}",
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

    @staticmethod
    def _compute_observation_diversity(observations: list[Observation]) -> float:
        """Compute diversity of observation categories for a dimension.

        More diverse categories indicate broader evidence coverage,
        increasing confidence in the assessment.
        """
        if not observations:
            return 0.0
        categories = {o.category for o in observations}
        diversity = min(len(categories) / 3.0, 1.0)
        return diversity

    @staticmethod
    def _compute_conflict_penalty(observations: list[Observation]) -> float:
        """Compute confidence penalty from conflicting observations.

        Conflicting observations reduce confidence because they
        indicate uncertainty in the dimension's assessment.
        """
        if not observations:
            return 0.0
        conflict_count = sum(
            1 for o in observations if o.category == "signal_conflict"
        )
        if conflict_count == 0:
            return 0.0
        total = len(observations)
        conflict_ratio = conflict_count / total
        return min(conflict_ratio * 0.3, 0.2)

    @staticmethod
    def _compute_coverage_factor(observation_count: int) -> float:
        """Compute confidence factor based on observation count.

        Very few observations indicate sparse evidence, reducing
        confidence in the dimension's assessment.
        """
        if observation_count == 0:
            return 0.0
        return min(
            observation_count / _MIN_OBSERVATIONS_FOR_FULL_CONFIDENCE, 1.0
        )
