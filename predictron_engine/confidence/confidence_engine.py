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

Ownership
---------
This module owns the pipeline's **per-dimension confidence assessment**
stage (:meth:`DefaultConfidenceEngine.assess`), producing one
:class:`ConfidenceAssessment` per scored dimension.  It is distinct from
two other confidence implementations in the codebase:

* :mod:`predictron_engine.reasoning.confidence` — reasoning-layer
  confidence for the observation set (:class:`ConfidenceBreakdown`).
* :mod:`predictron_engine.decision.calibration` — decision-level
  confidence and uncertainty for an investment decision.

These three compute confidence for different pipeline artifacts at
different stages and are intentionally NOT consolidated.

Sprint P8D additions (backward compatible):
  - ``assess()`` optionally accepts a
    :class:`~predictron_engine.reasoning.contradiction_graph.ContradictionGraph`.
    When supplied, the per-dimension conflict penalty reuses the richer
    O(n²) pairwise contradiction counts instead of only the coarse
    per-observation ``evidence_conflict_count`` proxy.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.reasoning.contradiction_graph import (
        ContradictionGraph,
    )

from predictron_engine.evaluation.evaluation_models import DimensionAssessment
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    NEGATIVE_SIGNAL_CATEGORIES,
    ConfidenceAssessment,
    Observation,
    ScoreResult,
)

logger = logging.getLogger(__name__)

_MIN_OBSERVATIONS_FOR_FULL_CONFIDENCE = 3

# Weight of the baseline confidence term.  The baseline scales with data
# completeness so it is justified by actual evidence rather than being an
# unconditional floor: empty input receives no invented confidence while
# complete evidence still sums to exactly 1.0.
_BASE_CONFIDENCE_WEIGHT: float = 0.15


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
        contradiction_graph: "ContradictionGraph | None" = None,
    ) -> list[ConfidenceAssessment]:
        """Assess confidence for each scored dimension.

        When a ``contradiction_graph`` is supplied, the per-dimension
        conflict penalty reuses its richer O(n²) pairwise contradiction
        counts so the confidence reflects the actual intensity of
        dimension-level contradictions rather than only the coarse
        per-observation ``evidence_conflict_count`` proxy.
        """
        logger.info("Running confidence assessment for %d dimensions", len(scores))

        completeness = features.data_completeness
        assessment_map = {a.dimension: a for a in (assessments or [])}

        obs_by_dim: dict[str, list[Observation]] = {}
        for obs in observations:
            obs_by_dim.setdefault(obs.dimension, []).append(obs)

        # When the graph is available, pre-compute per-dimension
        # conflicting-edge counts so the penalty uses richer data.
        graph_conflicts_by_dim: dict[str, int] = {}
        graph_total_conflicts = 0
        if contradiction_graph is not None:
            graph_total_conflicts = contradiction_graph.conflicting_count
            for dim, summary in contradiction_graph.per_dimension.items():
                graph_conflicts_by_dim[dim] = summary.conflicting_count

        confidence_assessments: list[ConfidenceAssessment] = []
        for score in scores:
            dim_observations = obs_by_dim.get(score.dimension, [])
            obs_count = len(dim_observations)

            obs_confidence = (
                sum(o.confidence for o in dim_observations) / obs_count
                if obs_count > 0
                else 0.0
            )

            observation_diversity = self._compute_observation_diversity(
                dim_observations
            )

            conflict_penalty = self._compute_conflict_penalty(
                dim_observations,
                graph_conflicts_by_dim=graph_conflicts_by_dim,
                graph_total_conflicts=graph_total_conflicts,
            )

            assessment_confidence = 0.0
            if score.dimension in assessment_map:
                assessment_confidence = assessment_map[score.dimension].confidence

            coverage_factor = self._compute_coverage_factor(obs_count)

            overall = (
                (completeness * 0.22)
                + (obs_confidence * 0.22)
                + (assessment_confidence * 0.18)
                + (observation_diversity * 0.12)
                + (coverage_factor * 0.11)
                + (completeness * _BASE_CONFIDENCE_WEIGHT)
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
                f"base_confidence={completeness * _BASE_CONFIDENCE_WEIGHT:.2f}",
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
        categories = sorted({o.category for o in observations})
        diversity = min(len(categories) / 3.0, 1.0)
        return diversity

    @staticmethod
    def _compute_conflict_penalty(
        observations: list[Observation],
        *,
        graph_conflicts_by_dim: dict[str, int] | None = None,
        graph_total_conflicts: int = 0,
    ) -> float:
        """Compute confidence penalty from conflicting observations.

        When a pre-computed ``graph_conflicts_by_dim`` mapping is
        available (from the contradiction graph), it is used in
        preference to the coarse per-observation ``evidence_conflict_count``
        proxy.  The total graph-level conflict count replaces the manual
        per-observation tally.
        """
        if not observations:
            return 0.0

        conflict_count = sum(
            1 for o in observations if o.category in NEGATIVE_SIGNAL_CATEGORIES
        )

        if graph_conflicts_by_dim is not None and observations:
            # Use the richer contradiction-graph conflict count for
            # this dimension rather than the per-observation proxy.
            dim = observations[0].dimension
            conflict_count += graph_conflicts_by_dim.get(dim, 0)
        else:
            conflict_count += sum(o.evidence_conflict_count for o in observations)

        if conflict_count == 0:
            return 0.0
        total = len(observations)
        conflict_ratio = min(conflict_count / total, 1.0)
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
