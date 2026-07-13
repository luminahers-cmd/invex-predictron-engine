"""Scoring Engine — assigns numerical scores to analysis dimensions.

The scoring layer produces a ScoreResult for each analysis dimension.
It consumes features and observations as input and produces numerical
scores (0-100) with rationales.

Key principles:
  - No hardcoded scoring formulas
  - Scores are produced by injectable DimensionScorer components
  - Each scorer operates independently on a single dimension
  - Default implementation returns placeholder values
  - Real scoring logic will be developed as separate scorer implementations

Extensibility:
  - Implement DimensionScorer for each dimension
  - Inject scorer sets for different analysis contexts
  - Scorer priority and weighting can be configured externally
"""

import logging
from typing import Protocol, runtime_checkable

from predictron_engine.knowledge.concepts import DIMENSION_LABELS, AnalysisDimension
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import Observation, ScoreResult

logger = logging.getLogger(__name__)


@runtime_checkable
class DimensionScorer(Protocol):
    """Protocol for individual dimension scorers.

    Each scorer evaluates features and observations for one analysis
    dimension and produces a ScoreResult with a score and rationale.
    """

    def score(
        self, features: ExtractedFeatures, observations: list[Observation]
    ) -> ScoreResult:
        """Score the relevant dimension based on features and observations."""
        ...


class PlaceholderDimensionScorer:
    """Default scorer that returns a neutral placeholder score.

    This scorer uses basic feature heuristics to produce a minimal
    score. It exists as a template — real scorers will replace it.
    """

    def __init__(self, dimension: AnalysisDimension) -> None:
        self._dimension = dimension

    def score(
        self, features: ExtractedFeatures, observations: list[Observation]
    ) -> ScoreResult:
        relevant_observations = [
            o for o in observations if o.dimension == self._dimension.value
        ]

        base_score = 50.0

        if features.has_pitch_deck:
            base_score += 5.0
        if features.founder_profile_count > 0:
            base_score += 3.0
        if features.description_length > 200:
            base_score += 2.0

        score = max(0.0, min(100.0, base_score))

        evidence = [obs.statement for obs in relevant_observations]

        label = DIMENSION_LABELS.get(self._dimension, self._dimension.value)
        rationale = f"Placeholder scoring for {label} dimension."

        return ScoreResult(
            dimension=self._dimension.value,
            score=score,
            rationale=rationale,
            evidence=evidence,
        )


def _build_default_scorers() -> list[DimensionScorer]:
    """Create a placeholder scorer for each default dimension."""
    return [PlaceholderDimensionScorer(dim) for dim in AnalysisDimension]


class DefaultScoringEngine:
    """Standard implementation of the ScoringEngine protocol.

    Delegates scoring to a set of injectable DimensionScorer components.
    When no scorers are provided, uses placeholder scorers for each
    default analysis dimension.
    """

    def __init__(self, scorers: list[DimensionScorer] | None = None) -> None:
        self._scorers = scorers if scorers is not None else _build_default_scorers()

    def score(
        self, features: ExtractedFeatures, observations: list[Observation]
    ) -> list[ScoreResult]:
        """Score all dimensions and return results."""
        logger.info("Running scoring engine with %d scorers", len(self._scorers))

        results: list[ScoreResult] = []
        for scorer in self._scorers:
            try:
                results.append(scorer.score(features, observations))
            except Exception:
                logger.warning(
                    "Scorer %s failed, skipping", type(scorer).__name__
                )

        logger.info("Produced %d dimension scores", len(results))
        return results
