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


class CompetitionScorer:
    """Scores the COMPETITIVE_POSITION dimension based on competition features.

    Uses weighted contributions from market concentration, competitive density,
    moat indicators, switching costs, network effects, and barriers to entry.
    All scoring logic is deterministic and traceable.
    """

    def __init__(self) -> None:
        self._dimension = AnalysisDimension.COMPETITIVE_POSITION

    def score(
        self, features: ExtractedFeatures, observations: list[Observation]
    ) -> ScoreResult:
        relevant_obs = [
            o for o in observations if o.dimension == self._dimension.value
        ]

        base_score = 50.0
        adjustments: list[tuple[str, float]] = []

        base_score, adj_conc = self._score_concentration(base_score, features)
        if adj_conc != 0:
            adjustments.append(("market_concentration", adj_conc))

        base_score, adj_den = self._score_density(base_score, features)
        if adj_den != 0:
            adjustments.append(("competitive_density", adj_den))

        base_score, adj_moat = self._score_moats(base_score, features)
        if adj_moat != 0:
            adjustments.append(("moat_indicators", adj_moat))

        base_score, adj_sw = self._score_switching(base_score, features)
        if adj_sw != 0:
            adjustments.append(("switching_costs", adj_sw))

        base_score, adj_net = self._score_network(base_score, features)
        if adj_net != 0:
            adjustments.append(("network_effects", adj_net))

        base_score, adj_bar = self._score_barriers(base_score, features)
        if adj_bar != 0:
            adjustments.append(("barriers_to_entry", adj_bar))

        base_score, adj_oss = self._score_open_source(base_score, features)
        if adj_oss != 0:
            adjustments.append(("open_source_competition", adj_oss))

        base_score, adj_diff = self._score_differentiation(base_score, features)
        if adj_diff != 0:
            adjustments.append(("differentiation", adj_diff))

        base_score, adj_obs = self._score_observations(base_score, relevant_obs)
        if adj_obs != 0:
            adjustments.append(("observations", adj_obs))

        final_score = max(0.0, min(100.0, base_score))
        rationale = (
            f"Competitive position scored from {len(adjustments)} signal dimensions. "
            f"Base 50, final {final_score:.1f}."
        )

        return ScoreResult(
            dimension=self._dimension.value,
            score=final_score,
            rationale=rationale,
            evidence=[o.statement for o in relevant_obs],
        )

    @staticmethod
    def _score_concentration(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.market_concentration:
            return score, 0.0
        key = features.market_concentration.lower()
        mapping = {
            "fragmented": 5.0,
            "moderately_concentrated": 0.0,
            "concentrated": -8.0,
            "dominated": -15.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_density(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.competitive_density:
            return score, 0.0
        key = features.competitive_density.lower()
        mapping = {
            "sparse": 5.0,
            "moderate": 0.0,
            "dense": -6.0,
            "hyper_competitive": -12.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_moats(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.competitive_moat_indicators:
            return score, 0.0
        count = len(features.competitive_moat_indicators)
        delta = min(15.0, count * 3.0)
        return score + delta, delta

    @staticmethod
    def _score_switching(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.switching_cost_signals:
            return score, 0.0
        count = len(features.switching_cost_signals)
        delta = min(12.0, count * 3.0)
        return score + delta, delta

    @staticmethod
    def _score_network(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.network_effect_competition:
            return score, 0.0
        key = features.network_effect_competition.lower()
        mapping = {
            "strong_network_effects": 12.0,
            "moderate_network_effects": 5.0,
            "no_network_effects": -3.0,
        }
        delta = mapping.get(key, 0.0)
        return score + delta, delta

    @staticmethod
    def _score_barriers(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.barriers_to_entry:
            return score, 0.0
        count = len(features.barriers_to_entry)
        delta = min(10.0, count * 3.0)
        return score + delta, delta

    @staticmethod
    def _score_open_source(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.open_source_competition:
            return score, 0.0
        count = len(features.open_source_competition)
        delta = min(10.0, count * 2.5)
        return score - delta, -delta

    @staticmethod
    def _score_differentiation(
        score: float, features: ExtractedFeatures
    ) -> tuple[float, float]:
        if not features.differentiation_signals:
            return score, 0.0
        count = len(features.differentiation_signals)
        delta = min(10.0, count * 2.5)
        return score + delta, delta

    @staticmethod
    def _score_observations(
        score: float, observations: list[Observation]
    ) -> tuple[float, float]:
        if not observations:
            return score, 0.0
        total_impact = sum(o.importance for o in observations if hasattr(o, "importance"))
        avg_importance = total_impact / len(observations)
        delta = avg_importance * 10.0
        return score + delta, delta


def _build_default_scorers() -> list[DimensionScorer]:
    """Create scorers for each default dimension."""
    scorers: list[DimensionScorer] = []
    for dim in AnalysisDimension:
        if dim == AnalysisDimension.COMPETITIVE_POSITION:
            scorers.append(CompetitionScorer())
        else:
            scorers.append(PlaceholderDimensionScorer(dim))
    return scorers


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
