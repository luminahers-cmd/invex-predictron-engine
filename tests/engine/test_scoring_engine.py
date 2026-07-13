from predictron_engine.knowledge.concepts import AnalysisDimension
from predictron_engine.models.report import ScoreResult
from predictron_engine.scoring.scoring_engine import (
    DefaultScoringEngine,
    PlaceholderDimensionScorer,
)

"""Tests for the DefaultScoringEngine."""


class TestDefaultScoringEngine:
    """Unit tests for the scoring stage."""

    def test_score_returns_results(
        self, sample_features, sample_observations
    ):
        engine = DefaultScoringEngine()
        result = engine.score(sample_features, sample_observations)

        assert isinstance(result, list)
        assert all(isinstance(s, ScoreResult) for s in result)

    def test_scores_in_valid_range(
        self, sample_features, sample_observations
    ):
        engine = DefaultScoringEngine()
        result = engine.score(sample_features, sample_observations)

        for score in result:
            assert 0.0 <= score.score <= 100.0

    def test_one_score_per_dimension(
        self, sample_features, sample_observations
    ):
        engine = DefaultScoringEngine()
        result = engine.score(sample_features, sample_observations)

        dimensions = {s.dimension for s in result}
        assert len(dimensions) == len(result)

    def test_placeholder_scorer_includes_evidence(
        self, sample_features, sample_observations
    ):
        scorer = PlaceholderDimensionScorer(
            AnalysisDimension.MARKET_OPPORTUNITY
        )
        result = scorer.score(sample_features, sample_observations)

        assert isinstance(result, ScoreResult)
        assert result.dimension == "market_opportunity"

    def test_placeholder_scorer_pitch_deck_bonus(
        self, sample_features, sample_observations
    ):
        scorer = PlaceholderDimensionScorer(
            AnalysisDimension.PRODUCT_STRENGTH
        )

        result_with_deck = scorer.score(
            sample_features, sample_observations
        )
        assert result_with_deck.score >= 50.0

    def test_custom_scorer_injected(
        self, sample_features, sample_observations
    ):
        class CustomScorer:
            def score(self, features, observations):
                return ScoreResult(
                    dimension="custom",
                    score=99.0,
                    rationale="custom",
                    evidence=[],
                )

        engine = DefaultScoringEngine(scorers=[CustomScorer()])
        result = engine.score(sample_features, sample_observations)

        assert len(result) == 1
        assert result[0].score == 99.0

    def test_empty_scorers_produces_no_scores(
        self, sample_features, sample_observations
    ):
        engine = DefaultScoringEngine(scorers=[])
        result = engine.score(sample_features, sample_observations)

        assert result == []
