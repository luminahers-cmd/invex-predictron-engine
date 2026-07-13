from predictron_engine.models.report import Recommendation
from predictron_engine.recommendations.recommendation_engine import (
    DefaultRecommendationEngine,
)

"""Tests for the DefaultRecommendationEngine."""


class TestDefaultRecommendationEngine:
    """Unit tests for the recommendation stage."""

    def test_recommend_returns_list(
        self, sample_features, sample_observations, sample_scores
    ):
        engine = DefaultRecommendationEngine()
        result = engine.recommend(
            sample_features, sample_observations, sample_scores
        )

        assert isinstance(result, list)
        assert all(isinstance(r, Recommendation) for r in result)

    def test_recommendations_have_required_fields(
        self, sample_features, sample_observations, sample_scores
    ):
        engine = DefaultRecommendationEngine()
        result = engine.recommend(
            sample_features, sample_observations, sample_scores
        )

        for rec in result:
            assert rec.category != ""
            assert rec.action != ""
            assert rec.priority in ("high", "medium", "low")

    def test_pitch_deck_recommendation_when_missing(self, minimal_features):
        engine = DefaultRecommendationEngine()
        result = engine.recommend(minimal_features, [], [])

        pitch_deck_recs = [
            r for r in result if "pitch deck" in r.action.lower()
        ]
        assert len(pitch_deck_recs) == 1

    def test_founder_recommendation_when_missing(self, minimal_features):
        engine = DefaultRecommendationEngine()
        result = engine.recommend(minimal_features, [], [])

        founder_recs = [
            r for r in result if "founder" in r.action.lower()
        ]
        assert len(founder_recs) == 1

    def test_custom_strategy_injected(
        self, sample_features, sample_observations, sample_scores
    ):
        class CustomStrategy:
            def generate(self, features, observations, scores):
                return [
                    Recommendation(
                        category="custom",
                        action="Custom action",
                        priority="high",
                        rationale="Custom rationale",
                    )
                ]

        engine = DefaultRecommendationEngine(strategies=[CustomStrategy()])
        result = engine.recommend(
            sample_features, sample_observations, sample_scores
        )

        assert len(result) == 1
        assert result[0].category == "custom"

    def test_empty_strategies_produces_no_recommendations(
        self, sample_features, sample_observations, sample_scores
    ):
        engine = DefaultRecommendationEngine(strategies=[])
        result = engine.recommend(
            sample_features, sample_observations, sample_scores
        )

        assert result == []
