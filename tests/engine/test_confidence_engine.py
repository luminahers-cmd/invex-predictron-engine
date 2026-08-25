from predictron_engine.confidence.confidence_engine import DefaultConfidenceEngine
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    ConfidenceAssessment,
    Observation,
)

"""Tests for the DefaultConfidenceEngine."""


class TestDefaultConfidenceEngine:
    """Unit tests for the confidence stage."""

    def test_assess_returns_assessments(
        self, sample_features, sample_observations, sample_scores
    ):
        engine = DefaultConfidenceEngine()
        result = engine.assess(sample_features, sample_observations, sample_scores)

        assert isinstance(result, list)
        assert all(isinstance(c, ConfidenceAssessment) for c in result)

    def test_one_assessment_per_score(
        self, sample_features, sample_observations, sample_scores
    ):
        engine = DefaultConfidenceEngine()
        result = engine.assess(sample_features, sample_observations, sample_scores)

        assert len(result) == len(sample_scores)

    def test_confidence_in_valid_range(
        self, sample_features, sample_observations, sample_scores
    ):
        engine = DefaultConfidenceEngine()
        result = engine.assess(sample_features, sample_observations, sample_scores)

        for assessment in result:
            assert 0.0 <= assessment.confidence <= 1.0

    def test_data_completeness_propagated(
        self, sample_features, sample_observations, sample_scores
    ):
        engine = DefaultConfidenceEngine()
        result = engine.assess(sample_features, sample_observations, sample_scores)

        for assessment in result:
            assert assessment.data_completeness == sample_features.data_completeness

    def test_factors_populated(
        self, sample_features, sample_observations, sample_scores
    ):
        engine = DefaultConfidenceEngine()
        result = engine.assess(sample_features, sample_observations, sample_scores)

        for assessment in result:
            assert len(assessment.factors) > 0

    def test_low_data_completeness_lowers_confidence(
        self, sample_observations, sample_scores
    ):
        engine = DefaultConfidenceEngine()

        high_completeness = ExtractedFeatures(
            data_completeness=0.9,
            description_length=500,
            has_pitch_deck=True,
            founder_profile_count=2,
        )
        low_completeness = ExtractedFeatures(
            data_completeness=0.1,
            description_length=50,
        )

        high_result = engine.assess(
            high_completeness, sample_observations, sample_scores
        )
        low_result = engine.assess(
            low_completeness, sample_observations, sample_scores
        )

        high_avg = sum(r.confidence for r in high_result) / len(high_result)
        low_avg = sum(r.confidence for r in low_result) / len(low_result)

        assert high_avg > low_avg

    def test_no_observations_reduces_confidence(
        self, sample_features, sample_scores
    ):
        engine = DefaultConfidenceEngine()
        result = engine.assess(sample_features, [], sample_scores)

        for assessment in result:
            assert assessment.confidence < 0.5

    def test_conflicting_observations_reduce_confidence(
        self, sample_features, sample_scores
    ):
        engine = DefaultConfidenceEngine()

        clean_observations = [
            Observation(
                dimension="market_opportunity",
                category="market_context",
                statement="Strong market position.",
                evidence=["industry=enterprise_saas"],
                confidence=0.7,
                importance=0.8,
                source_rule="MarketContextRule",
            ),
        ]

        conflicting_observations = [
            Observation(
                dimension="market_opportunity",
                category="market_context",
                statement="Strong market position.",
                evidence=["industry=enterprise_saas"],
                confidence=0.7,
                importance=0.8,
                source_rule="MarketContextRule",
            ),
            Observation(
                dimension="market_opportunity",
                category="signal_conflict",
                statement="Conflict detected in market signals.",
                evidence=[],
                confidence=0.6,
                importance=0.7,
                source_rule="CrossSignalReasoningRule",
            ),
        ]

        clean_result = engine.assess(
            sample_features, clean_observations, sample_scores
        )
        conflict_result = engine.assess(
            sample_features, conflicting_observations, sample_scores
        )

        clean_market = next(
            r for r in clean_result if r.dimension == "market_opportunity"
        )
        conflict_market = next(
            r for r in conflict_result if r.dimension == "market_opportunity"
        )

        assert clean_market.confidence > conflict_market.confidence

    def test_diverse_observations_increase_confidence(
        self, sample_features, sample_scores
    ):
        engine = DefaultConfidenceEngine()

        single_category = [
            Observation(
                dimension="market_opportunity",
                category="market_context",
                statement="Market analysis.",
                evidence=[],
                confidence=0.6,
                importance=0.6,
                source_rule="MarketContextRule",
            ),
        ]

        diverse_categories = [
            Observation(
                dimension="market_opportunity",
                category="market_context",
                statement="Market analysis.",
                evidence=[],
                confidence=0.6,
                importance=0.6,
                source_rule="MarketContextRule",
            ),
            Observation(
                dimension="market_opportunity",
                category="market_size",
                statement="Large market opportunity.",
                evidence=[],
                confidence=0.7,
                importance=0.7,
                source_rule="MarketContextRule",
            ),
            Observation(
                dimension="market_opportunity",
                category="market_growth",
                statement="Growing market segment.",
                evidence=[],
                confidence=0.65,
                importance=0.65,
                source_rule="MarketContextRule",
            ),
        ]

        single_result = engine.assess(
            sample_features, single_category, sample_scores
        )
        diverse_result = engine.assess(
            sample_features, diverse_categories, sample_scores
        )

        single_market = next(
            r for r in single_result if r.dimension == "market_opportunity"
        )
        diverse_market = next(
            r for r in diverse_result if r.dimension == "market_opportunity"
        )

        assert diverse_market.confidence >= single_market.confidence

    def test_zero_observations_gets_minimal_confidence(
        self, sample_features, sample_scores
    ):
        engine = DefaultConfidenceEngine()
        result = engine.assess(sample_features, [], sample_scores)

        for assessment in result:
            assert assessment.confidence <= 0.5

    def test_weight_sum_exactly_one(self):
        """Regression: factor weights + base must sum to exactly 1.0."""
        from predictron_engine.evaluation.evaluation_models import (
            DimensionAssessment,
        )

        engine = DefaultConfidenceEngine()
        features = ExtractedFeatures(
            data_completeness=1.0,
            description_length=500,
            has_pitch_deck=True,
            founder_profile_count=2,
        )
        obs = [
            Observation(
                dimension="market_opportunity",
                category="market_context",
                statement="x",
                evidence=[],
                confidence=1.0,
                importance=1.0,
                source_rule="r",
            ),
            Observation(
                dimension="market_opportunity",
                category="market_size",
                statement="y",
                evidence=[],
                confidence=1.0,
                importance=1.0,
                source_rule="r",
            ),
            Observation(
                dimension="market_opportunity",
                category="market_growth",
                statement="z",
                evidence=[],
                confidence=1.0,
                importance=1.0,
                source_rule="r",
            ),
        ]
        from predictron_engine.models.report import ScoreResult

        scores = [ScoreResult(dimension="market_opportunity", score=80.0)]
        assessments = [
            DimensionAssessment(
                dimension="market_opportunity",
                summary="s",
                rationale="r",
                confidence=1.0,
            ),
        ]
        result = engine.assess(features, obs, scores, assessments)
        assert abs(result[0].confidence - 1.0) < 1e-6
