from predictron_engine.confidence.confidence_engine import DefaultConfidenceEngine
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import ConfidenceAssessment

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
