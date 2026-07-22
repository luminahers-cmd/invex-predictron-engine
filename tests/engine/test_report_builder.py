from predictron_engine.models.report import Report
from predictron_engine.report.report_builder import DefaultReportBuilder

"""Tests for the DefaultReportBuilder."""


class TestDefaultReportBuilder:
    """Unit tests for the report builder stage."""

    def test_build_returns_report(
        self, sample_startup, sample_features, sample_evidence,
        sample_observations, sample_scores, sample_recommendations,
        sample_confidence,
    ):
        builder = DefaultReportBuilder()
        result = builder.build(
            sample_startup, sample_features, sample_evidence,
            sample_observations, sample_scores, sample_recommendations,
            sample_confidence,
        )

        assert isinstance(result, Report)

    def test_report_contains_all_inputs(
        self, sample_startup, sample_features, sample_evidence,
        sample_observations, sample_scores, sample_recommendations,
        sample_confidence,
    ):
        builder = DefaultReportBuilder()
        result = builder.build(
            sample_startup, sample_features, sample_evidence,
            sample_observations, sample_scores, sample_recommendations,
            sample_confidence,
        )

        assert result.startup == sample_startup
        assert result.features == sample_features
        assert result.evidence == sample_evidence
        assert result.observations == sample_observations
        assert result.scores == sample_scores
        assert result.recommendations == sample_recommendations
        assert result.confidence == sample_confidence

    def test_overall_score_is_average(
        self, sample_startup, sample_features, sample_evidence,
        sample_observations, sample_scores, sample_recommendations,
        sample_confidence,
    ):
        builder = DefaultReportBuilder()
        result = builder.build(
            sample_startup, sample_features, sample_evidence,
            sample_observations, sample_scores, sample_recommendations,
            sample_confidence,
        )

        expected = sum(s.score for s in sample_scores) / len(sample_scores)
        assert result.overall_score == round(expected, 2)

    def test_overall_confidence_is_average(
        self, sample_startup, sample_features, sample_evidence,
        sample_observations, sample_scores, sample_recommendations,
        sample_confidence,
    ):
        builder = DefaultReportBuilder()
        result = builder.build(
            sample_startup, sample_features, sample_evidence,
            sample_observations, sample_scores, sample_recommendations,
            sample_confidence,
        )

        expected = (
            sum(c.confidence for c in sample_confidence)
            / len(sample_confidence)
        )
        assert result.overall_confidence == round(expected, 4)

    def test_metadata_version(self, sample_startup, sample_features):
        builder = DefaultReportBuilder()
        result = builder.build(
            sample_startup, sample_features, [], [], [], [], [],
        )

        assert result.analysis_metadata.engine_version == "0.12.1"

    def test_empty_scores_gives_zero_overall(
        self, sample_startup, sample_features
    ):
        builder = DefaultReportBuilder()
        result = builder.build(
            sample_startup, sample_features, [], [], [], [], [],
        )

        assert result.overall_score == 0.0
        assert result.overall_confidence == 0.0
