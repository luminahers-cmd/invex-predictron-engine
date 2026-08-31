import pytest

from predictron_engine.decision.calibration import (
    build_calibration_summary,
    compute_decision_confidence,
)
from predictron_engine.evidence.models import EvidenceBundle
from predictron_engine.models.report import Report
from predictron_engine.report.report_builder import DefaultReportBuilder
from predictron_engine.synthesis.engine import DecisionSynthesisEngine

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

    def test_build_can_produce_complete_report(
        self, sample_startup, sample_features, sample_evidence,
        sample_observations, sample_scores, sample_recommendations,
        sample_confidence,
    ):
        """Regression: build() must produce a complete Report when given
        all required inputs, without requiring post-construction mutation.

        Sprint P6A previously patched decision_confidence,
        calibration_summary, decision_synthesis and processing_time_ms
        onto the report after build().
        """
        bundle = EvidenceBundle.empty(sample_startup.name)
        decision_confidence = compute_decision_confidence(
            bundle=bundle,
            observations=sample_observations,
            assessments=[],
            features=sample_features,
            scores=sample_scores,
        )
        calibration_summary = build_calibration_summary(
            decision_confidence,
            observation_count=len(sample_observations),
            assessed_dimension_count=0,
            evidence_document_count=bundle.total_pages,
        )
        synthesis = DecisionSynthesisEngine().synthesize(
            features=sample_features,
            observations=sample_observations,
            assessments=[],
            scores=sample_scores,
            recommendations=sample_recommendations,
            readiness=None,
            decision=None,
            decision_confidence=decision_confidence,
            calibration_summary=calibration_summary,
        )

        builder = DefaultReportBuilder()
        result = builder.build(
            sample_startup, sample_features, sample_evidence,
            sample_observations, sample_scores, sample_recommendations,
            sample_confidence,
            decision_confidence=decision_confidence,
            calibration_summary=calibration_summary,
            decision_synthesis=synthesis,
            processing_time_ms=123.45,
        )

        assert result.decision_confidence == decision_confidence
        assert result.calibration_summary == calibration_summary
        assert result.decision_synthesis == synthesis
        assert result.analysis_metadata.processing_time_ms == pytest.approx(123.45)

    def test_build_without_decision_inputs_keeps_defaults(self, sample_startup, sample_features):
        """Backward compatibility: omitting the new optional params is fine."""
        builder = DefaultReportBuilder()
        result = builder.build(
            sample_startup, sample_features, [], [], [], [], [],
        )
        assert result.decision_confidence is None
        assert result.calibration_summary is None
        assert result.decision_synthesis is None
        assert result.analysis_metadata.processing_time_ms == 0.0
