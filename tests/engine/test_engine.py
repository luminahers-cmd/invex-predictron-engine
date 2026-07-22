from predictron_engine.engine import PredictronEngine
from predictron_engine.models.report import Report

"""Tests for the PredictronEngine orchestrator."""


class TestEnginePipeline:
    """Integration tests for the complete analysis pipeline."""

    def test_full_pipeline_minimal_input(self, engine, minimal_request):
        report = engine.analyze(minimal_request)

        assert isinstance(report, Report)
        assert report.startup.name == "TestCo"
        assert "https://testco.example.com" in report.startup.website

    def test_full_pipeline_maximal_input(self, engine, full_request):
        report = engine.analyze(full_request)

        assert isinstance(report, Report)
        assert report.startup.name == "FullCo"
        assert report.startup.pitch_deck_url is not None
        assert len(report.startup.founder_linkedin_urls) == 2

    def test_report_has_all_pipeline_stages(self, engine, full_request):
        report = engine.analyze(full_request)

        assert report.features is not None
        assert isinstance(report.evidence, list)
        assert isinstance(report.observations, list)
        assert isinstance(report.scores, list)
        assert isinstance(report.recommendations, list)
        assert isinstance(report.confidence, list)

    def test_overall_score_in_valid_range(self, engine, full_request):
        report = engine.analyze(full_request)

        assert 0.0 <= report.overall_score <= 100.0

    def test_overall_confidence_in_valid_range(self, engine, full_request):
        report = engine.analyze(full_request)

        assert 0.0 <= report.overall_confidence <= 1.0

    def test_metadata_populated(self, engine, full_request):
        report = engine.analyze(full_request)

        meta = report.analysis_metadata
        assert meta.engine_version == "0.10.0"
        assert "normalize" in meta.pipeline_stages_completed
        assert "collect" in meta.pipeline_stages_completed
        assert "extract" in meta.pipeline_stages_completed
        assert "evidence" in meta.pipeline_stages_completed
        assert "reason" in meta.pipeline_stages_completed
        assert "evaluate" in meta.pipeline_stages_completed
        assert "score" in meta.pipeline_stages_completed
        assert "recommend" in meta.pipeline_stages_completed
        assert "confidence" in meta.pipeline_stages_completed
        assert "decide" in meta.pipeline_stages_completed
        assert "build_report" in meta.pipeline_stages_completed
        assert len(meta.pipeline_stages_completed) == 11

    def test_processing_time_positive(self, engine, full_request):
        report = engine.analyze(full_request)

        assert report.analysis_metadata.processing_time_ms >= 0.0

    def test_saas_input_classification(self, engine, saas_request):
        report = engine.analyze(saas_request)

        assert report.features.industry is not None
        assert report.features.business_model is not None
        assert report.features.has_pitch_deck is True
        assert report.features.founder_profile_count == 1

    def test_multiple_engines_are_independent(self, minimal_request):
        engine_a = PredictronEngine()
        engine_b = PredictronEngine()

        report_a = engine_a.analyze(minimal_request)
        report_b = engine_b.analyze(minimal_request)

        assert report_a.analysis_metadata.timestamp is not None
        assert report_b.analysis_metadata.timestamp is not None

    def test_analyze_with_debug_returns_tuple(self, engine, full_request):
        report, debug = engine.analyze_with_debug(full_request)

        assert isinstance(report, Report)
        assert debug.pipeline_summary.startup_name == "FullCo"
        assert debug.trace_graph is not None
        assert len(debug.stage_outputs) > 0
        assert len(debug.warnings) >= 0

    def test_analyze_with_debug_has_explanations(self, engine, full_request):
        report, debug = engine.analyze_with_debug(full_request)

        explained = [
            so for so in debug.stage_outputs if so.explanation
        ]
        assert len(explained) >= 4

    def test_analyze_with_debug_preserves_report(self, engine, full_request):
        report, debug = engine.analyze_with_debug(full_request)

        assert report.overall_score >= 0.0
        assert report.analysis_metadata.engine_version == "0.10.0"
        assert debug.pipeline_summary.engine_version == "0.10.0"
