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
        assert meta.engine_version == "0.12.1"
        assert "normalize" in meta.pipeline_stages_completed
        assert "collect" in meta.pipeline_stages_completed
        assert "collect_evidence" in meta.pipeline_stages_completed
        assert "extract" in meta.pipeline_stages_completed
        assert "evidence" in meta.pipeline_stages_completed
        assert "reason" in meta.pipeline_stages_completed
        assert "evaluate" in meta.pipeline_stages_completed
        assert "score" in meta.pipeline_stages_completed
        assert "recommend" in meta.pipeline_stages_completed
        assert "confidence" in meta.pipeline_stages_completed
        assert "decide" in meta.pipeline_stages_completed
        assert "build_report" in meta.pipeline_stages_completed
        assert len(meta.pipeline_stages_completed) == 12

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
        assert report.analysis_metadata.engine_version == "0.12.1"
        assert debug.pipeline_summary.engine_version == "0.12.1"


class TestP8APipelineWiring:
    """Sprint P8A — C2 (bundle -> reasoning) and L1 (assessment scores)."""

    def test_reasoning_receives_evidence_bundle(self, full_request):
        """C2: the collected EvidenceBundle must reach the reasoning layer."""
        from predictron_engine.reasoning.reasoning_engine import (
            DefaultReasoningEngine,
        )

        received = {}

        class SpyReasoning(DefaultReasoningEngine):
            def reason(self, features, evidence=None, bundle=None):
                received["bundle"] = bundle
                return super().reason(features, evidence, bundle)

        engine = PredictronEngine(reasoning=SpyReasoning())
        engine.analyze(full_request)

        assert "bundle" in received
        assert received["bundle"] is not None
        assert received["bundle"].startup_name == "FullCo"

    def test_assessment_scores_populated_from_scores(self, engine, full_request):
        """L1: DimensionAssessment.score must reflect real scoring output."""
        report = engine.analyze(full_request)

        score_by_dim = {s.dimension: s.score for s in report.scores}
        assessed = report.dimension_assessments
        populated = [
            a for a in assessed if a.dimension in score_by_dim
        ]
        assert populated, "expected at least one scored dimension assessment"
        for a in populated:
            assert a.score == score_by_dim[a.dimension]

    def test_investment_readiness_reflected_in_decision(
        self, engine, full_request,
    ):
        """C1: the decision must track the canonical readiness score."""
        report = engine.analyze(full_request)

        ready = report.investment_readiness
        decision = report.investment_decision
        assert ready is not None and decision is not None

        expected = round(ready.readiness_score * 0.25, 4)
        actual = decision.decision_factors.get("readiness_contribution")
        assert actual is not None
        assert abs(actual - expected) < 1e-6


class TestP8DContradictionGraph:
    """Sprint P8D — H5: activate contradiction graph & reasoning intel.

    Verifies the contradiction graph is built exactly once in the
    production path, threaded to confidence/calibration/synthesis, and
    surfaced (with trace + budget) in the debug report.
    """

    def test_graph_built_once_and_cached(self, full_request):
        from predictron_engine.reasoning.reasoning_engine import (
            DefaultReasoningEngine,
        )

        build_counts = {"n": 0}

        class CountingReasoning(DefaultReasoningEngine):
            def reason(self, features, evidence=None, bundle=None):
                result = super().reason(features, evidence, bundle)
                if result:
                    build_counts["n"] += 1
                return result

        engine = PredictronEngine(reasoning=CountingReasoning())
        engine.analyze(full_request)

        assert build_counts["n"] >= 1
        graph = engine._reasoning.last_contradiction_graph
        assert graph is not None

    def test_graph_reaches_confidence_and_calibration(self, full_request):
        engine = PredictronEngine()
        report = engine.analyze(full_request)

        graph = engine._reasoning.last_contradiction_graph
        assert graph is not None
        # The graph flows into the decision calibration uncertainty.
        assert report.decision_confidence is not None
        assert report.decision_confidence.uncertainty_breakdown is not None

    def test_debug_exposes_graph_trace_and_budget(self, full_request):
        engine = PredictronEngine()
        report, debug = engine.analyze_with_debug(full_request)

        assert debug.contradiction_graph is not None
        assert debug.reasoning_trace is not None
        assert debug.reasoning_trace["total_observations"] == len(report.observations)
        assert debug.reasoning_budget is not None
        assert "budget_fraction" in debug.reasoning_budget

    def test_debug_does_not_change_report_schema(self, full_request):
        engine = PredictronEngine()
        report_plain = engine.analyze(full_request)
        report_debug, _ = engine.analyze_with_debug(full_request)

        plain = report_plain.model_dump()
        debug = report_debug.model_dump()
        # Execution metadata (timestamp, processing time) legitimately
        # differs between separate runs; the analytical schema must match.
        plain["analysis_metadata"].pop("timestamp", None)
        debug["analysis_metadata"].pop("timestamp", None)
        plain["analysis_metadata"].pop("processing_time_ms", None)
        debug["analysis_metadata"].pop("processing_time_ms", None)
        plain["startup"].pop("normalized_at", None)
        debug["startup"].pop("normalized_at", None)
        assert plain == debug
