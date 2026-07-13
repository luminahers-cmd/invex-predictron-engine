"""Integration tests for the ValidationEngine."""

from predictron_engine.engine import PredictronEngine
from predictron_engine.validation.validation_engine import ValidationEngine
from tests.engine.test_validation.test_dataset import (
    ALL_STARTUPS,
    B2B_SAAS,
    CONSUMER_APP,
    FINTECH,
    ROBOTICS,
)


class TestValidationEngineIntegration:
    """Integration tests running the validation engine against real pipeline output."""

    def test_debug_report_structure(self):
        engine = PredictronEngine()
        val_engine = ValidationEngine()
        report = engine.analyze(B2B_SAAS)
        debug = val_engine.validate_report(report)

        assert debug.pipeline_summary.startup_name == "CloudSync Pro"
        assert debug.pipeline_summary.engine_version == "0.6.5"
        assert len(debug.stage_outputs) > 0
        assert debug.trace_graph is not None
        assert len(debug.trace_graph.nodes) > 0

    def test_debug_report_has_explanations(self):
        engine = PredictronEngine()
        val_engine = ValidationEngine()
        report = engine.analyze(B2B_SAAS)
        debug = val_engine.validate_report(report)

        explained_stages = [
            so.stage for so in debug.stage_outputs if so.explanation
        ]
        assert len(explained_stages) >= 4

    def test_debug_report_has_confidence_summary(self):
        engine = PredictronEngine()
        val_engine = ValidationEngine()
        report = engine.analyze(B2B_SAAS)
        debug = val_engine.validate_report(report)

        cs = debug.confidence_summary
        assert cs.overall_confidence >= 0.0
        assert isinstance(cs.per_dimension, dict)

    def test_debug_report_has_missing_inputs(self):
        engine = PredictronEngine()
        val_engine = ValidationEngine()
        report = engine.analyze(CONSUMER_APP)
        debug = val_engine.validate_report(report)

        mi = debug.missing_inputs
        assert isinstance(mi.missing_features, list)

    def test_debug_report_validation_results(self):
        engine = PredictronEngine()
        val_engine = ValidationEngine()
        report = engine.analyze(B2B_SAAS)
        debug = val_engine.validate_report(report)

        assert "pipeline" in debug.validation_results
        assert "consistency" in debug.validation_results
        assert "completeness" in debug.validation_results
        assert "report" in debug.validation_results

    def test_debug_report_key_artifacts(self):
        engine = PredictronEngine()
        val_engine = ValidationEngine()
        report = engine.analyze(B2B_SAAS)
        debug = val_engine.validate_report(report)

        assert len(debug.key_artifacts) > 0

    def test_all_startups_validate(self):
        engine = PredictronEngine()
        val_engine = ValidationEngine()
        for startup in ALL_STARTUPS:
            report = engine.analyze(startup)
            debug = val_engine.validate_report(report)
            assert debug.pipeline_summary.startup_name != ""
            assert debug.trace_graph is not None
            assert len(debug.stage_outputs) > 0

    def test_empty_pipeline_validation(self):
        from predictron_engine.models.extracted_features import ExtractedFeatures
        from predictron_engine.models.startup import Startup

        val_engine = ValidationEngine()
        startup = Startup(
            name="Empty", website="https://empty.com", description="Empty"
        )
        features = ExtractedFeatures(data_completeness=0.0)
        debug = val_engine.validate(
            startup=startup,
            features=features,
            evidence=[],
            observations=[],
            assessments=[],
            scores=[],
            recommendations=[],
            confidence=[],
        )
        assert debug.warning_count > 0

    def test_validation_never_modifies_report(self):
        engine = PredictronEngine()
        val_engine = ValidationEngine()
        report = engine.analyze(B2B_SAAS)

        original_score = report.overall_score
        original_rec_count = len(report.recommendations)
        original_obs_count = len(report.observations)

        val_engine.validate_report(report)

        assert report.overall_score == original_score
        assert len(report.recommendations) == original_rec_count
        assert len(report.observations) == original_obs_count

    def test_trace_graph_connectivity(self):
        engine = PredictronEngine()
        val_engine = ValidationEngine()
        report = engine.analyze(FINTECH)
        debug = val_engine.validate_report(report)

        graph = debug.trace_graph
        assert graph is not None
        for node_id, node in graph.nodes.items():
            assert node.artifact_id == node_id

    def test_trace_paths_from_recommendation(self):
        engine = PredictronEngine()
        val_engine = ValidationEngine()
        report = engine.analyze(ROBOTICS)
        debug = val_engine.validate_report(report)

        graph = debug.trace_graph
        rec_nodes = [
            nid for nid, n in graph.nodes.items()
            if n.artifact_type == "Recommendation"
        ]
        if rec_nodes:
            builder = val_engine._tracer
            paths = builder.get_trace_paths(graph, rec_nodes[0])
            assert len(paths) >= 1
            assert paths[0].target_id == rec_nodes[0]

    def test_debug_report_property_counts(self):
        engine = PredictronEngine()
        val_engine = ValidationEngine()
        report = engine.analyze(B2B_SAAS)
        debug = val_engine.validate_report(report)

        _ = debug.has_errors
        _ = debug.warning_count
        _ = debug.error_count
        _ = debug.info_count

    def test_validation_engine_default_constructors(self):
        val_engine = ValidationEngine()
        assert val_engine._pipeline is not None
        assert val_engine._consistency is not None
        assert val_engine._completeness is not None
        assert val_engine._report is not None
        assert val_engine._explainer is not None
        assert val_engine._tracer is not None
