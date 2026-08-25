"""Integration tests for the Sprint 6C synthesis engine and pipeline."""

from predictron_engine.engine import PredictronEngine
from predictron_engine.models.report import DecisionSynthesis
from predictron_engine.synthesis.engine import DecisionSynthesisEngine
from tests.engine.test_synthesis.conftest import (
    make_assessment,
    make_confidence,
    make_decision,
    make_features,
    make_observation,
    make_readiness,
    make_recommendation,
    make_score,
)


class TestSynthesisEngineComposition:
    """Unit-level composition of DecisionSynthesisEngine.synthesize."""

    def _synthesize(self, **overrides):
        kwargs = {
            "features": make_features(),
            "observations": [
                make_observation("market_opportunity", trust_score=0.8)
            ],
            "assessments": [make_assessment("market_opportunity", confidence=0.85)],
            "scores": [make_score("market_opportunity", 78.0)],
            "recommendations": [make_recommendation("Request the pitch deck.")],
            "readiness": make_readiness(gaps=["No churn data"]),
            "decision": make_decision(),
            "decision_confidence": make_confidence(),
        }
        kwargs.update(overrides)
        return DecisionSynthesisEngine().synthesize(**kwargs)

    def test_returns_decision_synthesis_container(self):
        synthesis = self._synthesize()
        assert isinstance(synthesis, DecisionSynthesis)
        assert synthesis.executive_summary
        assert synthesis.executive_summary_key_points

    def test_prioritizes_recommendations_with_ranks(self):
        synthesis = self._synthesize(
            recommendations=[
                make_recommendation(f"Action {index}.") for index in range(4)
            ]
        )
        ranks = [rec.rank for rec in synthesis.prioritized_recommendations]
        assert ranks == [1, 2, 3, 4]

    def test_confidence_is_passthrough_never_recomputed(self):
        confidence = make_confidence(value=0.42)
        synthesis = self._synthesize(
            decision_confidence=confidence,
            calibration_summary=None,
        )
        assert synthesis.overall_confidence == confidence.confidence
        assert synthesis.confidence_level == confidence.level.value
        assert synthesis.recommended_action == confidence.recommended_action

    def test_calibration_summary_action_wins_when_present(self):
        from predictron_engine.decision.models import (
            CalibrationSummary,
            ConfidenceLevel,
        )

        summary = CalibrationSummary(
            confidence=0.5,
            level=ConfidenceLevel.MEDIUM,
            uncertainty_score=0.5,
            recommended_action="Summary action.",
        )
        synthesis = self._synthesize(calibration_summary=summary)
        assert synthesis.recommended_action == "Summary action."

    def test_none_inputs_degrade_gracefully(self):
        synthesis = self._synthesize(
            observations=[],
            assessments=[],
            scores=[],
            recommendations=[],
            readiness=None,
            decision=None,
            decision_confidence=None,
        )
        assert isinstance(synthesis, DecisionSynthesis)
        assert "no investment decision" in synthesis.executive_summary
        assert synthesis.prioritized_recommendations == []
        assert synthesis.overall_confidence == 0.0


class TestPipelineIntegration:
    """Full-pipeline attachment through PredictronEngine.analyze."""

    def test_report_carries_synthesis(self, engine, saas_request):
        report = engine.analyze(saas_request)
        assert report.decision_synthesis is not None
        assert isinstance(report.decision_synthesis, DecisionSynthesis)

    def test_pipeline_stages_unchanged(self, engine, full_request):
        report = engine.analyze(full_request)
        stages = report.analysis_metadata.pipeline_stages_completed
        assert len(stages) == 12
        assert stages[-1] == "build_report"

    def test_confidence_passthrough_from_report(self, engine, saas_request):
        report = engine.analyze(saas_request)
        assert report.decision_confidence is not None
        synthesis = report.decision_synthesis
        assert synthesis.overall_confidence == report.decision_confidence.confidence
        assert synthesis.confidence_level == report.decision_confidence.level.value
        assert (
            synthesis.uncertainty_score
            == report.decision_confidence.uncertainty_score
        )

    def test_prioritized_subset_of_report_recommendations(self, engine, saas_request):
        report = engine.analyze(saas_request)
        original_actions = {rec.action for rec in report.recommendations}
        for rec in report.decision_synthesis.prioritized_recommendations:
            assert rec.action in original_actions
            assert rec.rank is not None
        original_ranks = [rec.rank for rec in report.recommendations]
        assert all(rank is None for rank in original_ranks)

    def test_full_pipeline_determinism(self, saas_request):
        one = PredictronEngine().analyze(saas_request).decision_synthesis
        two = PredictronEngine().analyze(saas_request).decision_synthesis
        assert one.model_dump() == two.model_dump()

    def test_minimal_input_still_synthesizes(self, engine, minimal_request):
        report = engine.analyze(minimal_request)
        assert report.decision_synthesis is not None


class TestBackwardCompatibility:
    def test_report_without_synthesis_serializes(self):
        from predictron_engine.report.report_builder import DefaultReportBuilder
        from tests.engine.conftest import sample_startup  # noqa: F401
        from tests.engine.test_synthesis.conftest import make_features

        startup = None
        try:
            from predictron_engine.models.startup import Startup

            startup = Startup(
                name="CompatCo",
                website="https://compatco.example.com",
                description="A compatibility startup.",
                pitch_deck_url=None,
                founder_linkedin_urls=[],
                raw_data={},
            )
        except Exception:  # pragma: no cover - defensive
            raise AssertionError("Startup model construction failed")

        report = DefaultReportBuilder().build(
            startup,
            make_features(),
            [],
            [],
            [],
            [],
            [],
            [],
            None,
            None,
        )
        payload = report.model_dump()
        assert payload["decision_synthesis"] is None
        assert payload["decision_confidence"] is None

    def test_rank_defaults_to_none(self):
        recommendation = make_recommendation()
        assert recommendation.rank is None

    def test_new_fields_are_additive_on_existing_models(self):
        fields = set(DecisionSynthesis.model_fields)
        expected = {
            "executive_summary",
            "executive_summary_key_points",
            "prioritized_recommendations",
            "trade_offs",
            "scenarios",
            "risks",
            "opportunities",
            "overall_confidence",
            "uncertainty_score",
            "confidence_level",
            "recommended_action",
        }
        assert expected <= fields
