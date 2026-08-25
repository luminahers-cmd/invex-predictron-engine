"""Backward-compatibility and pipeline-integration tests (Sprint 6B)."""

from predictron_engine.decision.calibration import compute_decision_confidence
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import Recommendation, Report
from predictron_engine.models.startup import Startup
from tests.engine.test_decision.conftest import (
    make_assessment,
    make_bundle,
    make_document,
    make_observation,
)


class TestModelBackwardCompatibility:
    def test_recommendation_constructs_without_new_fields(self):
        rec = Recommendation(category="risk", action="Mitigate churn")
        assert rec.expected_confidence is None
        assert rec.expected_uncertainty is None
        assert rec.recommended_action == ""

    def test_report_defaults_calibration_to_none(self):
        report = Report(
            startup=Startup(
                name="TestCo",
                website="https://example.com",
                description="A test company",
            ),
            features=ExtractedFeatures(),
        )
        assert report.decision_confidence is None
        assert report.calibration_summary is None


class TestCalibrationReusesExistingOutputs:
    """The calibration layer must consume existing artifacts only."""

    def test_accepts_empty_bundle_and_no_signals(self):
        dc = compute_decision_confidence(
            bundle=None,
            observations=[],
            assessments=[],
            features=ExtractedFeatures(),
        )
        assert dc.confidence == 0.0
        assert dc.uncertainty_score >= 0.0

    def test_trust_comes_from_bundle_not_recomputation(self):
        bundle = make_bundle(make_document(trust=0.4))
        before = compute_decision_confidence(
            bundle=bundle,
            observations=[make_observation()],
            assessments=[make_assessment()],
            features=ExtractedFeatures(),
        )
        # Mutating the source document must not change the already
        # computed calibration — inputs are read, not re-derived.
        bundle.documents[0].metadata.trust_score.overall = 0.99
        after = compute_decision_confidence(
            bundle=bundle,
            observations=[make_observation()],
            assessments=[make_assessment()],
            features=ExtractedFeatures(),
        )
        trust_before = before.breakdown.value_of("evidence_trust")
        trust_after = after.breakdown.value_of("evidence_trust")
        assert trust_after > 0.95  # recomputed per call, from the bundle
        assert trust_before is not None and trust_before < 0.5


class TestPackageSurface:
    def test_public_api_importable_from_package(self):
        from predictron_engine import decision

        for name in (
            "DecisionConfidence",
            "ConfidenceBreakdown",
            "CalibrationSummary",
            "ConfidenceLevel",
            "compute_decision_confidence",
            "classify_confidence_level",
            "apply_recommendation_risk",
            "build_calibration_summary",
        ):
            assert hasattr(decision, name), name

    def test_report_builder_signature_unchanged(self):
        import inspect

        from predictron_engine.report.report_builder import (
            DefaultReportBuilder,
        )

        params = list(inspect.signature(DefaultReportBuilder.build).parameters)
        assert params[-1] == "evidence_collection"
