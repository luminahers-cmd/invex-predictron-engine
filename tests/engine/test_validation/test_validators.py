"""Tests for all validators."""

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    ConfidenceAssessment,
    DimensionAssessment,
    EvidenceItem,
    Observation,
    Recommendation,
    ScoreResult,
)
from predictron_engine.models.startup import Startup
from predictron_engine.validation.validators.completeness_validator import (
    CompletenessValidator,
)
from predictron_engine.validation.validators.consistency_validator import (
    ConsistencyValidator,
)
from predictron_engine.validation.validators.pipeline_validator import (
    PipelineValidator,
)
from predictron_engine.validation.validators.report_validator import (
    ReportValidator,
)


def _make_startup(**kwargs) -> Startup:
    defaults = {
        "name": "TestCo",
        "website": "https://testco.example.com",
        "description": "A test startup for validation testing.",
    }
    defaults.update(kwargs)
    return Startup(**defaults)


def _make_features(**kwargs) -> ExtractedFeatures:
    defaults = {
        "industry": "fintech",
        "business_model": "saas",
        "data_completeness": 0.5,
        "founder_profile_count": 2,
    }
    defaults.update(kwargs)
    return ExtractedFeatures(**defaults)


class TestPipelineValidator:
    """Tests for PipelineValidator."""

    def test_valid_pipeline_no_findings(self):
        validator = PipelineValidator()
        findings = validator.validate(
            startup=_make_startup(),
            features=_make_features(),
            evidence=[
                EvidenceItem(
                    domain="industry",
                    category="test",
                    statement="test",
                    source="test",
                ),
            ],
            observations=[
                Observation(
                    dimension="market_opportunity",
                    category="test",
                    statement="test",
                    confidence=0.7,
                    importance=0.5,
                    source_rule="test",
                ),
            ],
            assessments=[
                DimensionAssessment(
                    dimension="market_opportunity",
                    summary="s",
                    rationale="r",
                    confidence=0.6,
                ),
            ],
            scores=[
                ScoreResult(
                    dimension="market_opportunity", score=65.0
                ),
            ],
            recommendations=[
                Recommendation(
                    category="opportunity",
                    action="test",
                    priority="medium",
                ),
            ],
            confidence=[
                ConfidenceAssessment(
                    dimension="market_opportunity",
                    confidence=0.6,
                    data_completeness=0.5,
                ),
            ],
        )
        errors = [f for f in findings if f.severity == "error"]
        assert len(errors) == 0

    def test_missing_startup_name(self):
        validator = PipelineValidator()

        class FakeStartup:
            name = ""
            website = "https://test.com"
            description = "A test startup."

        findings = validator.validate(
            startup=FakeStartup(),
            features=_make_features(),
            evidence=[],
            observations=[],
            assessments=[],
            scores=[],
            recommendations=[],
            confidence=[],
        )
        assert any(
            f.category == "missing_data" and "name" in f.message.lower()
            for f in findings
        )

    def test_missing_startup_description(self):
        validator = PipelineValidator()

        class FakeStartup:
            name = "TestCo"
            website = "https://test.com"
            description = ""

        findings = validator.validate(
            startup=FakeStartup(),
            features=_make_features(),
            evidence=[],
            observations=[],
            assessments=[],
            scores=[],
            recommendations=[],
            confidence=[],
        )
        assert any("description" in f.message.lower() for f in findings)

    def test_low_data_completeness(self):
        validator = PipelineValidator()
        findings = validator.validate(
            startup=_make_startup(),
            features=_make_features(data_completeness=0.05),
            evidence=[],
            observations=[],
            assessments=[],
            scores=[],
            recommendations=[],
            confidence=[],
        )
        assert any("completeness" in f.message.lower() for f in findings)

    def test_empty_evidence(self):
        validator = PipelineValidator()
        findings = validator.validate(
            startup=_make_startup(),
            features=_make_features(),
            evidence=[],
            observations=[],
            assessments=[],
            scores=[],
            recommendations=[],
            confidence=[],
        )
        assert any("evidence" in f.message.lower() for f in findings)

    def test_empty_observations(self):
        validator = PipelineValidator()
        findings = validator.validate(
            startup=_make_startup(),
            features=_make_features(),
            evidence=[],
            observations=[],
            assessments=[],
            scores=[],
            recommendations=[],
            confidence=[],
        )
        assert any("observation" in f.message.lower() for f in findings)

    def test_empty_recommendations_is_info(self):
        validator = PipelineValidator()
        findings = validator.validate(
            startup=_make_startup(),
            features=_make_features(),
            evidence=[],
            observations=[],
            assessments=[],
            scores=[],
            recommendations=[],
            confidence=[],
        )
        rec_findings = [
            f for f in findings
            if "recommendation" in f.message.lower()
        ]
        for f in rec_findings:
            assert f.severity == "info"


class TestConsistencyValidator:
    """Tests for ConsistencyValidator."""

    def test_no_conflicts(self):
        validator = ConsistencyValidator()
        findings = validator.validate(
            observations=[
                Observation(
                    dimension="market_opportunity",
                    category="test",
                    statement="test",
                    confidence=0.7,
                    importance=0.5,
                    source_rule="test",
                ),
            ],
            assessments=[
                DimensionAssessment(
                    dimension="market_opportunity",
                    summary="s",
                    rationale="r",
                    confidence=0.6,
                ),
            ],
            scores=[
                ScoreResult(dimension="market_opportunity", score=65.0),
            ],
            recommendations=[],
        )
        conflicts = [
            f for f in findings
            if f.category == "conflicting_observations"
        ]
        assert len(conflicts) == 0

    def test_conflicting_observations(self):
        validator = ConsistencyValidator()
        observations = [
            Observation(
                dimension="market_opportunity",
                category="test",
                statement="Strong market",
                confidence=0.9,
                importance=0.8,
                source_rule="test",
            ),
            Observation(
                dimension="market_opportunity",
                category="test",
                statement="Weak market signals",
                confidence=0.15,
                importance=0.3,
                source_rule="test",
            ),
        ]
        findings = validator.validate(
            observations=observations,
            assessments=[],
            scores=[],
            recommendations=[],
        )
        conflicts = [
            f for f in findings
            if f.category == "conflicting_observations"
        ]
        assert len(conflicts) == 1

    def test_unscored_assessment(self):
        validator = ConsistencyValidator()
        findings = validator.validate(
            observations=[],
            assessments=[
                DimensionAssessment(
                    dimension="market_opportunity",
                    summary="s",
                    rationale="r",
                    confidence=0.6,
                ),
            ],
            scores=[],
            recommendations=[],
        )
        alignment = [
            f for f in findings if f.category == "alignment"
        ]
        assert len(alignment) >= 1

    def test_unassessed_score(self):
        validator = ConsistencyValidator()
        findings = validator.validate(
            observations=[],
            assessments=[],
            scores=[
                ScoreResult(dimension="market_opportunity", score=65.0),
            ],
            recommendations=[],
        )
        alignment = [
            f for f in findings if f.category == "alignment"
        ]
        assert len(alignment) >= 1


class TestCompletenessValidator:
    """Tests for CompletenessValidator."""

    def test_no_unused_evidence(self):
        validator = CompletenessValidator()
        evidence = [
            EvidenceItem(
                domain="industry",
                category="test",
                statement="Fintech market",
                source="test",
            ),
        ]
        observations = [
            Observation(
                dimension="market_opportunity",
                category="test",
                statement="test",
                evidence=["Fintech market"],
                confidence=0.7,
                importance=0.5,
                source_rule="test",
            ),
        ]
        findings = validator.validate(
            features=_make_features(),
            evidence=evidence,
            observations=observations,
            assessments=[],
            scores=[],
            recommendations=[],
        )
        unused = [
            f for f in findings if f.category == "unused_evidence"
        ]
        assert len(unused) == 0

    def test_unused_evidence_detected(self):
        validator = CompletenessValidator()
        evidence = [
            EvidenceItem(
                domain="industry",
                category="test",
                statement="This is unused evidence",
                source="test",
            ),
        ]
        findings = validator.validate(
            features=_make_features(),
            evidence=evidence,
            observations=[],
            assessments=[],
            scores=[],
            recommendations=[],
        )
        unused = [
            f for f in findings if f.category == "unused_evidence"
        ]
        assert len(unused) == 1

    def test_unused_observations_detected(self):
        validator = CompletenessValidator()
        observations = [
            Observation(
                dimension="unknown_dim",
                category="test",
                statement="test",
                confidence=0.7,
                importance=0.5,
                source_rule="test",
            ),
        ]
        findings = validator.validate(
            features=_make_features(),
            evidence=[],
            observations=observations,
            assessments=[],
            scores=[],
            recommendations=[],
        )
        unused = [
            f for f in findings if f.category == "unused_observations"
        ]
        assert len(unused) == 1

    def test_dimension_coverage_gap(self):
        validator = CompletenessValidator()
        observations = [
            Observation(
                dimension="market_opportunity",
                category="test",
                statement="test",
                confidence=0.7,
                importance=0.5,
                source_rule="test",
            ),
        ]
        findings = validator.validate(
            features=_make_features(),
            evidence=[],
            observations=observations,
            assessments=[],
            scores=[],
            recommendations=[],
        )
        gaps = [
            f for f in findings if f.category == "coverage_gap"
        ]
        assert len(gaps) >= 1

    def test_missing_critical_features(self):
        validator = CompletenessValidator()
        features = ExtractedFeatures(data_completeness=0.1)
        findings = validator.validate(
            features=features,
            evidence=[],
            observations=[],
            assessments=[],
            scores=[],
            recommendations=[],
        )
        missing = [
            f for f in findings if f.category == "missing_input"
        ]
        assert len(missing) >= 2


class TestReportValidator:
    """Tests for ReportValidator."""

    def test_valid_report(self):
        from predictron_engine.models.report import AnalysisMetadata, Report

        report = Report(
            startup=_make_startup(),
            features=_make_features(),
            analysis_metadata=AnalysisMetadata(
                engine_version="0.6.5",
                pipeline_stages_completed=[
                    "normalize", "collect", "extract", "evidence",
                    "reason", "evaluate", "score", "recommend",
                    "confidence", "build_report",
                ],
            ),
        )
        validator = ReportValidator()
        findings = validator.validate(report)
        errors = [f for f in findings if f.severity == "error"]
        assert len(errors) == 0

    def test_missing_metadata(self):
        from predictron_engine.models.report import Report

        report = Report(
            startup=_make_startup(),
            features=_make_features(),
        )
        report.analysis_metadata = None  # type: ignore[assignment]
        validator = ReportValidator()
        findings = validator.validate(report)
        assert any(
            f.category == "missing_metadata" for f in findings
        )

    def test_outdated_version(self):
        from predictron_engine.models.report import AnalysisMetadata, Report

        report = Report(
            startup=_make_startup(),
            features=_make_features(),
            analysis_metadata=AnalysisMetadata(
                engine_version="0.1.0",
            ),
        )
        validator = ReportValidator()
        findings = validator.validate(report)
        assert any(
            "version" in f.message.lower() for f in findings
        )

    def test_empty_recommendation_action(self):
        from predictron_engine.models.report import Report

        report = Report(
            startup=_make_startup(),
            features=_make_features(),
            recommendations=[
                Recommendation(
                    category="test",
                    action="",
                    priority="medium",
                ),
            ],
        )
        validator = ReportValidator()
        findings = validator.validate(report)
        assert any(
            f.category == "missing_field" for f in findings
        )

    def test_observation_without_assessment(self):
        from predictron_engine.models.report import Report

        report = Report(
            startup=_make_startup(),
            features=_make_features(),
            observations=[
                Observation(
                    dimension="market_opportunity",
                    category="test",
                    statement="test",
                    confidence=0.7,
                    importance=0.5,
                    source_rule="test",
                ),
            ],
            dimension_assessments=[],
        )
        validator = ReportValidator()
        findings = validator.validate(report)
        assert any(
            f.category == "cross_reference" for f in findings
        )
