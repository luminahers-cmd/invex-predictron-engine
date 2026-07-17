"""Tests for the benchmark framework — cases, metrics, and validation.

These tests validate that the benchmark infrastructure itself works
correctly without requiring the full engine pipeline to run.
"""

from __future__ import annotations

from typing import Any

from benchmarks.benchmark_metrics import BenchmarkMetrics
from benchmarks.benchmark_runner import CaseResult
from benchmarks.benchmark_validator import (
    BenchmarkValidator,
    FindingSeverity,
)
from benchmarks.startup_cases.cases import (
    BENCHMARK_CASES,
    BenchmarkCaseMetadata,
    ExpectedOutcomes,
    get_all_case_ids,
    get_case_by_id,
    get_case_labels,
    get_cases_by_industry,
    get_cases_by_stage,
    get_industry_coverage,
    get_stage_coverage,
)


class TestCaseStructure:
    """Tests for benchmark case structure and metadata."""

    def test_all_cases_have_metadata(self) -> None:
        for case in BENCHMARK_CASES:
            assert "metadata" in case, f"Case {case['id']} missing metadata"
            assert isinstance(case["metadata"], BenchmarkCaseMetadata)

    def test_all_cases_have_expected_outcomes(self) -> None:
        for case in BENCHMARK_CASES:
            assert "expected_outcomes" in case
            assert isinstance(case["expected_outcomes"], ExpectedOutcomes)

    def test_all_cases_have_required_fields(self) -> None:
        required = ["id", "label", "metadata", "request", "expected_features", "expected_outcomes"]
        for case in BENCHMARK_CASES:
            for field in required:
                assert field in case, f"Case {case.get('id', '?')} missing '{field}'"

    def test_case_ids_are_unique(self) -> None:
        ids = [case["id"] for case in BENCHMARK_CASES]
        assert len(ids) == len(set(ids)), "Duplicate case IDs found"

    def test_metadata_industry_categories(self) -> None:
        valid_categories = {
            "AI", "SaaS", "FinTech", "Consumer", "Deep Tech",
            "Healthcare", "Robotics", "Marketplace", "Climate",
            "EdTech", "Enterprise", "Infrastructure",
        }
        for case in BENCHMARK_CASES:
            cat = case["metadata"].industry_category
            assert cat in valid_categories, (
                f"Case {case['id']} has invalid industry_category: {cat}"
            )

    def test_metadata_company_stages(self) -> None:
        valid_stages = {
            "Idea", "Pre-Seed", "Seed", "Series A",
            "Series B", "Series C", "Growth",
        }
        for case in BENCHMARK_CASES:
            stage = case["metadata"].company_stage
            assert stage in valid_stages, (
                f"Case {case['id']} has invalid company_stage: {stage}"
            )

    def test_metadata_coverage_tags_are_lists(self) -> None:
        for case in BENCHMARK_CASES:
            tags = case["metadata"].coverage_tags
            assert isinstance(tags, list)
            assert all(isinstance(t, str) for t in tags)


class TestCaseLookup:
    """Tests for case lookup helper functions."""

    def test_get_case_by_id_found(self) -> None:
        case = get_case_by_id("b2b_saas")
        assert case is not None
        assert case["id"] == "b2b_saas"

    def test_get_case_by_id_not_found(self) -> None:
        case = get_case_by_id("nonexistent")
        assert case is None

    def test_get_all_case_ids(self) -> None:
        ids = get_all_case_ids()
        assert len(ids) == len(BENCHMARK_CASES)
        assert "b2b_saas" in ids

    def test_get_case_labels(self) -> None:
        labels = get_case_labels()
        assert len(labels) == len(BENCHMARK_CASES)
        assert all(isinstance(v, str) for v in labels.values())

    def test_get_cases_by_industry(self) -> None:
        saas_cases = get_cases_by_industry("SaaS")
        assert len(saas_cases) >= 1
        for case in saas_cases:
            assert case["metadata"].industry_category == "SaaS"

    def test_get_cases_by_stage(self) -> None:
        seed_cases = get_cases_by_stage("Seed")
        assert len(seed_cases) >= 1
        for case in seed_cases:
            assert case["metadata"].company_stage == "Seed"

    def test_get_industry_coverage(self) -> None:
        coverage = get_industry_coverage()
        assert isinstance(coverage, dict)
        assert sum(coverage.values()) == len(BENCHMARK_CASES)

    def test_get_stage_coverage(self) -> None:
        coverage = get_stage_coverage()
        assert isinstance(coverage, dict)
        assert sum(coverage.values()) == len(BENCHMARK_CASES)


class TestExpectedOutcomes:
    """Tests for expected outcome specifications."""

    def test_score_ranges_are_valid(self) -> None:
        for case in BENCHMARK_CASES:
            eo = case["expected_outcomes"]
            assert eo.expected_min_score >= 0.0
            assert eo.expected_max_score <= 100.0
            assert eo.expected_min_score <= eo.expected_max_score

    def test_confidence_ranges_are_valid(self) -> None:
        for case in BENCHMARK_CASES:
            eo = case["expected_outcomes"]
            assert 0.0 <= eo.expected_min_confidence <= 1.0
            assert 0.0 <= eo.expected_max_confidence <= 1.0
            assert eo.expected_min_confidence <= eo.expected_max_confidence

    def test_non_negative_counts(self) -> None:
        for case in BENCHMARK_CASES:
            eo = case["expected_outcomes"]
            assert eo.expected_min_observations >= 0
            assert eo.expected_min_recommendations >= 0
            assert eo.expected_min_evidence >= 0


class TestBenchmarkMetrics:
    """Tests for the BenchmarkMetrics computation engine."""

    def test_metrics_with_empty_results(self) -> None:
        metrics = BenchmarkMetrics()
        report = metrics.compute([])
        assert report.total_cases == 0
        assert report.successful_cases == 0
        assert len(report.metrics) > 0

    def test_metrics_with_successful_results(self) -> None:
        result = CaseResult(
            case_id="test",
            case_label="Test Case",
            request={},
            success=True,
            processing_time_ms=10.0,
            report=_make_mock_report(),
            stage_timings={"total": 10.0},
        )
        metrics = BenchmarkMetrics()
        report = metrics.compute([result])
        assert report.total_cases == 1
        assert report.successful_cases == 1

        score_mean = report.get("score_mean")
        assert score_mean is not None
        assert score_mean.value > 0

        confidence_mean = report.get("confidence_mean")
        assert confidence_mean is not None
        assert confidence_mean.value > 0

    def test_metrics_with_failed_results(self) -> None:
        result = CaseResult(
            case_id="test",
            case_label="Test Case",
            request={},
            success=False,
            processing_time_ms=0.0,
            error="test error",
        )
        metrics = BenchmarkMetrics()
        report = metrics.compute([result])
        assert report.total_cases == 1
        assert report.successful_cases == 0

    def test_score_range_metric(self) -> None:
        r1 = _make_mock_report_with_score(50.0)
        r2 = _make_mock_report_with_score(80.0)
        result1 = CaseResult(
            case_id="a", case_label="A", request={}, success=True,
            processing_time_ms=1.0, report=r1,
        )
        result2 = CaseResult(
            case_id="b", case_label="B", request={}, success=True,
            processing_time_ms=1.0, report=r2,
        )
        metrics = BenchmarkMetrics()
        report = metrics.compute([result1, result2])
        score_range = report.get("score_range")
        assert score_range is not None
        assert score_range.value == 30.0

    def test_metrics_report_to_dict(self) -> None:
        metrics = BenchmarkMetrics()
        report = metrics.compute([])
        d = report.to_dict()
        assert "total_cases" in d
        assert "metrics" in d
        assert isinstance(d["metrics"], list)


class TestBenchmarkValidator:
    """Tests for the BenchmarkValidator."""

    def test_validator_passing_case(self) -> None:
        case = BENCHMARK_CASES[0]
        result = CaseResult(
            case_id=case["id"],
            case_label=case["label"],
            request=case["request"],
            success=True,
            processing_time_ms=10.0,
            report=_make_mock_report_for_case(case),
        )
        validator = BenchmarkValidator()
        vr = validator.validate_case(case, result)
        assert vr.case_id == case["id"]
        assert isinstance(vr.passed, bool)
        assert len(vr.findings) > 0

    def test_validator_failing_execution(self) -> None:
        case = BENCHMARK_CASES[0]
        result = CaseResult(
            case_id=case["id"],
            case_label=case["label"],
            request=case["request"],
            success=False,
            processing_time_ms=0.0,
            error="test error",
        )
        validator = BenchmarkValidator()
        vr = validator.validate_case(case, result)
        assert not vr.passed
        assert vr.fail_count >= 1

    def test_validator_no_expected_outcomes(self) -> None:
        case = {
            "id": "test",
            "label": "Test",
            "request": {},
            "expected_features": {},
        }
        result = CaseResult(
            case_id="test",
            case_label="Test",
            request={},
            success=True,
            processing_time_ms=1.0,
            report=_make_mock_report(),
        )
        validator = BenchmarkValidator()
        vr = validator.validate_case(case, result)
        assert vr.passed

    def test_validate_all(self) -> None:
        results = []
        for case in BENCHMARK_CASES[:3]:
            results.append(CaseResult(
                case_id=case["id"],
                case_label=case["label"],
                request=case["request"],
                success=True,
                processing_time_ms=1.0,
                report=_make_mock_report_for_case(case),
            ))
        validator = BenchmarkValidator()
        validations = validator.validate_all(BENCHMARK_CASES[:3], results)
        assert len(validations) == 3
        assert all(isinstance(v.passed, bool) for v in validations)

    def test_validate_all_missing_result(self) -> None:
        validator = BenchmarkValidator()
        validations = validator.validate_all(
            BENCHMARK_CASES[:2],
            [],
        )
        assert len(validations) == 2
        assert all(not v.passed for v in validations)

    def test_finding_to_dict(self) -> None:
        from benchmarks.benchmark_validator import ValidationFinding
        f = ValidationFinding(
            field_name="test",
            severity=FindingSeverity.PASS,
            expected="a",
            actual="a",
            message="ok",
        )
        d = f.to_dict()
        assert d["field"] == "test"
        assert d["severity"] == "pass"


class TestBenchmarkCaseCount:
    """Tests to verify benchmark suite size expectations."""

    def test_minimum_case_count(self) -> None:
        assert len(BENCHMARK_CASES) >= 10, (
            f"Benchmark suite has {len(BENCHMARK_CASES)} cases, expected >= 10"
        )

    def test_industry_diversity(self) -> None:
        coverage = get_industry_coverage()
        assert len(coverage) >= 6, (
            f"Only {len(coverage)} industry categories covered, expected >= 6"
        )

    def test_stage_diversity(self) -> None:
        coverage = get_stage_coverage()
        assert len(coverage) >= 3, (
            f"Only {len(coverage)} stages covered, expected >= 3"
        )


def _make_mock_report() -> Any:
    """Create a minimal mock Report for testing metrics/validation."""
    from unittest.mock import MagicMock

    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import (
        AnalysisMetadata,
        ConfidenceAssessment,
        DimensionAssessment,
        EvidenceItem,
        Observation,
        Recommendation,
        Report,
        ScoreResult,
    )
    from predictron_engine.models.startup import Startup

    report = MagicMock(spec=Report)
    report.overall_score = 55.0
    report.overall_confidence = 0.72
    report.features = ExtractedFeatures(
        industry="enterprise_saas",
        business_model="saas",
        customer_type="b2b",
        has_revenue=True,
        data_completeness=0.85,
    )
    report.startup = Startup(
        name="TestCo",
        website="https://testco.example.com",
        description="Test startup",
    )
    report.evidence = [
        EvidenceItem(
            domain="industry",
            category="market_context",
            statement="Test evidence",
            source="test_source",
            relevance_score=0.9,
        ),
    ]
    report.observations = [
        Observation(
            dimension="market_opportunity",
            category="market_context",
            statement="Test observation",
            confidence=0.8,
            importance=0.7,
            source_rule="test_rule",
        ),
    ]
    report.dimension_assessments = [
        DimensionAssessment(
            dimension="market_opportunity",
            summary="Good market",
            rationale="Strong indicators",
            confidence=0.8,
            score=70.0,
        ),
    ]
    report.scores = [
        ScoreResult(
            dimension="market_opportunity",
            score=70.0,
            rationale="Strong market signals",
        ),
    ]
    report.recommendations = [
        Recommendation(
            category="due_diligence",
            action="Investigate further",
            priority="high",
            title="Deep Dive",
            confidence=0.8,
        ),
    ]
    report.confidence = [
        ConfidenceAssessment(
            dimension="market_opportunity",
            confidence=0.85,
            data_completeness=0.9,
        ),
    ]
    report.analysis_metadata = AnalysisMetadata(
        engine_version="0.9.0",
        processing_time_ms=15.0,
    )
    return report


def _make_mock_report_with_score(score: float) -> Any:
    """Create a mock Report with a specific overall score."""
    report = _make_mock_report()
    report.overall_score = score
    return report


def _make_mock_report_for_case(case: dict) -> Any:
    """Create a mock Report tailored to match a benchmark case's expectations."""
    from unittest.mock import MagicMock

    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import (
        AnalysisMetadata,
        ConfidenceAssessment,
        DimensionAssessment,
        EvidenceItem,
        Observation,
        Recommendation,
        Report,
        ScoreResult,
    )
    from predictron_engine.models.startup import Startup

    ef = case.get("expected_features", {})

    report = MagicMock(spec=Report)
    report.overall_score = 60.0
    report.overall_confidence = 0.70
    report.features = ExtractedFeatures(
        industry=ef.get("industry"),
        business_model=ef.get("business_model"),
        customer_type=ef.get("customer_type"),
        has_revenue=ef.get("has_revenue"),
        data_completeness=0.85,
    )
    report.startup = Startup(
        name=case.get("request", {}).get("startup_name", "Test"),
        website="https://test.example.com",
        description="Test",
    )
    report.evidence = [
        EvidenceItem(
            domain="industry",
            category="market_context",
            statement="Evidence",
            source="test",
            relevance_score=0.8,
        ),
    ]
    report.observations = [
        Observation(
            dimension="market_opportunity",
            category="market_context",
            statement="Observation",
            confidence=0.7,
            importance=0.6,
            source_rule="rule_1",
        ),
        Observation(
            dimension="product_strength",
            category="product_assessment",
            statement="Product observation",
            confidence=0.8,
            importance=0.7,
            source_rule="rule_2",
        ),
        Observation(
            dimension="traction_signals",
            category="traction",
            statement="Traction observation",
            confidence=0.6,
            importance=0.5,
            source_rule="rule_3",
        ),
    ]
    dims = [
        "market_opportunity", "product_strength", "founder_quality",
        "traction_signals", "business_model_viability",
        "competitive_position", "team_execution",
    ]
    report.dimension_assessments = [
        DimensionAssessment(
            dimension=d, summary=f"Summary for {d}",
            rationale=f"Rationale for {d}", confidence=0.7, score=55.0,
        )
        for d in dims
    ]
    report.scores = [
        ScoreResult(
            dimension=d, score=55.0, rationale=f"Score rationale for {d}",
        )
        for d in dims
    ]
    report.recommendations = [
        Recommendation(
            category="due_diligence",
            action="Investigate",
            priority="high",
            title="Deep Dive",
            confidence=0.8,
        ),
        Recommendation(
            category="risk_mitigation",
            action="Address risks",
            priority="medium",
            title="Risk Review",
            confidence=0.7,
        ),
    ]
    report.confidence = [
        ConfidenceAssessment(
            dimension=d, confidence=0.75, data_completeness=0.85,
        )
        for d in dims
    ]
    report.analysis_metadata = AnalysisMetadata(
        engine_version="0.9.0", processing_time_ms=12.0,
    )
    return report
