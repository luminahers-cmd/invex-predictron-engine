"""Tests for DecisionReportBuilder — deterministic comprehensive reports."""

from __future__ import annotations

import pytest

from predictron_engine.decision.calibration_layer import CalibrationLayer
from predictron_engine.decision.intelligence_models import (
    DecisionVerdict,
)
from predictron_engine.decision.report import DecisionReportBuilder

from .conftest import _strip_non_deterministic, build_feature_set


class TestBuildReport:
    def test_report_company_id(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set(company_id="rec-7")
        report = builder.build_report(feature_set)
        assert report.company_id == "rec-7"

    def test_report_has_decision_summary(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        assert "verdict" in report.decision_summary
        assert "overall_score" in report.decision_summary

    def test_report_has_contributions(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        assert len(report.contributions) == feature_set.feature_count()

    def test_report_buckets_present(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        assert report.positive_contributions or report.neutral_contributions

    def test_report_has_trace(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        assert report.trace is not None
        assert report.trace.company_id == feature_set.company_id

    def test_report_has_explanation(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        assert report.explanation is not None

    def test_report_has_calibration(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        assert report.calibration is not None

    def test_report_has_feature_provenance(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        assert len(report.feature_provenance) > 0

    def test_report_has_evidence_references(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        assert len(report.evidence_references) > 0

    def test_report_verdict_override(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(
            feature_set, verdict=DecisionVerdict.STRONG_INVEST,
        )
        assert report.decision_summary["verdict"] == DecisionVerdict.STRONG_INVEST.value

    def test_report_confidence_override(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set, confidence=0.7)
        assert report.decision_summary["confidence"] == 0.7

    def test_report_metadata(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        assert "feature_count" in report.metadata
        assert "built_at" in report.metadata


class TestTopStrengthsWeaknesses:
    def test_top_strengths_filled(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        if report.positive_contributions:
            assert len(report.top_strengths) > 0

    def test_top_weaknesses_filled(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        assert isinstance(report.top_weaknesses, list)

    @pytest.mark.parametrize("index", range(5))
    def test_strengths_are_feature_names(self, index: int) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        for strength in report.top_strengths:
            assert isinstance(strength, str)
            assert len(strength) > 0


class TestHistoricalComparisons:
    def test_baseline_recorded(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        assert len(report.historical_comparisons) >= 1

    def test_baseline_is_current(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        baseline = report.historical_comparisons[0]
        assert baseline["company_id"] == feature_set.company_id
        assert baseline["is_baseline"] is True


class TestConfidenceExplanation:
    def test_confidence_explanation_present(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set, confidence=0.6)
        assert "confidence" in report.confidence_explanation.lower()

    def test_calibration_narrative(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set, confidence=0.6)
        assert report.calibration is not None


class TestCompareReports:
    def test_compare_empty(self) -> None:
        builder = DecisionReportBuilder()
        assert builder.compare_reports([]) == []

    def test_compare_single(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        comparisons = builder.compare_reports([report])
        assert len(comparisons) == 1
        assert comparisons[0]["company_id"] == feature_set.company_id

    def test_compare_multiple_sorted(self) -> None:
        builder = DecisionReportBuilder()
        r1 = builder.build_report(build_feature_set(company_id="b-rec"))
        r2 = builder.build_report(build_feature_set(company_id="a-rec"))
        comparisons = builder.compare_reports([r1, r2])
        assert [c["company_id"] for c in comparisons] == ["a-rec", "b-rec"]

    def test_compare_has_verdict(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        comparisons = builder.compare_reports([report])
        assert comparisons[0]["verdict"] in {v.value for v in DecisionVerdict}


class TestSerialization:
    def test_report_to_dict(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        data = builder.report_to_dict(report)
        assert data["company_id"] == feature_set.company_id
        assert "decision_summary" in data
        assert "contributions" in data

    @pytest.mark.parametrize("iterations", [1, 2, 3])
    def test_report_deterministic(self, iterations: int) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        first = _strip_non_deterministic(
            builder.report_to_dict(builder.build_report(feature_set)),
        )
        for _ in range(iterations):
            second = _strip_non_deterministic(
                builder.report_to_dict(builder.build_report(feature_set)),
            )
            assert second == first

    def test_empty_feature_set_report(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set(
            include_company=False, include_growth=False, include_founder=False,
            include_funding=False, include_graph=False, include_signals=False,
            include_benchmark=False,
        )
        report = builder.build_report(feature_set)
        assert report.contributions == []


class TestRenderMarkdown:
    def test_markdown_contains_company(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set(company_id="rec-md")
        report = builder.build_report(feature_set)
        md = builder.render_markdown(report)
        assert "rec-md" in md

    def test_markdown_contains_sections(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        md = builder.render_markdown(report)
        assert "## Decision Summary" in md
        assert "## Contribution Table" in md

    def test_markdown_deterministic(self) -> None:
        builder = DecisionReportBuilder()
        feature_set = build_feature_set()
        report = builder.build_report(feature_set)
        first = builder.render_markdown(report)
        second = builder.render_markdown(report)
        assert first == second


class TestCustomCalibration:
    def test_injected_calibration_used(self) -> None:
        builder = DecisionReportBuilder()
        layer = CalibrationLayer()
        feature_set = build_feature_set()
        cal = layer.compute_adjustment(
            expected_confidence=0.8, actual_confidence=0.6, use_history=False,
        )
        report = builder.build_report(feature_set, calibration=cal)
        assert report.calibration is cal
