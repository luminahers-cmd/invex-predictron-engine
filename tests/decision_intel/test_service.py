"""Tests for DecisionIntelligenceService — top-level orchestration."""

from __future__ import annotations

import pytest

from predictron_engine.decision.calibration_layer import CalibrationPoint
from predictron_engine.decision.intelligence_models import (
    DecisionIntelligenceReport,
    DecisionTrace,
    DecisionVerdict,
    Explanation,
)
from predictron_engine.decision.service import DecisionIntelligenceService

from .conftest import _strip_non_deterministic, build_feature_set


class TestProduceDecision:
    def test_returns_report(self, service: DecisionIntelligenceService) -> None:
        feature_set = build_feature_set()
        result = service.produce_decision(feature_set)
        assert isinstance(result, DecisionIntelligenceReport)

    def test_report_has_trace(self, service: DecisionIntelligenceService) -> None:
        feature_set = build_feature_set()
        report = service.produce_decision(feature_set)
        assert report.trace is not None

    def test_report_has_explanation(self, service: DecisionIntelligenceService) -> None:
        feature_set = build_feature_set()
        report = service.produce_decision(feature_set)
        assert report.explanation is not None

    def test_verdict_override(self, service: DecisionIntelligenceService) -> None:
        feature_set = build_feature_set()
        report = service.produce_decision(
            feature_set, verdict=DecisionVerdict.PASS,
        )
        assert report.decision_summary["verdict"] == DecisionVerdict.PASS.value

    def test_confidence_override(self, service: DecisionIntelligenceService) -> None:
        feature_set = build_feature_set()
        report = service.produce_decision(feature_set, confidence=0.9)
        assert report.decision_summary["confidence"] == 0.9

    def test_recommendation_override(self, service: DecisionIntelligenceService) -> None:
        feature_set = build_feature_set()
        report = service.produce_decision(
            feature_set, recommendation_text="Custom rec",
        )
        assert report.explanation is not None
        assert report.explanation.recommendation == "Custom rec"

    def test_calibration_override(self, service: DecisionIntelligenceService) -> None:
        feature_set = build_feature_set()
        cal = service.calibrate(
            expected_confidence=0.8, actual_confidence=0.7,
        )
        report = service.produce_decision(feature_set, calibration=cal)
        assert report.calibration is cal


class TestProduceTrace:
    def test_returns_trace(self, service: DecisionIntelligenceService) -> None:
        feature_set = build_feature_set()
        trace = service.produce_trace(feature_set)
        assert isinstance(trace, DecisionTrace)

    def test_trace_company_id(self, service: DecisionIntelligenceService) -> None:
        feature_set = build_feature_set(company_id="rec-x")
        trace = service.produce_trace(feature_set)
        assert trace.company_id == "rec-x"

    def test_trace_confidence(self, service: DecisionIntelligenceService) -> None:
        feature_set = build_feature_set()
        trace = service.produce_trace(feature_set, confidence=0.4)
        assert trace.confidence == 0.4


class TestProduceExplanation:
    def test_returns_explanation(self, service: DecisionIntelligenceService) -> None:
        feature_set = build_feature_set()
        explanation = service.produce_explanation(feature_set)
        assert isinstance(explanation, Explanation)

    def test_verdict_override(self, service: DecisionIntelligenceService) -> None:
        feature_set = build_feature_set()
        explanation = service.produce_explanation(
            feature_set, verdict=DecisionVerdict.WATCH,
        )
        assert "Watchlist" in explanation.headline


class TestCalibrate:
    def test_returns_adjustment(self, service: DecisionIntelligenceService) -> None:
        adjustment = service.calibrate(
            expected_confidence=0.8, actual_confidence=0.6,
        )
        assert adjustment.adjustment_applied > 0.0

    def test_case_id_propagates(self, service: DecisionIntelligenceService) -> None:
        adjustment = service.calibrate(
            expected_confidence=0.8,
            actual_confidence=0.6,
            benchmark_case_id="case-42",
        )
        assert adjustment.benchmark_case_id == "case-42"


class TestCompare:
    def test_compare_reports(self, service: DecisionIntelligenceService) -> None:
        reports = [
            service.produce_decision(build_feature_set(company_id=f"rec-{i}"))
            for i in range(3)
        ]
        comparisons = service.compare(reports)
        assert len(comparisons) == 3

    def test_compare_empty(self, service: DecisionIntelligenceService) -> None:
        assert service.compare([]) == []


class TestContributions:
    def test_compute_contributions(self, service: DecisionIntelligenceService) -> None:
        feature_set = build_feature_set()
        contributions = service.compute_contributions(feature_set)
        assert len(contributions) == feature_set.feature_count()

    def test_classify_contributions(self, service: DecisionIntelligenceService) -> None:
        feature_set = build_feature_set()
        classified = service.classify_contributions(feature_set)
        assert set(classified.keys()) == {
            "positive", "negative", "neutral", "confidence",
        }


class TestHistoricalCalibration:
    def test_add_historical_point(self, service: DecisionIntelligenceService) -> None:
        point = CalibrationPoint(
            benchmark_case_id="c", expected_confidence=0.5, actual_confidence=0.5,
        )
        service.add_historical_calibration_point(point)
        assert len(service.calibration_layer.historical_points) == 1

    def test_calibration_summary_empty(self, service: DecisionIntelligenceService) -> None:
        summary = service.calibration_summary([])
        assert summary["adjustment_count"] == 0


class TestServiceBundledDeterminism:
    @pytest.mark.parametrize("iterations", [1, 2, 3, 5])
    def test_report_build_deterministic(
        self,
        service: DecisionIntelligenceService,
        iterations: int,
    ) -> None:
        feature_set = build_feature_set()
        first = _strip_non_deterministic(service.produce_decision(feature_set).to_dict())
        for _ in range(iterations):
            assert _strip_non_deterministic(
                service.produce_decision(feature_set).to_dict(),
            ) == first

    def test_service_reuses_engines(self, service: DecisionIntelligenceService) -> None:
        assert service.contribution_engine is not None
        assert service.trace_engine is not None
        assert service.explainability_engine is not None
        assert service.calibration_layer is not None
        assert service.report_builder is not None
