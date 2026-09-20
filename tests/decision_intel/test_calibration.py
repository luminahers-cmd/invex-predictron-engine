"""Tests for CalibrationLayer — benchmark-driven confidence adjustment."""

from __future__ import annotations

import pytest

from predictron_engine.decision.calibration_layer import (
    CalibrationLayer,
    CalibrationPoint,
    calibrate_confidence,
)

DELTA_CASES = [
    (0.80, 0.80),  # aligned
    (0.80, 0.60),  # underconfident (actual low, boost)
    (0.60, 0.80),  # overconfident (reduce)
    (0.90, 0.50),  # large gap
    (0.50, 0.50),  # aligned at midpoint
    (1.00, 1.00),  # full confidence aligned
    (0.00, 0.00),  # zero aligned
    (0.90, 1.00),  # actual higher than expected
]


class TestComputeAdjustment:
    @pytest.mark.parametrize("expected,actual", DELTA_CASES)
    def test_adjustment_clamped(
        self,
        expected: float,
        actual: float,
    ) -> None:
        layer = CalibrationLayer()
        adjustment = layer.compute_adjustment(
            expected_confidence=expected,
            actual_confidence=actual,
            use_history=False,
        )
        assert -0.20 <= adjustment.adjustment_applied <= 0.20

    @pytest.mark.parametrize("expected,actual", DELTA_CASES)
    def test_delta_matches(
        self,
        expected: float,
        actual: float,
    ) -> None:
        layer = CalibrationLayer()
        adjustment = layer.compute_adjustment(
            expected_confidence=expected,
            actual_confidence=actual,
            use_history=False,
        )
        assert adjustment.calibration_delta == round(expected - actual, 4)

    @pytest.mark.parametrize("expected,actual", DELTA_CASES)
    def test_expected_and_actual_preserved(
        self,
        expected: float,
        actual: float,
    ) -> None:
        layer = CalibrationLayer()
        adjustment = layer.compute_adjustment(
            expected_confidence=expected,
            actual_confidence=actual,
            use_history=False,
        )
        assert adjustment.expected_confidence == expected
        assert adjustment.actual_confidence == actual

    def test_aligned_confidence_small_adjustment(self) -> None:
        layer = CalibrationLayer()
        adjustment = layer.compute_adjustment(
            expected_confidence=0.8,
            actual_confidence=0.8,
            use_history=False,
        )
        assert abs(adjustment.adjustment_applied) < 0.01

    def test_underconfident_gets_boost(self) -> None:
        layer = CalibrationLayer()
        adjustment = layer.compute_adjustment(
            expected_confidence=0.8,
            actual_confidence=0.6,
            use_history=False,
        )
        assert adjustment.adjustment_applied > 0.0

    def test_overconfident_gets_reduction(self) -> None:
        layer = CalibrationLayer()
        adjustment = layer.compute_adjustment(
            expected_confidence=0.6,
            actual_confidence=0.8,
            use_history=False,
        )
        assert adjustment.adjustment_applied < 0.0


class TestHistoricalBlending:
    def test_no_history_uses_raw_delta(self) -> None:
        layer = CalibrationLayer()
        adj_no_hist = layer.compute_adjustment(
            expected_confidence=0.8,
            actual_confidence=0.6,
            use_history=True,
        )
        adj_no_use = layer.compute_adjustment(
            expected_confidence=0.8,
            actual_confidence=0.6,
            use_history=False,
        )
        # With no history both branches are identical.
        assert adj_no_hist.adjustment_applied == adj_no_use.adjustment_applied

    def test_history_alters_adjustment(self) -> None:
        layer = CalibrationLayer()
        layer.add_historical_point(CalibrationPoint(
            benchmark_case_id="hist", expected_confidence=0.8, actual_confidence=0.5,
        ))
        adj_with = layer.compute_adjustment(
            expected_confidence=0.8, actual_confidence=0.6, use_history=True,
        )
        adj_without = layer.compute_adjustment(
            expected_confidence=0.8, actual_confidence=0.6, use_history=False,
        )
        assert adj_with.adjustment_applied != adj_without.adjustment_applied

    def test_historical_calibration_list_populated(self) -> None:
        layer = CalibrationLayer()
        layer.add_historical_point(CalibrationPoint(
            benchmark_case_id="h1", expected_confidence=0.8, actual_confidence=0.5,
        ))
        adjustment = layer.compute_adjustment(
            expected_confidence=0.8, actual_confidence=0.6,
        )
        assert len(adjustment.historical_calibration) == 1

    @pytest.mark.parametrize("expected,actual", DELTA_CASES[:4])
    def test_rationale_references_confidence(
        self,
        expected: float,
        actual: float,
    ) -> None:
        layer = CalibrationLayer()
        adjustment = layer.compute_adjustment(
            expected_confidence=expected,
            actual_confidence=actual,
        )
        assert f"{expected:.2f}" in adjustment.adjustment_rationale
        assert f"{actual:.2f}" in adjustment.adjustment_rationale


class TestCalibrationPoint:
    def test_delta_property(self) -> None:
        point = CalibrationPoint(
            benchmark_case_id="c", expected_confidence=0.8, actual_confidence=0.6,
        )
        assert point.delta == -0.2

    def test_to_dict(self) -> None:
        point = CalibrationPoint(
            benchmark_case_id="c",
            expected_confidence=0.8,
            actual_confidence=0.6,
            timestamp="2025-01-01",
        )
        data = point.to_dict()
        assert data["benchmark_case_id"] == "c"
        assert data["delta"] == -0.2
        assert data["timestamp"] == "2025-01-01"


class TestHistoryManagement:
    def test_add_historical_point(self) -> None:
        layer = CalibrationLayer()
        layer.add_historical_point(CalibrationPoint(
            benchmark_case_id="c", expected_confidence=0.5, actual_confidence=0.5,
        ))
        assert len(layer.historical_points) == 1

    def test_add_historical_points(self) -> None:
        layer = CalibrationLayer()
        layer.add_historical_points([
            CalibrationPoint(benchmark_case_id="a", expected_confidence=0.5, actual_confidence=0.5),
            CalibrationPoint(benchmark_case_id="b", expected_confidence=0.5, actual_confidence=0.6),
        ])
        assert len(layer.historical_points) == 2

    def test_initial_history_empty(self) -> None:
        layer = CalibrationLayer()
        assert layer.historical_points == []

    def test_history_preserves_order(self) -> None:
        layer = CalibrationLayer()
        layer.add_historical_point(CalibrationPoint(
            benchmark_case_id="first", expected_confidence=0.5, actual_confidence=0.5,
        ))
        layer.add_historical_point(CalibrationPoint(
            benchmark_case_id="second", expected_confidence=0.5, actual_confidence=0.5,
        ))
        ids = [p.benchmark_case_id for p in layer.historical_points]
        assert ids == ["first", "second"]


class TestComputeAdjustedConfidence:
    @pytest.mark.parametrize("base,adj,expected", [
        (0.5, 0.1, 0.6),
        (0.8, -0.2, 0.6),
        (0.95, 0.2, 1.0),
        (0.05, -0.2, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.5, 1.0),
    ])
    def test_adjusted_confidence(
        self,
        base: float,
        adj: float,
        expected: float,
    ) -> None:
        layer = CalibrationLayer()
        assert layer.compute_adjusted_confidence(base, adj) == expected

    @pytest.mark.parametrize("iterations", [1, 2, 3])
    def test_deterministic(self, iterations: int) -> None:
        layer = CalibrationLayer()
        for _ in range(iterations):
            assert layer.compute_adjusted_confidence(0.6, 0.1) == 0.7


class TestCalibrationError:
    def test_no_points(self) -> None:
        layer = CalibrationLayer()
        stats = layer.compute_calibration_error()
        assert stats["count"] == 0

    def test_with_points(self, calibration_points: list[CalibrationPoint]) -> None:
        layer = CalibrationLayer()
        layer.add_historical_points(calibration_points)
        stats = layer.compute_calibration_error()
        assert stats["count"] == 3
        assert stats["mean_absolute_delta"] > 0.0

    def test_explicit_points_override(self, calibration_points: list[CalibrationPoint]) -> None:
        layer = CalibrationLayer()
        stats = layer.compute_calibration_error(points=calibration_points)
        assert stats["count"] == 3

    @pytest.mark.parametrize("count", [1, 2, 3, 10])
    def test_deterministic_stats(self, count: int) -> None:
        layer = CalibrationLayer()
        for i in range(count):
            layer.add_historical_point(CalibrationPoint(
                benchmark_case_id=f"c{i}",
                expected_confidence=0.5,
                actual_confidence=0.5 + (i % 3) * 0.1,
            ))
        stats = layer.compute_calibration_error()
        assert stats["count"] == count


class TestSummary:
    def test_no_adjustments(self) -> None:
        layer = CalibrationLayer()
        summary = layer.build_calibration_summary([])
        assert summary["adjustment_count"] == 0

    def test_with_one_adjustment(self) -> None:
        layer = CalibrationLayer()
        adj = layer.compute_adjustment(
            expected_confidence=0.8, actual_confidence=0.6, use_history=False,
        )
        summary = layer.build_calibration_summary([adj])
        assert summary["adjustment_count"] == 1
        assert summary["mean_delta"] == 0.2


class TestVersionReport:
    def test_version_report(self) -> None:
        layer = CalibrationLayer()
        report = layer.to_version_report()
        assert "engine_version" in report
        assert "history_count" in report
        assert "calibration_stats" in report


class TestCalibrateConfidence:
    def test_functional_form(self) -> None:
        adjustment = calibrate_confidence(
            expected_confidence=0.8,
            actual_confidence=0.6,
        )
        assert adjustment.adjustment_applied > 0.0

    def test_with_history(self) -> None:
        adjustment = calibrate_confidence(
            expected_confidence=0.8,
            actual_confidence=0.6,
            historical_points=[
                CalibrationPoint(
                    benchmark_case_id="h", expected_confidence=0.8,
                    actual_confidence=0.5,
                ),
            ],
        )
        assert adjustment.historical_calibration

    def test_disabled_history(self) -> None:
        adjustment = calibrate_confidence(
            expected_confidence=0.8,
            actual_confidence=0.6,
            historical_points=[
                CalibrationPoint(
                    benchmark_case_id="h", expected_confidence=0.8,
                    actual_confidence=0.5,
                ),
            ],
            use_history=False,
        )
        assert adjustment.historical_calibration == []

    @pytest.mark.parametrize("expected,actual", DELTA_CASES)
    def test_parametrized_functional_form(
        self,
        expected: float,
        actual: float,
    ) -> None:
        adjustment = calibrate_confidence(
            expected_confidence=expected,
            actual_confidence=actual,
            use_history=False,
        )
        assert adjustment.calibration_delta == round(expected - actual, 4)
