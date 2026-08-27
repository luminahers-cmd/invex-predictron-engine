"""Tests for Sprint 8 — Calibration Improvements.

Validates Expected Calibration Error computation, overconfidence
detection, and calibration report generation.
"""

from __future__ import annotations

from predictron_engine.decision.calibration_sprint8 import (
    CalibrationBin,
    build_calibration_report,
    compute_calibration_error,
    detect_overconfidence,
    validate_pipeline_confidence,
)


class TestComputeCalibrationError:
    """Tests for compute_calibration_error function."""

    def test_perfect_calibration(self) -> None:
        predictions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        outcomes = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        ece, max_ce, bins = compute_calibration_error(predictions, outcomes)
        assert ece >= 0.0
        assert max_ce >= 0.0
        assert len(bins) == 10

    def test_empty_inputs(self) -> None:
        ece, max_ce, bins = compute_calibration_error([], [])
        assert ece == 0.0
        assert max_ce == 0.0
        assert bins == ()

    def test_overconfident_case(self) -> None:
        predictions = [0.9, 0.95, 0.85]
        outcomes = [0, 0, 0]
        ece, max_ce, _ = compute_calibration_error(predictions, outcomes)
        assert ece >= 0.0
        assert max_ce >= 0.5

    def test_underconfident_case(self) -> None:
        predictions = [0.2, 0.25, 0.15]
        outcomes = [1, 1, 1]
        ece, max_ce, _ = compute_calibration_error(predictions, outcomes)
        assert ece >= 0.0
        assert max_ce >= 0.5

    def test_deterministic(self) -> None:
        predictions = [0.1, 0.2, 0.9, 1.0, 0.5, 0.6]
        outcomes = [0, 0, 1, 1, 0, 1]
        ece1, _, _ = compute_calibration_error(predictions, outcomes)
        ece2, _, _ = compute_calibration_error(predictions, outcomes)
        assert ece1 == ece2

    def test_bins_have_correct_counts(self) -> None:
        predictions = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        outcomes = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        _, _, bins = compute_calibration_error(predictions, outcomes)
        total_in_bins = sum(b.sample_count for b in bins)
        assert total_in_bins == 10

    def test_bin_properties(self) -> None:
        predictions = [0.95, 1.0]
        outcomes = [1, 1]
        _, _, bins = compute_calibration_error(predictions, outcomes)
        last = bins[-1]
        assert last.bin_lower == 0.9
        assert last.bin_upper == 1.0
        assert last.sample_count == 2


class TestDetectOverconfidence:
    """Tests for detect_overconfidence function."""

    def test_no_overconfidence(self) -> None:
        bins = (
            CalibrationBin(
                bin_lower=0.0, bin_upper=0.5, sample_count=10,
                mean_predicted_confidence=0.25, actual_accuracy=0.23,
                overconfident=False,
            ),
            CalibrationBin(
                bin_lower=0.5, bin_upper=1.0, sample_count=10,
                mean_predicted_confidence=0.75, actual_accuracy=0.72,
                overconfident=False,
            ),
        )
        detected, oc_bins = detect_overconfidence(bins)
        assert detected is False
        assert len(oc_bins) == 0

    def test_overconfidence_detected(self) -> None:
        bins = (
            CalibrationBin(
                bin_lower=0.0, bin_upper=0.5, sample_count=10,
                mean_predicted_confidence=0.45, actual_accuracy=0.1,
                overconfident=True,
            ),
            CalibrationBin(
                bin_lower=0.5, bin_upper=1.0, sample_count=10,
                mean_predicted_confidence=0.8, actual_accuracy=0.7,
                overconfident=False,
            ),
        )
        detected, oc_bins = detect_overconfidence(bins)
        assert detected is True
        assert len(oc_bins) == 1

    def test_overconfidence_from_compute(self) -> None:
        predictions = [0.9, 0.95, 0.4]
        outcomes = [1, 1, 0]
        _, _, bins = compute_calibration_error(predictions, outcomes)
        detected, _ = detect_overconfidence(bins)
        assert detected is True


class TestBuildCalibrationReport:
    """Tests for build_calibration_report function."""

    def test_empty_data(self) -> None:
        report = build_calibration_report([], [])
        assert report.calibration_quality == "insufficient_data"

    def test_basic_report(self) -> None:
        report = build_calibration_report(
            [0.5, 0.6, 0.5, 0.6, 0.5],
            [0.5, 0.6, 0.5, 0.6, 0.5],
        )
        assert report.calibration_quality != "insufficient_data"
        assert report.expected_calibration_error >= 0.0
        assert report.total_samples == 5

    def test_report_serialization(self) -> None:
        report = build_calibration_report([0.5], [0.6])
        d = report.to_dict()
        assert "expected_calibration_error" in d
        assert "overconfidence_detected" in d
        assert "calibration_quality" in d
        assert isinstance(d["bins"], list)

    def test_heuristic_outcomes_used_when_no_ground_truth(self) -> None:
        report = build_calibration_report(
            [0.8, 0.3, 0.9, 0.2],
            [0.8, 0.3, 0.9, 0.2],
        )
        assert report.calibration_quality != "insufficient_data"
        assert report.total_samples == 4

    def test_ground_truth_used_when_provided(self) -> None:
        report = build_calibration_report(
            [0.8, 0.3], [0.9, 0.2], correct_predictions=[1, 0],
        )
        assert report.calibration_quality != "insufficient_data"
        assert report.total_samples == 2


class TestValidatePipelineConfidence:
    """Tests for validate_pipeline_confidence function."""

    def test_empty_results(self) -> None:
        report = validate_pipeline_confidence([])
        assert report.calibration_quality == "insufficient_data"

    def test_basic_validation(self) -> None:
        results = [
            {"predicted_confidence": 0.8, "correct": 1},
            {"predicted_confidence": 0.3, "correct": 0},
            {"predicted_confidence": 0.7, "correct": 1},
        ]
        report = validate_pipeline_confidence(results)
        assert report.calibration_quality != "insufficient_data"
        assert report.expected_calibration_error >= 0.0
        assert report.total_samples == 3

    def test_deterministic(self) -> None:
        results = [
            {"predicted_confidence": 0.5, "correct": 1},
            {"predicted_confidence": 0.6, "correct": 0},
        ]
        r1 = validate_pipeline_confidence(results)
        r2 = validate_pipeline_confidence(results)
        assert r1.expected_calibration_error == r2.expected_calibration_error
