"""Calibration Improvements — confidence validation and overconfidence detection (Sprint 8).

Extends the Sprint 6B calibration with:
  - Confidence validation against historical outcomes
  - Calibration error reporting (Expected Calibration Error)
  - Overconfidence detection
  - Reliability diagram data

Key principles:
  - All functions are pure and deterministic
  - Calibration error is computed from binned predictions
  - Overconfidence is detected when predicted confidence exceeds
    actual accuracy in a bin
  - No external dependencies

Public API:
  - :func:`compute_calibration_error` — Expected Calibration Error (ECE)
  - :func:`detect_overconfidence` — detect when confidence is too high
  - :func:`build_calibration_report` — complete calibration validation report
  - :class:`CalibrationReport` — calibration validation container
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration (deterministic constants)
# ---------------------------------------------------------------------------

_NUM_BINS: int = 10
_OVERCONFIDENCE_THRESHOLD: float = 0.15
_CALIBRATION_WARN_THRESHOLD: float = 0.10


@dataclass(frozen=True)
class CalibrationBin:
    """One bin in the reliability diagram.

    Each bin covers a range of predicted confidence values. The bin
    tracks the mean predicted confidence and the actual accuracy
    (fraction of correct predictions).
    """

    bin_lower: float
    bin_upper: float
    sample_count: int
    mean_predicted_confidence: float
    actual_accuracy: float
    overconfident: bool = False

    @property
    def calibration_gap(self) -> float:
        """Gap between predicted and actual accuracy."""
        return round(self.mean_predicted_confidence - self.actual_accuracy, 4)

    def to_dict(self) -> dict:
        return {
            "bin_lower": self.bin_lower,
            "bin_upper": self.bin_upper,
            "sample_count": self.sample_count,
            "mean_predicted_confidence": self.mean_predicted_confidence,
            "actual_accuracy": self.actual_accuracy,
            "calibration_gap": self.calibration_gap,
            "overconfident": self.overconfident,
        }


@dataclass
class CalibrationReport:
    """Complete calibration validation report.

    Contains Expected Calibration Error, per-bin reliability data,
    overconfidence detection results, and overall assessment.
    """

    expected_calibration_error: float = 0.0
    maximum_calibration_error: float = 0.0
    overconfidence_detected: bool = False
    overconfident_bins: tuple[CalibrationBin, ...] = ()
    bins: tuple[CalibrationBin, ...] = ()
    total_samples: int = 0
    calibration_quality: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "expected_calibration_error": self.expected_calibration_error,
            "maximum_calibration_error": self.maximum_calibration_error,
            "overconfidence_detected": self.overconfidence_detected,
            "overconfident_bins_count": len(self.overconfident_bins),
            "bins": [b.to_dict() for b in self.bins],
            "total_samples": self.total_samples,
            "calibration_quality": self.calibration_quality,
        }


# ---------------------------------------------------------------------------
# Core calibration metrics
# ---------------------------------------------------------------------------


def compute_calibration_error(
    confidence_predictions: list[float],
    binary_outcomes: list[int],
    *,
    num_bins: int = _NUM_BINS,
) -> tuple[float, float, tuple[CalibrationBin, ...]]:
    """Compute Expected Calibration Error and build reliability diagram data.

    Parameters
    ----------
    confidence_predictions:
        Predicted confidence values in [0, 1] for each sample.
    binary_outcomes:
        Binary outcomes (1 = correct/positive, 0 = incorrect/negative).
        Must have same length as confidence_predictions.
    num_bins:
        Number of bins for the reliability diagram.

    Returns
    -------
    tuple of (ECE, max_calibration_error, bins)
    """
    if not confidence_predictions or not binary_outcomes:
        return 0.0, 0.0, ()

    n = len(confidence_predictions)
    assert len(binary_outcomes) == n, "Predictions and outcomes must have same length"

    bin_width = 1.0 / num_bins
    bins: list[CalibrationBin] = []

    total_ece = 0.0
    max_ce = 0.0

    for i in range(num_bins):
        bin_lower = i * bin_width
        bin_upper = (i + 1) * bin_width

        in_bin = [
            j for j in range(n)
            if bin_lower <= confidence_predictions[j] < bin_upper
            or (i == num_bins - 1 and confidence_predictions[j] == 1.0)
        ]

        if not in_bin:
            bins.append(
                CalibrationBin(
                    bin_lower=round(bin_lower, 4),
                    bin_upper=round(bin_upper, 4),
                    sample_count=0,
                    mean_predicted_confidence=0.0,
                    actual_accuracy=0.0,
                )
            )
            continue

        bin_size = len(in_bin)
        mean_conf = sum(confidence_predictions[j] for j in in_bin) / bin_size
        actual_acc = sum(binary_outcomes[j] for j in in_bin) / bin_size
        gap = abs(mean_conf - actual_acc)
        overconfident = mean_conf - actual_acc > _OVERCONFIDENCE_THRESHOLD

        total_ece += (bin_size / n) * gap
        max_ce = max(max_ce, gap)

        bins.append(
            CalibrationBin(
                bin_lower=round(bin_lower, 4),
                bin_upper=round(bin_upper, 4),
                sample_count=bin_size,
                mean_predicted_confidence=round(mean_conf, 4),
                actual_accuracy=round(actual_acc, 4),
                overconfident=overconfident,
            )
        )

    return round(total_ece, 4), round(max_ce, 4), tuple(bins)


def detect_overconfidence(
    bins: tuple[CalibrationBin, ...],
) -> tuple[bool, tuple[CalibrationBin, ...]]:
    """Detect overconfidence from reliability diagram bins.

    Overconfidence is detected when the predicted confidence in any bin
    exceeds the actual accuracy by more than the threshold.

    Returns
    -------
    tuple of (overconfidence_detected, overconfident_bins)
    """
    overconfident = [b for b in bins if b.overconfident]
    return len(overconfident) > 0, tuple(overconfident)


def _classify_calibration_quality(ece: float) -> str:
    """Deterministic calibration quality classification."""
    if ece <= 0.03:
        return "excellent"
    elif ece <= 0.07:
        return "good"
    elif ece <= 0.12:
        return "fair"
    elif ece <= 0.20:
        return "poor"
    else:
        return "very_poor"


# ---------------------------------------------------------------------------
# Pipeline-level calibration validation
# ---------------------------------------------------------------------------


def validate_pipeline_confidence(
    analysis_results: list[dict],
) -> CalibrationReport:
    """Validate confidence calibration across multiple analysis results.

    Each analysis result should contain:
      - "predicted_confidence": float (0-1)
      - "correct": int (1 if the decision was correct, 0 otherwise)

    Parameters
    ----------
    analysis_results:
        List of analysis result dicts with confidence and outcome data.
    """
    predictions = [
        r["predicted_confidence"] for r in analysis_results
        if "predicted_confidence" in r
    ]
    outcomes = [
        r["correct"] for r in analysis_results
        if "correct" in r
    ]

    if not predictions or not outcomes:
        return CalibrationReport(calibration_quality="insufficient_data")

    ece, max_ce, bins = compute_calibration_error(predictions, outcomes)
    overconfident, oc_bins = detect_overconfidence(bins)
    quality = _classify_calibration_quality(ece)

    return CalibrationReport(
        expected_calibration_error=ece,
        maximum_calibration_error=max_ce,
        overconfidence_detected=overconfident,
        overconfident_bins=oc_bins,
        bins=bins,
        total_samples=len(predictions),
        calibration_quality=quality,
    )


def build_calibration_report(
    confidence_values: list[float],
    evidence_completeness: list[float],
    correct_predictions: list[int] | None = None,
) -> CalibrationReport:
    """Build calibration report from analysis outputs.

    When correct_predictions is None, uses a heuristic: predictions
    with higher confidence and completeness are considered more likely
    correct (for self-assessment without ground truth).

    Parameters
    ----------
    confidence_values:
        Per-analysis overall confidence values.
    evidence_completeness:
        Per-analysis data completeness scores.
    correct_predictions:
        Optional ground truth. When None, uses heuristic proxy.
    """
    if not confidence_values:
        return CalibrationReport(calibration_quality="insufficient_data")

    if correct_predictions is None:
        correct_predictions = _heuristic_outcomes(
            confidence_values, evidence_completeness,
        )

    ece, max_ce, bins = compute_calibration_error(
        confidence_values, correct_predictions,
    )
    overconfident, oc_bins = detect_overconfidence(bins)
    quality = _classify_calibration_quality(ece)

    return CalibrationReport(
        expected_calibration_error=ece,
        maximum_calibration_error=max_ce,
        overconfidence_detected=overconfident,
        overconfident_bins=oc_bins,
        bins=bins,
        total_samples=len(confidence_values),
        calibration_quality=quality,
    )


def _heuristic_outcomes(
    confidence_values: list[float],
    evidence_completeness: list[float],
) -> list[int]:
    """Deterministic heuristic for outcome proxy when ground truth is absent.

    A prediction is considered "correct" when confidence aligns with
    data quality: high confidence with high completeness, or low
    confidence with low completeness.
    """
    outcomes: list[int] = []
    for conf, comp in zip(confidence_values, evidence_completeness):
        alignment = 1.0 - abs(conf - comp)
        outcomes.append(1 if alignment >= 0.5 else 0)
    return outcomes
