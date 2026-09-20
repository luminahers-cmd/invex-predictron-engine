"""Calibration Layer — consumes Benchmark Platform outputs to adjust confidence.

Adjusts confidence calibration only; never modifies feature values.

Reports for each calibration:
  - expected confidence (from benchmark ground truth)
  - actual confidence (computed by the decision layer)
  - calibration delta (expected - actual)
  - historical calibration (past deltas for trend analysis)

Consumes (without duplicating):
  - :mod:`benchmarks.ground_truth_eval.models` — ground truth outcomes
  - :mod:`benchmarks.benchmark_metrics.BenchmarkMetrics` — benchmark metrics
  - :mod:`predictron_engine.decision.calibration_sprint8` — ECE reporting

Key principles:
  - Fully deterministic adjustment
  - Only adjusts confidence, never feature values
  - No ML, no external APIs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from predictron_engine.decision.intelligence_models import CalibrationAdjustment
from predictron_engine.version import ENGINE_VERSION

__all__ = [
    "CalibrationAdjustment",
    "CalibrationLayer",
    "CalibrationPoint",
    "calibrate_confidence",
]

__all__ = [
    "CalibrationAdjustment",
    "CalibrationLayer",
    "CalibrationPoint",
    "calibrate_confidence",
]

# ---------------------------------------------------------------------------
# Calibration constants.
# ---------------------------------------------------------------------------

# Floor/ceil for adjustment — calibration never fully removes confidence.
_MIN_ADJUSTMENT = -0.20
_MAX_ADJUSTMENT = 0.20

# Confidence beyond this delta triggers a narrative warning.
_NOTABLE_DELTA = 0.10

# Blend factor for historical adjustment (exponential smoothing).
_HISTORICAL_BLEND = 0.5


@dataclass(frozen=True)
class CalibrationPoint:
    """A single historical calibration observation."""

    benchmark_case_id: str
    expected_confidence: float
    actual_confidence: float
    timestamp: str = ""

    @property
    def delta(self) -> float:
        """Actual minus expected confidence."""
        return round(self.actual_confidence - self.expected_confidence, 4)

    def to_dict(self) -> dict[str, Any]:
        """Deterministic serialization."""
        return {
            "benchmark_case_id": self.benchmark_case_id,
            "expected_confidence": self.expected_confidence,
            "actual_confidence": self.actual_confidence,
            "delta": self.delta,
            "timestamp": self.timestamp,
        }


class CalibrationLayer:
    """Adjusts decision confidence using Benchmark Platform outputs."""

    def __init__(self) -> None:
        self._history: list[CalibrationPoint] = []

    def add_historical_point(
        self,
        point: CalibrationPoint,
    ) -> None:
        """Record a historical calibration point."""
        self._history.append(point)

    def add_historical_points(
        self,
        points: list[CalibrationPoint],
    ) -> None:
        """Record multiple historical calibration points."""
        self._history.extend(points)

    @property
    def historical_points(self) -> list[CalibrationPoint]:
        """Deterministic history (insertion order preserved)."""
        return list(self._history)

    def compute_adjustment(
        self,
        *,
        expected_confidence: float,
        actual_confidence: float,
        benchmark_case_id: str = "default",
        use_history: bool = True,
    ) -> CalibrationAdjustment:
        """Compute a calibrated confidence adjustment.

        Deterministic adjustment formula:
          raw_delta = expected_confidence - actual_confidence
          adjustment = clip(raw_delta * 0.5, MIN_ADJUSTMENT, MAX_ADJUSTMENT)
          if use_history: blend with historical mean delta

        Args:
            expected_confidence: Confidence expected from benchmark (0-1).
            actual_confidence: Confidence computed by decision layer (0-1).
            benchmark_case_id: Identifier of the benchmark case.
            use_history: Whether to blend historical calibration.

        Returns:
            A CalibrationAdjustment with full narrative.
        """
        expected = self._clamp(expected_confidence)
        actual = self._clamp(actual_confidence)

        delta = round(expected - actual, 6)
        adjustment = round(self._adjustment_from_delta(delta), 6)

        historical = self._historical_dicts() if use_history else []

        if use_history and historical:
            historical_mean_delta = self._mean(
                [p["delta"] for p in historical]
            )
            adjustment = round(
                adjustment * _HISTORICAL_BLEND
                + historical_mean_delta * (1.0 - _HISTORICAL_BLEND),
                6,
            )

        adjustment = round(
            max(_MIN_ADJUSTMENT, min(_MAX_ADJUSTMENT, adjustment)),
            6,
        )

        rationale = self._build_rationale(
            expected, actual, delta, adjustment, historical,
        )

        return CalibrationAdjustment(
            benchmark_case_id=benchmark_case_id,
            expected_confidence=expected,
            actual_confidence=actual,
            calibration_delta=round(delta, 4),
            historical_calibration=historical,
            adjustment_applied=adjustment,
            adjustment_rationale=rationale,
        )

    def compute_adjusted_confidence(
        self,
        base_confidence: float,
        adjustment: float,
    ) -> float:
        """Apply a calibration adjustment to a confidence value.

        Clamps the result to [0, 1].
        """
        return round(self._clamp(base_confidence + adjustment), 4)

    def compute_calibration_error(
        self,
        points: list[CalibrationPoint] | None = None,
    ) -> dict[str, float]:
        """Compute calibration statistics over historical points.

        Returns:
          - mean_absolute_delta
          - mean_delta
          - max_absolute_delta
          - count
        """
        pts = points if points is not None else self._history
        if not pts:
            return {
                "mean_absolute_delta": 0.0,
                "mean_delta": 0.0,
                "max_absolute_delta": 0.0,
                "count": 0,
            }

        deltas = [p.delta for p in pts]
        abs_deltas = [abs(d) for d in deltas]
        return {
            "mean_absolute_delta": round(sum(abs_deltas) / len(abs_deltas), 4),
            "mean_delta": round(sum(deltas) / len(deltas), 4),
            "max_absolute_delta": round(max(abs_deltas), 4),
            "count": len(pts),
        }

    def build_calibration_summary(
        self,
        adjustments: list[CalibrationAdjustment],
    ) -> dict[str, Any]:
        """Build a summary across multiple calibration adjustments."""
        if not adjustments:
            return {
                "adjustment_count": 0,
                "mean_delta": 0.0,
                "mean_adjustment_applied": 0.0,
                "overconfident_count": 0,
                "underconfident_count": 0,
            }

        deltas = [a.calibration_delta for a in adjustments]
        applied = [a.adjustment_applied for a in adjustments]
        overconfident = sum(1 for d in deltas if d > _NOTABLE_DELTA)
        underconfident = sum(1 for d in deltas if d < -_NOTABLE_DELTA)

        return {
            "adjustment_count": len(adjustments),
            "mean_delta": round(sum(deltas) / len(deltas), 4),
            "mean_adjustment_applied": round(sum(applied) / len(applied), 4),
            "overconfident_count": overconfident,
            "underconfident_count": underconfident,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _adjustment_from_delta(delta: float) -> float:
        """Half of the delta, clamped."""
        adjusted = delta * 0.5
        return max(_MIN_ADJUSTMENT, min(_MAX_ADJUSTMENT, adjusted))

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _historical_dicts(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self._history]

    def _build_rationale(
        self,
        expected: float,
        actual: float,
        delta: float,
        adjustment: float,
        historical: list[dict[str, Any]],
    ) -> str:
        """Build a deterministic rationale for the calibration adjustment."""
        parts = [
            f"Expected confidence {expected:.2f} vs actual {actual:.2f} "
            f"(delta {delta:+.4f}).",
        ]

        if abs(delta) < 0.02:
            parts.append("Calibration is well-aligned; minimal adjustment.")
        elif delta > 0:
            parts.append(
                f"Actual confidence is {delta:.2f} below expected; "
                f"confidence boosted by {adjustment:+.4f}."
            )
        else:
            parts.append(
                f"Actual confidence is {abs(delta):.2f} above expected; "
                f"confidence reduced by {adjustment:+.4f}."
            )

        if historical:
            mean_hist_delta = self._mean([h["delta"] for h in historical])
            parts.append(
                f"Historical mean delta {mean_hist_delta:+.4f} "
                f"blended into adjustment."
            )

        return " ".join(parts)

    def to_version_report(self) -> dict[str, Any]:
        """Engine-versioned summary of this calibration layer."""
        return {
            "engine_version": ENGINE_VERSION,
            "history_count": len(self._history),
            "calibration_stats": self.compute_calibration_error(),
        }


# ---------------------------------------------------------------------------
# Convenience factory matching the module-level function style
# used elsewhere in the decision package.
# ---------------------------------------------------------------------------


def calibrate_confidence(
    *,
    expected_confidence: float,
    actual_confidence: float,
    benchmark_case_id: str = "default",
    historical_points: list[CalibrationPoint] | None = None,
    use_history: bool = True,
) -> CalibrationAdjustment:
    """One-shot calibration adjustment using fresh history.

    Pure convenience: constructs a temporary CalibrationLayer, seeds
    it with optional historical points, and delegates to
    :meth:`CalibrationLayer.compute_adjustment`.
    """
    layer = CalibrationLayer()
    if historical_points:
        layer.add_historical_points(historical_points)
    return layer.compute_adjustment(
        expected_confidence=expected_confidence,
        actual_confidence=actual_confidence,
        benchmark_case_id=benchmark_case_id,
        use_history=use_history,
    )
