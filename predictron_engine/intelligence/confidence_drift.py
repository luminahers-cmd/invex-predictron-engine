"""Sprint 9 — Confidence drift analysis.

Detects how calibrated and assessed confidence changes between two
collections of performance snapshots (e.g. two engine versions or two
time windows).  Purely deterministic; the collections are supplied by the
caller.

Overconfidence classification is *delegated* to the Sprint 8 calibration
monitoring module (``calibration_sprint8.build_calibration_report``) by
the caller and passed in; this module never re-derives calibration logic.
"""

from __future__ import annotations

from predictron_engine.intelligence.models import (
    CalibrationDrift,
    ConfidenceDrift,
    DriftDirection,
    PerformanceSnapshot,
)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _direction(a: float, b: float) -> DriftDirection:
    if a == b:
        return DriftDirection.UNCHANGED
    return DriftDirection.IMPROVED if b > a else DriftDirection.REGRESSED


def _calibration_drift(
    from_value: float,
    to_value: float,
    *,
    overconfident: bool = False,
) -> CalibrationDrift:
    return CalibrationDrift(
        from_confidence=from_value,
        to_confidence=to_value,
        delta=round(to_value - from_value, 4),
        direction=_direction(from_value, to_value),
        overconfident=overconfident,
    )


def compute_confidence_drift(
    snapshots_a: list[PerformanceSnapshot],
    snapshots_b: list[PerformanceSnapshot],
    *,
    overconfident: bool = False,
    drifted_threshold: float = 0.05,
) -> ConfidenceDrift:
    """Compute confidence drift between two collections of snapshots.

    Parameters
    ----------
    snapshots_a, snapshots_b:
        Baseline and comparison snapshot collections.
    overconfident:
        Overconfidence flag for the comparison run, delegated from the
        Sprint 8 calibration monitor (caller supplies it).
    drifted_threshold:
        Absolute per-dimension assessed-confidence change (in [0,1]) that
        classifies a dimension as "drifted".  Reserved for callers that
        need the drifted-dimension set.

    Returns
    -------
    A :class:`ConfidenceDrift` with decision drift and per-dimension drift.
    """
    a_mean = _mean([s.decision_confidence for s in snapshots_a])
    b_mean = _mean([s.decision_confidence for s in snapshots_b])

    decision_drift = _calibration_drift(
        a_mean, b_mean, overconfident=overconfident
    )

    dims = sorted(
        {
            dim
            for s in (snapshots_a + snapshots_b)
            for dim in s.assessed_confidences
        }
    )
    assessment_drifts: dict[str, CalibrationDrift] = {}
    for dim in dims:
        a_vals = [
            s.assessed_confidences.get(dim, 0.0) for s in snapshots_a
        ]
        b_vals = [
            s.assessed_confidences.get(dim, 0.0) for s in snapshots_b
        ]
        if not a_vals and not b_vals:
            continue
        assessment_drifts[dim] = _calibration_drift(
            _mean(a_vals), _mean(b_vals)
        )

    return ConfidenceDrift(
        decision_drift=decision_drift,
        assessment_drifts=assessment_drifts,
        sample_count_a=len(snapshots_a),
        sample_count_b=len(snapshots_b),
    )
