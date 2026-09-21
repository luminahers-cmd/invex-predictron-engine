"""Phase 6 — deterministic drift detection over monitor snapshots.

Drift compares two :class:`MonitorSnapshot` populations (``before`` /
``after``) across exactly five dimensions:

1. **calibration drift** — delta of the stored Expected Calibration Error
   (produced by the canonical Sprint 8 calibration report at snapshot time).
2. **confidence drift** — delta of the stored mean predicted confidence.
   When two live populations are available the app layer produces this
   value through the canonical
   :func:`~predictron_engine.intelligence.confidence_drift.compute_confidence_drift`
   (its ``decision_drift.delta`` is arithmetically identical to the mean
   delta, so stored-snapshot drift never disagrees with live drift).
3. **prediction-distribution drift** — total-variation distance between the
   ``decision`` label frequency distributions:
   ``0.5 * sum(|p_k - q_k|)`` over the sorted union of labels.
4. **outcome-distribution drift** — the same total-variation distance over
   the ``outcome_verdict`` frequency distributions.
5. **horizon-performance drift** — per-horizon accuracy deltas
   (``after.accuracy - before.accuracy``), each in percentage points; the
   signal magnitude is the mean of the per-horizon magnitudes.

Mathematical definitions
------------------------
* ``magnitude`` is ``100 * |delta|`` for rates, ``100 * total_variation``
  for distribution drift (percentage-point scale, matching the benchmark
  severity bands it reuses).
* ``direction`` reuses :func:`predictron_engine.intelligence.history.direction_between`.
* ``severity`` maps magnitude to ``low | minor | moderate | major`` via
  ``MONITOR_DRIFT_SEVERITY_MINOR / _MODERATE / _MAJOR`` (mirrors
  ``benchmarks.ground_truth_eval.drift``).
* ``affected`` is ``magnitude >= MONITOR_DRIFT_EPSILON``.

No heuristic or hidden threshold exists.
"""

from __future__ import annotations

from datetime import date

from predictron_engine.intelligence.confidence_drift import compute_confidence_drift
from predictron_engine.intelligence.history import direction_between
from predictron_engine.intelligence.models import (
    ConfidenceDrift,
    PerformanceSnapshot,
)
from predictron_engine.monitoring.models import (
    MONITOR_CONFIDENCE_DRIFT_THRESHOLD,
    MONITOR_DRIFT_EPSILON,
    MONITOR_DRIFT_SEVERITY_MAJOR,
    MONITOR_DRIFT_SEVERITY_MINOR,
    MONITOR_DRIFT_SEVERITY_MODERATE,
    MonitorDriftReport,
    MonitorDriftSignal,
    MonitorSnapshot,
)


def _severity(magnitude: float) -> str:
    """Deterministic severity bands over percentage-point magnitudes."""
    if magnitude >= MONITOR_DRIFT_SEVERITY_MAJOR:
        return "major"
    if magnitude >= MONITOR_DRIFT_SEVERITY_MODERATE:
        return "moderate"
    if magnitude >= MONITOR_DRIFT_SEVERITY_MINOR:
        return "minor"
    return "low"


def _metric_signal(
    name: str,
    from_value: float | None,
    to_value: float | None,
    *,
    detail: dict[str, object] | None = None,
) -> MonitorDriftSignal | None:
    """One drift signal from two metric endpoints (``None`` when undefined)."""
    if from_value is None or to_value is None:
        return None
    delta = round(to_value - from_value, 6)
    magnitude = round(abs(delta) * 100.0, 6)
    return MonitorDriftSignal(
        signal=name,
        from_value=from_value,
        to_value=to_value,
        delta=delta,
        magnitude=magnitude,
        direction=direction_between(from_value, to_value),
        severity=_severity(magnitude),
        affected=magnitude >= MONITOR_DRIFT_EPSILON,
        detail=detail or {},
    )


def _distribution_drift_signal(
    name: str,
    before: MonitorSnapshot,
    after: MonitorSnapshot,
    distribution_kind: str,
) -> MonitorDriftSignal | None:
    """Total-variation distance between two stored frequency distributions."""
    before_dist = before.distributions.get(distribution_kind)
    after_dist = after.distributions.get(distribution_kind)
    if before_dist is None or after_dist is None:
        return None
    universe = sorted(set(before_dist.universe) | set(after_dist.universe))
    p = before_dist.frequencies()
    q = after_dist.frequencies()
    tv = 0.5 * sum(abs(p.get(label, 0.0) - q.get(label, 0.0)) for label in universe)
    magnitude = round(tv * 100.0, 6)
    # Single-number from/to: the share of the largest category in each
    # population, so the signal direction reflects how the dominant mass
    # moved while magnitude reflects the total-variation distance.
    from_dominant = _dominant_fraction(before_dist.counts)
    to_dominant = _dominant_fraction(after_dist.counts)
    return MonitorDriftSignal(
        signal=name,
        from_value=from_dominant,
        to_value=to_dominant,
        delta=round(tv, 6),
        magnitude=magnitude,
        direction=direction_between(from_dominant, to_dominant),
        severity=_severity(magnitude),
        affected=magnitude >= MONITOR_DRIFT_EPSILON,
        detail={
            "total_variation": round(tv, 6),
            "categories": {label: (p.get(label, 0.0), q.get(label, 0.0)) for label in universe},
        },
    )


def _dominant_fraction(counts: dict[str, int]) -> float:
    """Share of the largest category (0.0 when the population is empty)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return round(max(counts.values()) / total, 6)


def _horizon_performance_signal(
    before: MonitorSnapshot, after: MonitorSnapshot
) -> MonitorDriftSignal | None:
    """Mean of per-horizon accuracy deltas (present in both populations)."""
    deltas: list[float] = []
    detail_map: dict[str, object] = {}
    horizon_days = sorted(
        set(before.horizon_breakdown) | set(after.horizon_breakdown)
    )
    for days in horizon_days:
        before_bucket = before.horizon_breakdown.get(str(days))
        after_bucket = after.horizon_breakdown.get(str(days))
        if before_bucket is None or after_bucket is None:
            continue
        from_acc = before_bucket.accuracy
        to_acc = after_bucket.accuracy
        if from_acc is None or to_acc is None:
            continue
        delta = to_acc - from_acc
        deltas.append(delta)
        detail_map[str(days)] = {
            "from": from_acc,
            "to": to_acc,
            "delta": round(delta, 6),
        }
    if not deltas:
        return None
    mean_delta = round(sum(deltas) / len(deltas), 6)
    magnitude = round(abs(mean_delta) * 100.0, 6)
    return MonitorDriftSignal(
        signal="horizon_performance",
        from_value=min(deltas),
        to_value=max(deltas),
        delta=mean_delta,
        magnitude=magnitude,
        direction=direction_between(min(deltas), max(deltas)),
        severity=_severity(magnitude),
        affected=magnitude >= MONITOR_DRIFT_EPSILON,
        detail={"horizons": detail_map},
    )


def detect_monitor_drift(
    before: MonitorSnapshot, after: MonitorSnapshot
) -> MonitorDriftReport:
    """Detect drift between two snapshots across all five dimensions.

    Snapshots must share the same ``scope`` and ``engine_version`` for the
    comparison to be meaningful; a mismatch raises ``ValueError`` (a named,
    deterministic guard — never silent).
    """
    if before.scope != after.scope:
        raise ValueError(
            "cannot compare snapshots across scopes: "
            f"{before.scope!r} vs {after.scope!r}"
        )
    if before.engine_version != after.engine_version:
        raise ValueError(
            "cannot compare snapshots across engine versions: "
            f"{before.engine_version!r} vs {after.engine_version!r}"
        )

    signals: list[MonitorDriftSignal] = []
    for key, signal_name in (
        ("calibration.ece", "calibration"),
        ("confidence.mean", "confidence"),
    ):
        signal = _metric_signal(
            signal_name,
            before.metrics.get(key),
            after.metrics.get(key),
        )
        if signal is not None:
            signals.append(signal)

    for signal_name, distribution_kind in (
        ("prediction_distribution", "decision"),
        ("outcome_distribution", "outcome_verdict"),
    ):
        signal = _distribution_drift_signal(
            signal_name, before, after, distribution_kind
        )
        if signal is not None:
            signals.append(signal)

    signal = _horizon_performance_signal(before, after)
    if signal is not None:
        signals.append(signal)

    signals.sort(key=lambda entry: entry.signal)
    return MonitorDriftReport(
        baseline_id=before.snapshot_id,
        comparison_id=after.snapshot_id,
        baseline_period=_as_date(before.anchor_date),
        comparison_period=_as_date(after.anchor_date),
        signals=signals,
    )


def confidence_drift_between(
    population_a: list[PerformanceSnapshot],
    population_b: list[PerformanceSnapshot],
    *,
    overconfident: bool = False,
    drifted_threshold: float = MONITOR_CONFIDENCE_DRIFT_THRESHOLD,
) -> ConfidenceDrift:
    """Confidence drift between two live populations.

    Pure delegation to the canonical Sprint 9
    :func:`compute_confidence_drift` — no drift math is re-implemented.
    Callers project ledger rows onto :class:`PerformanceSnapshot` via
    ``intelligence.history.snapshot_from_confidence``.
    """
    return compute_confidence_drift(
        population_a,
        population_b,
        overconfident=overconfident,
        drifted_threshold=drifted_threshold,
    )


def _as_date(value: date) -> date:
    return value


__all__ = [
    "confidence_drift_between",
    "detect_monitor_drift",
]
