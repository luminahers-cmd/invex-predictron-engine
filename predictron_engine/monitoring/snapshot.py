"""Phase 6 — deterministic monitor snapshot builder.

A :class:`MonitorSnapshot` is the single point-in-time population summary of
the monitoring layer.  Every composite field is produced by the deterministic
builders in :mod:`health`, :mod:`summaries`, :mod:`drift`, and
:mod:`reanalysis` — nothing here re-computes analytic values.

Mathematical definitions
------------------------
* ``counts.forecasts`` — ``len(records)``.
* ``counts.forecasts_outcome_linked`` — forecasts with ``outcome_id`` set.
* ``counts.companies`` — distinct ``company_id`` in ``records``.
* ``counts.snapshots`` — distinct pinned ``snapshot_id`` in ``records``.
* ``counts.evaluations`` — ``len(evaluations)``.
* ``metrics.*`` — canonical evaluation metric values
  (:func:`compute_evaluation_metrics`), calibration from the Sprint 8
  machinery (:func:`summaries.calibration_summary`), and
  ``confidence.mean`` via :func:`summaries.mean_confidence`.
* ``distributions`` — the verdict / outcome / decision frequency tables.
* ``horizon_breakdown`` / ``sector_breakdown`` / ``health`` — the same
  decompositions the app serves live.

``snapshot_id`` is deterministic: ``sha256(scope | period_kind | anchor)
``; when an explicit id is supplied it must still be unique per
``(scope, period_kind, anchor_date)``.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime

from predictron_engine.dataset.evaluation import PredictionEvaluation
from predictron_engine.dataset.metrics import compute_evaluation_metrics
from predictron_engine.monitoring import health, summaries
from predictron_engine.monitoring.models import (
    MONITOR_DEFAULT_ROLLING_WINDOW,
    ForecastRecord,
    MonitorPeriodKind,
    MonitorSnapshot,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _snapshot_id(scope: str, period_kind: MonitorPeriodKind, anchor_date: date) -> str:
    material = "|".join((scope, period_kind.value, anchor_date.isoformat()))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _build_payload(
    *,
    records: list[ForecastRecord],
    evaluations: list[PredictionEvaluation],
    window: int | None,
) -> tuple[dict[str, int], dict[str, float | None], dict[str, str]]:
    """Flat counts/metrics/meta payloads (windowed variant when requested)."""
    metrics_obj = compute_evaluation_metrics(evaluations)
    calibration = summaries.calibration_summary(evaluations)

    counts: dict[str, int] = {
        "forecasts": len(records),
        "forecasts_outcome_linked": sum(
            1 for record in records if record.outcome_id is not None
        ),
        "companies": len({record.company_id for record in records}),
        "snapshots": len({record.snapshot_id for record in records}),
        "evaluations": len(evaluations),
        "scoreable": int(metrics_obj.scoreable),
    }

    metrics: dict[str, float | None] = {
        "metrics.accuracy": metrics_obj.accuracy,
        "metrics.precision": metrics_obj.precision,
        "metrics.recall": metrics_obj.recall,
        "metrics.f1": metrics_obj.f1,
        "metrics.balanced_accuracy": metrics_obj.balanced_accuracy,
        "metrics.coverage": metrics_obj.coverage,
        "calibration.ece": float(calibration["expected_calibration_error"]),
        "calibration.max_ece": float(calibration["maximum_calibration_error"]),
        "calibration.overconfidence": 1.0 if calibration["overconfidence_detected"] else 0.0,
        "calibration.total_samples": float(calibration["total_samples"]),
    }
    confidence_mean = summaries.mean_confidence(records)
    if confidence_mean is not None:
        metrics["confidence.mean"] = confidence_mean

    if window is not None:
        metrics["metrics.accuracy"] = summaries.rolling_metrics(
            evaluations, window
        ).accuracy

    return counts, metrics, {"window": str(window) if window is not None else "all"}


def build_monitor_snapshot(
    records: list[ForecastRecord],
    evaluations: list[PredictionEvaluation],
    *,
    anchor_date: date,
    period_kind: MonitorPeriodKind,
    scope: str = "repository",
    engine_version: str = "6.0.0",
    recorded_at: datetime | None = None,
    snapshot_id: str | None = None,
    as_of: datetime | None = None,
) -> MonitorSnapshot:
    """Full-population monitor snapshot for ``anchor_date``."""
    counts, metrics, meta = _build_payload(
        records=records, evaluations=evaluations, window=None
    )
    snapshot = _assemble(
        records=records,
        evaluations=evaluations,
        counts=counts,
        metrics=metrics,
        meta=meta,
        anchor_date=anchor_date,
        period_kind=period_kind,
        scope=scope,
        engine_version=engine_version,
        recorded_at=recorded_at,
        snapshot_id=snapshot_id,
        as_of=as_of,
    )
    return snapshot


def build_monitor_rolling_snapshot(
    records: list[ForecastRecord],
    evaluations: list[PredictionEvaluation],
    *,
    anchor_date: date,
    period_kind: MonitorPeriodKind,
    scope: str = "repository",
    engine_version: str = "6.0.0",
    window: int = MONITOR_DEFAULT_ROLLING_WINDOW,
    recorded_at: datetime | None = None,
    snapshot_id: str | None = None,
    as_of: datetime | None = None,
) -> MonitorSnapshot:
    """Windowed monitor snapshot (rolled aggregates over the window).

    Only the aggregate metrics and verdict distributions are windowed; the
    structural decomposition (horizon / sector / health) always spans the
    full population.
    """
    counts, metrics, meta = _build_payload(
        records=records, evaluations=evaluations, window=window
    )
    snapshot = _assemble(
        records=records,
        evaluations=evaluations,
        counts=counts,
        metrics=metrics,
        meta=meta,
        anchor_date=anchor_date,
        period_kind=period_kind,
        scope=scope,
        engine_version=engine_version,
        recorded_at=recorded_at,
        snapshot_id=snapshot_id,
        as_of=as_of,
    )
    return snapshot


def _assemble(
    *,
    records: list[ForecastRecord],
    evaluations: list[PredictionEvaluation],
    counts: dict[str, int],
    metrics: dict[str, float | None],
    meta: dict[str, str],
    anchor_date: date,
    period_kind: MonitorPeriodKind,
    scope: str,
    engine_version: str,
    recorded_at: datetime | None,
    snapshot_id: str | None,
    as_of: datetime | None = None,
) -> MonitorSnapshot:
    resolved_id = snapshot_id or _snapshot_id(scope, period_kind, anchor_date)
    entry_metrics = metrics.copy()
    analysis_at = as_of or _now()
    snapshot = MonitorSnapshot(
        scope=scope,
        snapshot_id=resolved_id,
        period_kind=period_kind,
        anchor_date=anchor_date,
        engine_version=engine_version,
        recorded_at=recorded_at or analysis_at,
        content_hash="",
        counts=counts,
        metrics=entry_metrics,
        meta=meta,
        distributions={
            "evaluation_verdict": summaries.evaluation_verdict_distribution(
                evaluations
            ),
            "outcome_verdict": summaries.outcome_verdict_distribution(evaluations),
            "decision": summaries.decision_distribution(records),
        },
        horizon_breakdown={
            str(days): performance
            for days, performance in summaries.horizon_breakdown(
                evaluations, records
            ).items()
        },
        sector_breakdown=summaries.sector_breakdown(evaluations),
        health=health.health_distribution(
            health.build_health_entries(records, as_of=analysis_at)
        ),
    )
    snapshot.content_hash = snapshot.content_fingerprint()
    return snapshot


__all__ = ["build_monitor_rolling_snapshot", "build_monitor_snapshot"]
