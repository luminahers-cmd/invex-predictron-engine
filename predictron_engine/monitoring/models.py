"""Phase 6 — Continuous Intelligence: monitoring value objects.

Deterministic, pure pydantic models that describe the *live prediction
ledger* and its append-only history.  No computation happens here — every
value is produced by the deterministic builders in :mod:`health`,
:mod:`summaries`, :mod:`drift`, :mod:`trends`, :mod:`reanalysis`, and
:mod:`snapshot`.

Mathematical definitions (documented per requirement — no hidden heuristics)
----------------------------------------------------------------------------
Every metric stored in a :class:`MonitorSnapshot` has an explicit formula.
Where the formula already exists elsewhere in the repository the canonical
source is named and reused, never re-implemented:

* **Accuracy** ``(TP+TN)/(TP+TN+FP+FN)`` over binary-scoreable evaluations
  (canonical: ``predictron_engine.dataset.metrics.compute_evaluation_metrics``).
* **Calibration ECE / MCE** — binned Expected Calibration Error
  (canonical: ``predictron_engine.decision.calibration_sprint8``).
* **Mean confidence** ``mean(forecast.confidence)`` rounded to 4 dp,
  ``None`` when no forecast exists.
* **Confidence drift** — the signed delta of the population mean predicted
  confidence, an arithmetic identity of
  ``predictron_engine.intelligence.confidence_drift.compute_confidence_drift``
  (``decision_drift.delta == mean_b - mean_a``).
* **Distribution drift** — total-variation distance between two
  category-frequency servers over the sorted union of labels:
  ``0.5 * sum(|p_k - q_k|)``, in ``[0, 1]``.
* **Trend** — the slope of a metric's first-vs-last endpoint over an
  ordered series of snapshots (canonical:
  ``predictron_engine.intelligence.history.direction_between``).

Thresholds (all at module level — nothing hidden)
-------------------------------------------------
Every threshold is a named constant.  Constants that already exist
elsewhere (benchmark drift severity bands, Sprint 8 overconfidence and
quality bands, confidence drift threshold) are reused by documented mirror
(the engine must not import ``benchmarks``, so semantically-identical named
constants live here; see :mod:`drift`).
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field

from predictron_engine.intelligence.models import TrendDirection

# ---------------------------------------------------------------------------
# Thresholds (named configuration constants algorithm)
# ---------------------------------------------------------------------------

#: Age (in days) after which an open forecast is classified ``stale``.
MONITOR_DEFAULT_STALE_AFTER_DAYS: int = 365

#: Age (in days) after which a forecast's snapshot is recommended for
#: re-analysis (``snapshot_age_exceeded``).
MONITOR_REANALYSIS_SNAPSHOT_MAX_AGE_DAYS: int = 365

#: Absolute metric delta (in percentage points) below which a drift signal
#: is not flagged.  Mirrors ``SCORE_DRIFT_EPSILON`` / ``CONFIDENCE_DRIFT_EPSILON``
#: in ``benchmarks.ground_truth_eval.drift`` (0.01 on a 0..1 scale, i.e. one
#: percentage point).
MONITOR_DRIFT_EPSILON: float = 1.0

#: Severity bands on drift magnitudes expressed in percentage points.
#: Mirrors ``SEVERITY_MINOR`` / ``SEVERITY_MODERATE`` / ``SEVERITY_MAJOR`` of
#: ``benchmarks.ground_truth_eval.drift`` so drift severity stays uniform
#: across the benchmark platform and the monitoring layer.
MONITOR_DRIFT_SEVERITY_MINOR: float = 1.0
MONITOR_DRIFT_SEVERITY_MODERATE: float = 3.0
MONITOR_DRIFT_SEVERITY_MAJOR: float = 10.0

#: Maximum number of points kept in a rolling-window aggregate.
MONITOR_DEFAULT_ROLLING_WINDOW: int = 30

#: Confidence drift classification threshold (absolute per-dimension
#: assessed-confidence change in [0,1]); the same default used by
#: ``compute_confidence_drift``.
MONITOR_CONFIDENCE_DRIFT_THRESHOLD: float = 0.05

# ---------------------------------------------------------------------------
# Metric map keys (deterministic, stable for the time-series contract)
# ---------------------------------------------------------------------------

METRIC_KEYS = {
    "total": "counts.total",
    "forecasts": "counts.forecasts",
    "forecasts_outcome_linked": "counts.forecasts_outcome_linked",
    "companies": "counts.companies",
    "snapshots": "counts.snapshots",
    "outcomes": "counts.outcomes",
    "evaluations": "counts.evaluations",
    "accuracy": "metrics.accuracy",
    "precision": "metrics.precision",
    "recall": "metrics.recall",
    "f1": "metrics.f1",
    "balanced_accuracy": "metrics.balanced_accuracy",
    "coverage": "metrics.coverage",
    "ece": "calibration.ece",
    "max_ece": "calibration.max_ece",
    "calibration_overconfidence": "calibration.overconfidence",
    "calibration_samples": "calibration.total_samples",
    "confidence_mean": "confidence.mean",
    "confidence_delta": "confidence.delta",
    "scoreable": "metrics.scoreable",
}


# ---------------------------------------------------------------------------
# Enums and value objects
# ---------------------------------------------------------------------------


class MonitorPeriodKind(str, Enum):
    """Period cadence of an append-only monitor snapshot."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ForecastHealth(str, Enum):
    """Derived prediction-health state of one forecast.

    Vocabulary (lower-cased for on-the-wire consistency with every other
    status enum in the repository): ``active`` (within horizon), ``due``
    (horizon reached, per :func:`forecast_lifecycle.derive_forecast_status`),
    ``overdue`` (``as_of`` past ``due_at`` with no outcome attached),
    ``stale`` (analysis older than ``MONITOR_DEFAULT_STALE_AFTER_DAYS`` and
    still unresolved), ``resolved`` (outcome attached).
    """

    ACTIVE = "active"
    DUE = "due"
    OVERDUE = "overdue"
    STALE = "stale"
    RESOLVED = "resolved"


class ReanalysisReason(str, Enum):
    """Deterministic reason a frozen prediction merits re-analysis.

    * ``new_outcome_recorded`` — an outcome was recorded after the frozen
      snapshot and is not yet evaluated against it.
    * ``evidence_changed`` — a newer snapshot exists for the company than
      the snapshot the forecast is pinned to.
    * ``prediction_expired`` — the forecast is overdue with no outcome
      attached.
    * ``snapshot_age_exceeded`` — the snapshot is older than
      ``MONITOR_REANALYSIS_SNAPSHOT_MAX_AGE_DAYS``.
    """

    NEW_OUTCOME_RECORDED = "new_outcome_recorded"
    EVIDENCE_CHANGED = "evidence_changed"
    PREDICTION_EXPIRED = "prediction_expired"
    SNAPSHOT_AGE_EXCEEDED = "snapshot_age_exceeded"


class ForecastRecord(BaseModel):
    """Pure projection of one persisted ledger forecast.

    Precisely the stored fields the monitoring analytics need; the app layer
    maps its ORM rows on to this shape (no engine dependency on ``app``).
    """

    id: str = Field(..., description="Deterministic forecast identifier")
    company_id: str = Field(..., description="Owning company identifier")
    snapshot_id: str = Field(..., description="Pinned frozen prediction snapshot")
    decision: str = Field(..., description="Frozen decision label")
    confidence: float = Field(..., ge=0.0, le=1.0)
    composite_score: float = Field(..., ge=0.0, le=100.0)
    status: str = Field(..., description="Persisted lifecycle status value")
    analysis_timestamp: datetime = Field(...)
    due_at: datetime = Field(...)
    horizon_days: int = Field(..., gt=0)
    outcome_id: str | None = Field(default=None)
    outcome_verdict: str | None = Field(default=None)
    evaluation_verdict: str | None = Field(default=None)
    evaluation_created_at: datetime | None = Field(default=None)
    latest_outcome_at: datetime | None = Field(
        default=None,
        description="Latest observed outcome for the company (re-analysis)",
    )
    latest_snapshot_at: datetime | None = Field(
        default=None,
        description="Latest snapshot for the company (re-analysis)",
    )


class HealthEntry(BaseModel):
    """Derived prediction-health row for one forecast at ``as_of``.

    ``age_days`` is ``(as_of - analysis_timestamp).days``; ``days_until_due``
    is ``(due_at - as_of).days`` (negative once overdue) reusing the
    ``PredictionForecast.days_until_due`` semantics; ``days_overdue`` is
    ``-(days_until_due)`` when negative, else ``0``.
    """

    forecast_id: str = Field(..., description="Deterministic forecast identifier")
    company_id: str = Field(...)
    snapshot_id: str = Field(...)
    decision: str = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    status: str = Field(..., description="Persisted lifecycle status value")
    health: ForecastHealth = Field(..., description="Derived health state")
    analysis_timestamp: datetime = Field(...)
    due_at: datetime = Field(...)
    as_of: datetime = Field(...)
    age_days: int = Field(..., ge=0)
    days_until_due: float = Field(...)
    days_overdue: float = Field(..., ge=0.0)
    outcome_id: str | None = Field(default=None)
    outcome_verdict: str | None = Field(default=None)
    evaluation_status: str = Field(
        default="pending",
        description="'evaluated' when a deterministic evaluation exists, else 'pending'",
    )
    evaluation_verdict: str | None = Field(default=None)


class Distribution(BaseModel):
    """A category-frequency distribution over an explicit label universe.

    ``counts`` covers every label in the universe (present labels carry
    their count; absent labels carry ``0``) so distributions stay comparable
    across snapshots.  ``total`` is the number of observed members.
    """

    label: str = Field(..., description="Distribution kind, e.g. 'decision'")
    universe: list[str] = Field(..., description="Sorted label universe")
    counts: dict[str, int] = Field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def frequencies(self) -> dict[str, float]:
        """Relative frequencies ``count/total`` (0.0 when no members)."""
        total = self.total
        if total == 0:
            return {label: 0.0 for label in self.universe}
        return {
            label: round(self.counts.get(label, 0) / total, 6) for label in self.universe
        }

    def as_relative(self) -> dict[str, int]:
        return dict(self.counts)


class HorizonPerformance(BaseModel):
    """Per-horizon performance over evaluations whose snapshot is forecast.

    Attribution (documented): an evaluation is attributed to every horizon
    that forecasts its snapshot — a multi-horizon snapshot is evaluated
    against each of its horizons, so a single evaluation may appear in more
    than one bucket.  All rates reuse ``compute_evaluation_metrics``.
    """

    horizon_days: int = Field(..., gt=0)
    forecasts: int = Field(default=0, ge=0)
    evaluated: int = Field(default=0, ge=0)
    scoreable: int = Field(default=0, ge=0)
    accuracy: float | None = Field(default=None)


class SectorPerformance(BaseModel):
    """Per-dimension (sector) performance over frozen predictions.

    A sector is one key of the frozen ``dimension_scores`` payload (the
    engine's Market / Founder / Product / ... dimensions).  The ``overall``
    bucket aggregates the whole population.  Rates reuse
    ``compute_evaluation_metrics``.
    """

    sector: str = Field(..., description="Dimension score key (or 'overall')")
    evaluations: int = Field(default=0, ge=0)
    scoreable: int = Field(default=0, ge=0)
    accuracy: float | None = Field(default=None)


class MonitorDriftSignal(BaseModel):
    """One deterministic drift dimension between two snapshot populations.

    ``magnitude`` is the absolute delta expressed in **percentage points**
    (``100 * |delta|`` for rates, ``100 * total_variation`` for
    distributions) so severity bands mirror the benchmark platform.
    """

    signal: str = Field(..., description="Drift kind, e.g. 'calibration'")
    from_value: float
    to_value: float
    delta: float = Field(..., description="Signed numeric delta (to - from)")
    magnitude: float = Field(..., ge=0.0, description="|delta| in percentage points")
    direction: TrendDirection = Field(..., description="Endpoint slope direction")
    severity: str = Field(..., description="low | minor | moderate | major")
    affected: bool = Field(..., description="True when magnitude >= drift epsilon")
    detail: dict[str, object] = Field(default_factory=dict)


class MonitorDriftReport(BaseModel):
    """Full drift comparison between two monitor snapshots."""

    baseline_id: str = Field(...)
    comparison_id: str = Field(...)
    baseline_period: date = Field(...)
    comparison_period: date = Field(...)
    signals: list[MonitorDriftSignal] = Field(default_factory=list)

    def signal(self, name: str) -> MonitorDriftSignal | None:
        for entry in self.signals:
            if entry.signal == name:
                return entry
        return None


class TimePoint(BaseModel):
    """One point of a dashboard-consumable time-series.

    ``anchor`` is the snapshot period date and ``value`` the metric value
    (``None`` metrics are omitted from series by :mod:`trends`).
    """

    anchor: date = Field(..., description="Snapshot period date")
    value: float = Field(...)


class MetricTrend(BaseModel):
    """A time-indexed series plus its deterministic endpoint slope.

    ``series`` is ordered by ``anchor`` ascending.  This is the first-class
    time-series contract exposed to dashboards — no redesign needed to plot
    ``accuracy`` day 1, day 2, day 3, ...
    """

    metric: str = Field(..., description="Metric map key")
    direction: TrendDirection = Field(..., description="First-vs-last slope")
    from_value: float
    to_value: float
    series: list[TimePoint] = Field(default_factory=list)


class ReanalysisRecommendation(BaseModel):
    """Deterministic re-analysis recommendation for one frozen forecast."""

    company_id: str = Field(...)
    forecast_id: str = Field(...)
    snapshot_id: str = Field(...)
    reasons: list[str] = Field(
        default_factory=list, description="Sorted, deduplicated reason values"
    )
    latest_outcome_at: datetime | None = Field(default=None)
    latest_snapshot_at: datetime | None = Field(default=None)


# ---------------------------------------------------------------------------
# MonitorSnapshot
# ---------------------------------------------------------------------------


class MonitorSnapshot(BaseModel):
    """One point-in-time, append-only snapshot of the monitoring layer.

    ``counts`` is a flat ``metric -> int`` map; ``metrics`` a flat
    ``metric -> float`` map; ``distributions`` the label-universe frequency
    maps; ``horizon_breakdown`` / ``sector_breakdown`` the per-bucket
    performance; ``health`` the derived health distribution.  All keys are
    stable so dashboards can consume the history without redesign.

    ``recorded_at`` is informational only and is excluded from the content
    hash (mirroring ``BenchmarkHistory`` / ``PredictionStore``).
    """

    scope: str = Field(default="repository", description="Snapshot scope")
    snapshot_id: str = Field(..., description="Deterministic snapshot identifier")
    period_kind: MonitorPeriodKind = Field(...)
    anchor_date: date = Field(...)
    engine_version: str = Field(...)
    recorded_at: datetime = Field(...)
    content_hash: str = Field(..., description="Integrity hash of the analytic payload")
    counts: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    meta: dict[str, str] = Field(default_factory=dict)
    distributions: dict[str, Distribution] = Field(default_factory=dict)
    horizon_breakdown: dict[str, HorizonPerformance] = Field(default_factory=dict)
    sector_breakdown: dict[str, SectorPerformance] = Field(default_factory=dict)
    health: dict[str, int] = Field(default_factory=dict)

    def analytic_payload(self) -> dict[str, object]:
        """The canonical, hash-covered analytic surface (no timestamps)."""
        return {
            "scope": self.scope,
            "snapshot_id": self.snapshot_id,
            "period_kind": self.period_kind.value,
            "anchor_date": self.anchor_date.isoformat(),
            "engine_version": self.engine_version,
            "counts": self.counts,
            "metrics": {k: v for k, v in self.metrics.items()},
            "meta": self.meta,
            "distributions": {
                kind: {"universe": dist.universe, "counts": dist.counts}
                for kind, dist in self.distributions.items()
            },
            "horizon_breakdown": {
                str(day): hb.model_dump(mode="json")
                for day, hb in self.horizon_breakdown.items()
            },
            "sector_breakdown": {
                sector: sb.model_dump(mode="json")
                for sector, sb in self.sector_breakdown.items()
            },
            "health": self.health,
        }

    def content_fingerprint(self) -> str:
        """Deterministic SHA-256 of the canonical analytic payload."""
        material = json_dumps_canonical(self.analytic_payload())
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def verify(self) -> bool:
        """``True`` when the stored hash matches the recomputed payload."""
        return self.content_hash == self.content_fingerprint()


class MonitorSnapshotSummary(BaseModel):
    """Lightweight header of a stored snapshot (index/listing)."""

    snapshot_id: str = Field(...)
    scope: str = Field(default="repository")
    period_kind: MonitorPeriodKind = Field(...)
    anchor_date: date = Field(...)
    engine_version: str = Field(...)
    recorded_at: datetime = Field(...)
    content_hash: str = Field(...)

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "scope": self.scope,
            "period_kind": self.period_kind.value,
            "anchor_date": self.anchor_date.isoformat(),
            "engine_version": self.engine_version,
            "recorded_at": self.recorded_at.isoformat(),
            "content_hash": self.content_hash,
        }

    def _fingerprint(self) -> str:
        snapshot = MonitorSnapshot(
            scope=self.scope,
            snapshot_id=self.snapshot_id,
            period_kind=self.period_kind,
            anchor_date=self.anchor_date,
            engine_version=self.engine_version,
            recorded_at=self.recorded_at,
            content_hash="",
        )
        return snapshot.content_fingerprint()


def json_dumps_canonical(payload: object) -> str:
    """Canonical JSON serialization (sorted keys, compact separators)."""
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


__all__ = [
    "Distribution",
    "ForecastHealth",
    "ForecastRecord",
    "HealthEntry",
    "HorizonPerformance",
    "METRIC_KEYS",
    "MetricTrend",
    "MonitorDriftReport",
    "MonitorDriftSignal",
    "MonitorPeriodKind",
    "MonitorSnapshot",
    "MonitorSnapshotSummary",
    "MONITOR_CONFIDENCE_DRIFT_THRESHOLD",
    "MONITOR_DEFAULT_ROLLING_WINDOW",
    "MONITOR_DEFAULT_STALE_AFTER_DAYS",
    "MONITOR_DRIFT_EPSILON",
    "MONITOR_DRIFT_SEVERITY_MAJOR",
    "MONITOR_DRIFT_SEVERITY_MINOR",
    "MONITOR_DRIFT_SEVERITY_MODERATE",
    "MONITOR_REANALYSIS_SNAPSHOT_MAX_AGE_DAYS",
    "ReanalysisReason",
    "ReanalysisRecommendation",
    "SectorPerformance",
    "TimePoint",
    "json_dumps_canonical",
]
