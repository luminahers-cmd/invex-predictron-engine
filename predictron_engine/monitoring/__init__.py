"""Phase 6 — Continuous Intelligence & Drift Detection.

Public surface of the deterministic monitoring layer: health projection,
summary analytics, drift detection, time-series trends, re-analysis
recommendations, snapshot building, and the append-only snapshot history.
All analytic values are defined in the module docstrings (deterministic,
canonical-source reuse — see :mod:`~predictron_engine.monitoring.models`).
"""

from predictron_engine.monitoring.drift import (
    confidence_drift_between,
    detect_monitor_drift,
)
from predictron_engine.monitoring.health import (
    build_health_entries,
    derive_forecast_health,
    health_distribution,
)
from predictron_engine.monitoring.history import MonitorHistory
from predictron_engine.monitoring.models import (
    METRIC_KEYS,
    Distribution,
    ForecastHealth,
    ForecastRecord,
    HealthEntry,
    HorizonPerformance,
    MetricTrend,
    MonitorDriftReport,
    MonitorDriftSignal,
    MonitorPeriodKind,
    MonitorSnapshot,
    MonitorSnapshotSummary,
    ReanalysisReason,
    ReanalysisRecommendation,
    SectorPerformance,
    TimePoint,
    json_dumps_canonical,
)
from predictron_engine.monitoring.reanalysis import build_reanalysis_recommendations
from predictron_engine.monitoring.snapshot import (
    build_monitor_rolling_snapshot,
    build_monitor_snapshot,
)
from predictron_engine.monitoring.summaries import (
    calibration_summary,
    decision_distribution,
    evaluation_verdict_distribution,
    horizon_breakdown,
    mean_confidence,
    outcome_verdict_distribution,
    rolling_mean_confidence,
    rolling_metrics,
    sector_breakdown,
    status_distribution,
)
from predictron_engine.monitoring.trends import compute_trend, metric_series

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
    "MonitorHistory",
    "MonitorPeriodKind",
    "MonitorSnapshot",
    "MonitorSnapshotSummary",
    "ReanalysisReason",
    "ReanalysisRecommendation",
    "SectorPerformance",
    "TimePoint",
    "build_health_entries",
    "build_monitor_rolling_snapshot",
    "build_monitor_snapshot",
    "build_reanalysis_recommendations",
    "calibration_summary",
    "compute_trend",
    "confidence_drift_between",
    "decision_distribution",
    "derive_forecast_health",
    "detect_monitor_drift",
    "evaluation_verdict_distribution",
    "health_distribution",
    "horizon_breakdown",
    "json_dumps_canonical",
    "mean_confidence",
    "metric_series",
    "outcome_verdict_distribution",
    "rolling_metrics",
    "rolling_mean_confidence",
    "sector_breakdown",
    "status_distribution",
]
