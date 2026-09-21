"""Monitoring API response schemas (CIH Phase 6).

Responses reuse the engine monitoring value objects verbatim so the
on-the-wire shapes are exactly what the deterministic build layer produces
(no lossy re-mapping).  Every value is derived by the engine builders in
``predictron_engine.monitoring`` — this module only wraps them in response
envelopes.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from predictron_engine.monitoring.models import (
    Distribution,
    HealthEntry,
    HorizonPerformance,
    MetricTrend,
    MonitorDriftSignal,
    MonitorPeriodKind,
    ReanalysisRecommendation,
    SectorPerformance,
)


class MonitorSummaryResponse(BaseModel):
    """Live, repo-wide monitoring summary over the current ledger."""

    scope: str = Field(...)
    period_kind: MonitorPeriodKind = Field(...)
    anchor_date: date = Field(...)
    generated_at: datetime = Field(...)
    counts: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    distributions: dict[str, Distribution] = Field(default_factory=dict)
    horizon_breakdown: dict[str, HorizonPerformance] = Field(default_factory=dict)
    sector_breakdown: dict[str, SectorPerformance] = Field(default_factory=dict)
    health: dict[str, int] = Field(default_factory=dict)


class MonitorHealthListResponse(BaseModel):
    """Derived forecast-health rows plus their distribution."""

    as_of: datetime = Field(...)
    generated_at: datetime = Field(...)
    scope: str = Field(...)
    entries: list[HealthEntry] = Field(default_factory=list)
    distribution: dict[str, int] = Field(default_factory=dict)


class MonitorDriftResponse(BaseModel):
    """Deterministic drift comparison between two snapshot populations."""

    baseline_id: str = Field(...)
    comparison_id: str = Field(...)
    baseline_period: date = Field(...)
    comparison_period: date = Field(...)
    signals: list[MonitorDriftSignal] = Field(default_factory=list)


class MonitorTrendsResponse(BaseModel):
    """Dashboard-consumable time-series for the requested metrics."""

    period_kind: MonitorPeriodKind = Field(...)
    as_of: date = Field(...)
    trends: list[MetricTrend] = Field(default_factory=list)


class MonitorReanalysisResponse(BaseModel):
    """Deterministic re-analysis recommendations for the caller's scope."""

    as_of: datetime = Field(...)
    generated_at: datetime = Field(...)
    scope: str = Field(...)
    recommendations: list[ReanalysisRecommendation] = Field(default_factory=list)


__all__ = [
    "MonitorDriftResponse",
    "MonitorHealthListResponse",
    "MonitorReanalysisResponse",
    "MonitorSummaryResponse",
    "MonitorTrendsResponse",
]
