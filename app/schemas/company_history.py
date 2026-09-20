"""Pydantic schemas for the Company Intelligence Hub temporal layer (Phase 3).

Additive read-side schemas: the enriched per-snapshot timeline view, the
deterministic trend summary, and the latest-history view. The Phase 1
``CompanyHistoryResponse`` in :mod:`app.schemas.company` is deliberately
left untouched so existing callers keep their contract.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.company_profile import CompanyBenchmarkSummary


class TrendDirection(str, Enum):
    """Deterministic trend classification over a chronological series.

    ``increasing`` / ``decreasing`` require *every* usable consecutive
    change to move in that direction; ``stable`` requires every usable
    change to sit within rounding tolerance; ``mixed`` means usable changes
    point both ways; ``insufficient`` means fewer than two usable
    observations exist (nothing is extrapolated or predicted).
    """

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    MIXED = "mixed"
    INSUFFICIENT = "insufficient"


class CompanyTrendSummary(BaseModel):
    """One deterministic trend direction per tracked numeric series."""

    confidence: TrendDirection = Field(
        ..., description="Trend of snapshot confidence over time"
    )
    readiness: TrendDirection = Field(
        ..., description="Trend of readiness scores over time"
    )
    composite_score: TrendDirection = Field(
        ..., description="Trend of composite scores over time"
    )
    benchmark_percentile: TrendDirection = Field(
        ..., description="Trend of benchmark percentile rank over time"
    )
    snapshot_count: int = Field(
        ..., ge=0, description="Snapshots contributing to the trends"
    )


class CompanyTimelineEntry(BaseModel):
    """One enriched snapshot on a company's temporal timeline.

    ``benchmark``, ``decision_explanation``, ``key_facts`` and
    ``evidence_count`` derive deterministically from stored data only: the
    benchmark placement from the composite score against the offline dataset
    population, and the explanation / facts / evidence count from the
    persisted analysis report linked to this snapshot. Each derived field is
    ``None``/empty when its source data is unavailable — derived fields never
    fabricate values.
    """

    snapshot_id: str = Field(..., description="Deterministic snapshot identifier")
    company_id: str = Field(..., description="Owning company identifier")
    analysis_id: str = Field(..., description="Source analysis request ID")
    report_id: str = Field(..., description="Source analysis report ID")
    created_at: datetime = Field(..., description="When the snapshot was created")
    decision: str | None = Field(default=None, description="Decision category")
    confidence: float | None = Field(default=None, ge=0, le=1, description="Confidence")
    readiness_score: float | None = Field(
        default=None, ge=0, le=100, description="Investment readiness score"
    )
    composite_score: float | None = Field(
        default=None, ge=0, le=100, description="Composite investment score"
    )
    dimension_scores: dict[str, float] = Field(
        default_factory=dict, description="Per-dimension scores"
    )
    benchmark: CompanyBenchmarkSummary | None = Field(
        default=None, description="Benchmark placement of the composite score"
    )
    decision_explanation: list[str] = Field(
        default_factory=list,
        description="Deterministic explanation strings from the stored report rationale",
    )
    key_facts: list[str] = Field(
        default_factory=list,
        description="Deterministic key facts derived from the snapshot and stored report",
    )
    evidence_count: int | None = Field(
        default=None, ge=0, description="Evidence item count from the stored report"
    )


class CompanyTimelineResponse(BaseModel):
    """Paginated enriched timeline (newest-first) plus the global trend."""

    company_id: str = Field(..., description="Deterministic company identifier")
    total: int = Field(default=0, ge=0, description="Snapshots in the timeline")
    entries: list[CompanyTimelineEntry] = Field(default_factory=list)
    trend: CompanyTrendSummary = Field(..., description="Trend over the full timeline")


class CompanyHistoryLatestResponse(BaseModel):
    """Latest enriched snapshot plus the company's overall trend summary."""

    company_id: str = Field(..., description="Deterministic company identifier")
    total: int = Field(default=0, ge=0, description="Snapshots recorded for the company")
    entry: CompanyTimelineEntry | None = Field(
        default=None, description="Most recent enriched snapshot, when one exists"
    )
    trend: CompanyTrendSummary = Field(..., description="Trend over the full timeline")


__all__ = [
    "CompanyHistoryLatestResponse",
    "CompanyTimelineEntry",
    "CompanyTimelineResponse",
    "CompanyTrendSummary",
    "TrendDirection",
]
