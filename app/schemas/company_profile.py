"""Pydantic schemas for the Company Intelligence Hub profile read layer (Phase 2).

The company profile is the unified, read-only summary of a company: live
post-analysis snapshots from the Phase 1 registry merged with the grounded
historical dataset (dataset records, knowledge graph, signals, features).

The profile deliberately exposes *derived summaries* only — never internal
storage details, provenance dicts, or raw engine internals. Coverage gating
guarantees that placeholder offline data is never surfaced as live
intelligence.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.company import CompanySnapshotResponse


class CoverageStatus(str, Enum):
    """Deterministic coverage classification of a company profile.

    ``live`` means the profile is backed by grounded post-analysis snapshots
    from the Phase 1 registry. ``offline`` means only grounded historical
    dataset records exist. ``offline_placeholder`` marks offline records that
    failed every deterministic quality gate — they are reported as placeholders
    and never presented as live intelligence.
    """

    NONE = "none"
    LIVE = "live"
    OFFLINE = "offline"
    OFFLINE_PLACEHOLDER = "offline_placeholder"


class ProfileCompleteness(BaseModel):
    """Deterministic field-completeness of a company identity."""

    total_fields: int = Field(..., description="Number of tracked identity fields")
    populated_fields: int = Field(..., description="Number of tracked fields carrying a value")
    fraction: float = Field(..., ge=0.0, le=1.0, description="populated_fields / total_fields")
    missing_fields: list[str] = Field(
        default_factory=list, description="Tracked fields with no value (sorted)"
    )


class CompanyProfileOfflineSummary(BaseModel):
    """Summary of the matching historical dataset records for a company.

    ``placeholder`` is True when every matching record fails the deterministic
    quality gates; ``placeholder_reasons`` lists every failing gate observed.
    """

    record_count: int = Field(..., ge=0, description="Number of matching dataset records")
    sources: list[str] = Field(default_factory=list, description="Distinct record sources (sorted)")
    placeholder: bool = Field(..., description="True when all records are placeholders")
    placeholder_count: int = Field(
        default=0, ge=0, description="Records failing every quality gate"
    )
    placeholder_reasons: list[str] = Field(
        default_factory=list, description="Failing quality-gate reasons (sorted, de-duplicated)"
    )
    first_analysis_date: datetime | None = Field(
        default=None, description="Timestamp of the earliest matching record"
    )
    last_analysis_date: datetime | None = Field(
        default=None, description="Timestamp of the latest matching record"
    )


class CompanyGraphNeighbor(BaseModel):
    """One adjacent knowledge-graph node of the company."""

    node_id: str
    node_type: str = ""
    label: str = ""
    edge_types: list[str] = Field(
        default_factory=list, description="Edge kinds linking the company"
    )


class CompanyGraphSummary(BaseModel):
    """Deterministic knowledge-graph summary of a company."""

    node_id: str | None = Field(default=None, description="Company node id in the knowledge graph")
    degree: int = Field(default=0, ge=0, description="Distinct incident edges")
    neighbor_count: int = Field(default=0, ge=0, description="Number of adjacent nodes")
    neighbors: list[CompanyGraphNeighbor] = Field(
        default_factory=list, description="Adjacent nodes, sorted deterministically"
    )


class CompanyFeatureItem(BaseModel):
    """One computed feature exposed in the profile."""

    feature_id: str
    feature_name: str = ""
    category: str = ""
    value: Any = None
    value_type: str = ""
    status: str = ""


class CompanyFeatureSummary(BaseModel):
    """Deterministic feature summary of a company."""

    feature_count: int = Field(default=0, ge=0, description="Registered features")
    computed_count: int = Field(default=0, ge=0, description="Features computed successfully")
    failed_count: int = Field(default=0, ge=0, description="Features that failed to compute")
    by_category: dict[str, int] = Field(
        default_factory=dict, description="Computed count per category"
    )
    features: list[CompanyFeatureItem] = Field(
        default_factory=list, description="Per-feature items, sorted by feature_id"
    )


class CompanySignalSummary(BaseModel):
    """Deterministic signal-timeline summary of a company."""

    signal_count: int = Field(default=0, ge=0, description="Number of signals on the timeline")
    type_counts: dict[str, int] = Field(default_factory=dict, description="Signals per type")
    span_days: float = Field(
        default=0.0, ge=0.0, description="Calendar span between first/last signal"
    )
    first_signal_at: datetime | None = Field(default=None, description="First signal timestamp")
    last_signal_at: datetime | None = Field(default=None, description="Last signal timestamp")
    momentum_score: float = Field(
        default=0.0, description="Timeline momentum at the reference time"
    )
    is_active: bool = Field(default=False, description="Recent-activity flag")
    highlights: list[str] = Field(
        default_factory=list, description="Short derived highlight strings"
    )


class CompanyHistorySummary(BaseModel):
    """Aggregated summary of the live (registry) snapshot history."""

    snapshot_count: int = Field(default=0, ge=0, description="Total analysis snapshots stored")
    first_seen: datetime | None = Field(default=None, description="Registry creation time")
    last_seen: datetime | None = Field(default=None, description="Registry last-update time")
    latest_decision: str | None = Field(default=None, description="Decision of the latest analysis")
    latest_confidence: float | None = Field(
        default=None, ge=0, le=1, description="Confidence of the latest analysis"
    )
    latest_composite_score: float | None = Field(
        default=None, ge=0, le=100, description="Composite score of the latest analysis"
    )


class CompanyBenchmarkSummary(BaseModel):
    """Deterministic benchmark placement of the company's composite score.

    Percentiles are computed over the composite scores of every stored
    dataset record — the same population and formula used by the benchmark
    comparison endpoints.
    """

    composite_score: float = Field(..., ge=0, le=100, description="Company composite score")
    benchmark_mean: float = Field(..., description="Mean of the historical score population")
    benchmark_std_dev: float = Field(..., description="Std-dev of the historical score population")
    percentile_rank: float = Field(..., description="Percentile rank of the score (0-100)")
    z_score: float = Field(..., description="Standardized deviation from the population mean")
    sample_size: int = Field(..., ge=0, description="Number of historical scores in the population")


class CompanyDecisionSummary(BaseModel):
    """Deterministic decision-intelligence summary derived by re-running the
    Decision Intelligence pipeline over the company's computed feature set.

    The summary is a read-only join onto one deterministic snapshot of
    features (``source_record_id`` identifies exactly which offline feature
    snapshot produced it).
    """

    verdict: str = Field(..., description="Investment verdict of the decision trace")
    confidence: float = Field(..., ge=0, le=1, description="Decision confidence (0-1)")
    overall_score: float = Field(..., description="Composite score of the decision trace")
    feature_count: int = Field(..., ge=0, description="Features contributing to the decision")
    positive_factor_count: int = Field(..., ge=0, description="Supporting contributions")
    negative_factor_count: int = Field(..., ge=0, description="Detracting contributions")
    node_count: int = Field(..., ge=0, description="Nodes in the decision reasoning trace")
    edge_count: int = Field(..., ge=0, description="Edges in the decision reasoning trace")
    trace_id: str | None = Field(default=None, description="Decision trace identifier")
    top_strengths: list[str] = Field(
        default_factory=list, description="Top supporting factor names"
    )
    top_weaknesses: list[str] = Field(
        default_factory=list, description="Top detracting factor names"
    )
    source_record_id: str | None = Field(
        default=None, description="Offline dataset record feeding the trace (feature snapshot)"
    )


class CompanyProfileResponse(BaseModel):
    """Unified deterministic profile of one company.

    Composition rules
    -----------------
    * ``coverage == live`` — ``latest_snapshot`` and ``history`` reflect the
      Phase 1 registry; offline summaries may also be present when grounded
      records match.
    * ``coverage == offline`` — no live snapshots; offline summaries derive
      from grounded dataset records.
    * ``coverage == offline_placeholder`` — offline records exist but are
      placeholders; intelligence summaries are omitted so placeholders are
      never presented as live intelligence.
    * ``coverage == none`` — nothing is known about the company.
    """

    company_id: str = Field(..., description="Deterministic company identifier")
    canonical_name: str = Field(..., description="Normalized canonical company name")
    primary_name: str = Field(..., description="Display name as last submitted")
    canonical_domain: str | None = Field(default=None, description="Normalized canonical domain")
    website: str | None = Field(default=None, description="Display website URL")
    coverage: CoverageStatus = Field(..., description="Coverage classification (see enum)")
    coverage_reasons: list[str] = Field(
        default_factory=list, description="Why the coverage classification was chosen"
    )
    completeness: ProfileCompleteness = Field(..., description="Field-completeness")
    latest_snapshot: CompanySnapshotResponse | None = Field(
        default=None, description="Latest live snapshot, when grounded"
    )
    history: CompanyHistorySummary = Field(
        default_factory=CompanyHistorySummary, description="Live registry history summary"
    )
    offline: CompanyProfileOfflineSummary | None = Field(
        default=None, description="Historical dataset summary, when records match"
    )
    graph_summary: CompanyGraphSummary | None = Field(
        default=None, description="Knowledge-graph summary, when grounded offline data exists"
    )
    feature_summary: CompanyFeatureSummary | None = Field(
        default=None, description="Feature summary, when grounded offline data exists"
    )
    signal_summary: CompanySignalSummary | None = Field(
        default=None, description="Signal-timeline summary, when grounded offline data exists"
    )
    benchmark_summary: CompanyBenchmarkSummary | None = Field(
        default=None, description="Benchmark placement of the company's composite score"
    )
    decision_summary: CompanyDecisionSummary | None = Field(
        default=None, description="Decision-trace summary derived from the offline feature snapshot"
    )
    generated_at: datetime = Field(
        ..., description="Reference timestamp of the profile (deterministic)"
    )


__all__ = [
    "CompanyBenchmarkSummary",
    "CompanyDecisionSummary",
    "CompanyFeatureItem",
    "CompanyFeatureSummary",
    "CompanyGraphNeighbor",
    "CompanyGraphSummary",
    "CompanyHistorySummary",
    "CompanyProfileOfflineSummary",
    "CompanyProfileResponse",
    "CompanySignalSummary",
    "CoverageStatus",
    "ProfileCompleteness",
]
