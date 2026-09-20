"""Pydantic schemas for the Company Intelligence Hub (Phase 1).

These are read-side response schemas. Requests are written by the ingest
service from completed analyses; there is no public write endpoint.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class CompanySummaryResponse(BaseModel):
    """Compact company record for list endpoints."""

    company_id: str = Field(..., description="Deterministic company identifier")
    canonical_name: str = Field(..., description="Normalized canonical company name")
    canonical_domain: str | None = Field(
        default=None, description="Normalized canonical domain (host, no www)"
    )
    primary_name: str = Field(..., description="Display name as last submitted")
    website: str | None = Field(default=None, description="Display website URL")
    latest_decision: str | None = Field(
        default=None, description="Decision category of the latest analysis"
    )
    latest_confidence: float | None = Field(
        default=None, ge=0, le=1, description="Confidence of the latest analysis"
    )
    latest_composite_score: float | None = Field(
        default=None, ge=0, le=100, description="Composite score of the latest analysis"
    )
    snapshot_count: int = Field(default=0, ge=0, description="Number of analysis snapshots")
    first_seen: datetime = Field(..., description="When the company was first seen")
    last_seen: datetime = Field(..., description="When the company was last seen")


class CompanySnapshotResponse(BaseModel):
    """One immutable analysis snapshot in a company's history."""

    id: str = Field(..., description="Deterministic snapshot identifier")
    company_id: str = Field(..., description="Owning company identifier")
    analysis_id: str = Field(..., description="Source analysis request ID")
    report_id: str = Field(..., description="Source analysis report ID")
    decision: str | None = Field(
        default=None, description="Decision category of the analysis"
    )
    confidence: float | None = Field(
        default=None, ge=0, le=1, description="Analysis confidence"
    )
    composite_score: float | None = Field(
        default=None, ge=0, le=100, description="Composite investment score"
    )
    readiness_score: float | None = Field(
        default=None, ge=0, le=100, description="Investment readiness score"
    )
    dimension_scores: dict[str, float] = Field(
        default_factory=dict, description="Per-dimension scores"
    )
    created_at: datetime = Field(..., description="When the analysis was completed")


class CompanyDetailResponse(CompanySummaryResponse):
    """Company metadata plus its latest snapshot."""

    latest_snapshot: CompanySnapshotResponse | None = Field(
        default=None, description="Most recent analysis snapshot, if any"
    )


class CompanyListResponse(BaseModel):
    """Paginated list of companies."""

    companies: list[CompanySummaryResponse] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)


class CompanyHistoryResponse(BaseModel):
    """Ordered (newest-first) snapshot history for one company."""

    company_id: str
    snapshots: list[CompanySnapshotResponse] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
