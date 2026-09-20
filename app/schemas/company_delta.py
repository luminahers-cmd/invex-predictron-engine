"""Pydantic schemas for the Company Intelligence Hub delta layer (Phase 3).

Deterministic, field-level snapshot comparison. Every ``CompanyDelta``
compares one earlier snapshot against one later snapshot and reports, per
tracked field, the previous/current values and a machine-readable status and
``change`` marker. Values and markers derive exclusively from stored
snapshots and their persisted analysis reports — nothing is generated or
predicted.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.company_history import CompanyTrendSummary


class DeltaKind(str, Enum):
    """How a compared field's values are treated."""

    NUMERIC = "numeric"
    NOMINAL = "nominal"
    COLLECTION = "collection"


class DeltaStatus(str, Enum):
    """Outcome of comparing one field between two snapshots.

    ``added`` / ``removed`` describe one-sided presence; ``missing`` means
    neither snapshot carries the field; ``changed`` / ``unchanged`` compare
    two present values.
    """

    CHANGED = "changed"
    UNCHANGED = "unchanged"
    ADDED = "added"
    REMOVED = "removed"
    MISSING = "missing"


class DeltaField(BaseModel):
    """One field-level comparison between two snapshots."""

    field: str = Field(..., description="Tracked field name")
    delta_type: DeltaKind = Field(..., description="Comparison semantics")
    status: DeltaStatus = Field(..., description="Comparison outcome")
    previous: Any = Field(default=None, description="Value on the earlier snapshot")
    current: Any = Field(default=None, description="Value on the later snapshot")
    change: float | str | None = Field(
        default=None,
        description="Numeric delta (current - previous) or \"same\"/\"different\" marker",
    )


class CompanyDelta(BaseModel):
    """Structured comparison of two snapshots (earlier -> later).

    ``previous_snapshot_id`` / ``current_snapshot_id`` name the compared
    snapshots explicitly so the ordering is never ambiguous regardless of the
    order the deltas are served in.
    """

    previous_snapshot_id: str
    current_snapshot_id: str
    previous_created_at: datetime | None = Field(
        default=None, description="Timestamp of the earlier snapshot"
    )
    current_created_at: datetime = Field(
        ..., description="Timestamp of the later snapshot"
    )
    fields: list[DeltaField] = Field(default_factory=list)
    changed_fields: int = Field(default=0, ge=0, description="Fields that changed")
    unchanged_fields: int = Field(default=0, ge=0, description="Fields that stayed equal")


class CompanyDeltaListResponse(BaseModel):
    """Paginated delta sequence (newest-first) plus the global trend."""

    company_id: str = Field(..., description="Deterministic company identifier")
    total: int = Field(default=0, ge=0, description="Snapshots pairs compared")
    deltas: list[CompanyDelta] = Field(default_factory=list)
    trend: CompanyTrendSummary = Field(..., description="Trend over the full timeline")


__all__ = [
    "CompanyDelta",
    "CompanyDeltaListResponse",
    "DeltaField",
    "DeltaKind",
    "DeltaStatus",
]
