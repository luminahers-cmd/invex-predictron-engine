"""Continuous Learning Intelligence persistence models (CIH Phase 7).

Append-only persistence for the deterministic learning layer:

* ``learning_snapshots`` — one recorded learning snapshot per
  ``(scope, period_kind, anchor_date)``.  The primary key is a deterministic
  SHA-256 digest so re-recording the same anchor is an idempotent no-op.
* ``learning_observations`` — the canonical observations rows of a recorded
  snapshot (attribute resolution + accuracy/confidence signals).
* ``learning_patterns`` — the per-dimension cohort patterns of a recorded
  snapshot (accuracy, confusion counts, confidence bias).
* ``learning_reports`` — the full serialized :class:`LearningSnapshot`
  payload with an integrity ``content_hash``.

All IDs are deterministic and derived from content — no UUIDs, no timestamps
in primary keys.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def utc_now() -> datetime:
    """Current UTC timestamp used as a safe Python-side column default."""
    return datetime.now(UTC)


class LearningSnapshotRecord(Base):
    """Header row of one recorded learning snapshot."""

    __tablename__ = "learning_snapshots"
    __table_args__ = (
        Index("ix_learning_snapshots_scope_anchor", "scope", "anchor_date"),
        Index("ix_learning_snapshots_created", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scope: Mapped[str] = mapped_column(String(255), nullable=False)
    period_kind: Mapped[str] = mapped_column(String(20), nullable=False)
    anchor_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    engine_version: Mapped[str] = mapped_column(String(50), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(20), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    counts: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False, default=dict)
    metrics: Mapped[dict[str, float | None]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    distributions: Mapped[dict[str, dict[str, int]]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    meta: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False, default=dict)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class LearningObservationRecord(Base):
    """One canonical observation row of a recorded learning snapshot."""

    __tablename__ = "learning_observations"
    __table_args__ = (
        Index("ix_learning_observations_snapshot_id", "snapshot_id"),
        Index("ix_learning_observations_category", "category"),
        Index("ix_learning_observations_dimension", "dimension"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    dimension: Mapped[str] = mapped_column(String(50), nullable=False)
    value: Mapped[str] = mapped_column(String(255), nullable=False)
    metric: Mapped[str] = mapped_column(String(50), nullable=False)
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    direction: Mapped[str] = mapped_column(String(20), nullable=False, default="flat")
    baseline: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class LearningPatternRecord(Base):
    """One per-dimension cohort pattern row of a recorded learning snapshot."""

    __tablename__ = "learning_patterns"
    __table_args__ = (
        Index("ix_learning_patterns_snapshot_id", "snapshot_id"),
        Index("ix_learning_patterns_dimension_value", "dimension", "value"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    dimension: Mapped[str] = mapped_column(String(50), nullable=False)
    value: Mapped[str] = mapped_column(String(255), nullable=False)
    samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    scoreable: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confidence_bias: Mapped[float | None] = mapped_column(Float, nullable=True)
    false_positive_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    false_negative_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    recommendation: Mapped[str] = mapped_column(Text, nullable=False, default="")
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class LearningReportRecord(Base):
    """Full serialized learning snapshot payload with integrity hash."""

    __tablename__ = "learning_reports"
    __table_args__ = (
        Index("ix_learning_reports_snapshot_id", "snapshot_id"),
        Index("ix_learning_reports_created", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


__all__ = [
    "LearningObservationRecord",
    "LearningPatternRecord",
    "LearningReportRecord",
    "LearningSnapshotRecord",
    "utc_now",
]
