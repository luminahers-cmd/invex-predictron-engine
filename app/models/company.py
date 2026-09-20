"""Company Intelligence Hub (CIH) persistence models.

Company entities are first-class, durable, deterministic identities that
accumulate analysis history through append-only snapshots. Company IDs are
never random UUIDs — they are SHA-256 digests of the canonical identity
material (canonical domain, canonical name, or fallback slug).

Phase 1 ships persistence only. No forecasting, ML, embeddings, graph, or
signal tables are introduced here.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import TimestampMixin
from app.db.session import Base


def utc_now() -> datetime:
    """Current UTC timestamp used as a safe Python-side column default."""
    return datetime.now(UTC)


class Company(Base, TimestampMixin):
    """One durable, deterministic company identity.

    ``company_id`` is derived from the highest-priority canonical identity
    signal (canonical domain > canonical company name > normalized fallback
    slug) and is stable across repeated analyses of the same company.

    ``user_id`` records the user that created the company record; it is set
    on first insert and intentionally not overwritten by later analyses so
    the registry keeps stable ownership.
    """

    __tablename__ = "companies"
    __table_args__ = (
        Index("ix_companies_user_created", "user_id", "created_at"),
        Index("ix_companies_last_seen", "last_seen"),
        Index("uq_companies_canonical_domain", "canonical_domain", unique=True),
        Index("ix_companies_canonical_name_key", "canonical_name_key"),
        Index("ix_companies_fallback_slug", "fallback_slug"),
    )

    company_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False)
    canonical_name_key: Mapped[str] = mapped_column(String(255), nullable=False)
    canonical_domain: Mapped[str | None] = mapped_column(
        String(255), nullable=True, default=None
    )
    fallback_slug: Mapped[str] = mapped_column(
        String(128), nullable=False, default="unknown", server_default="unknown"
    )
    primary_name: Mapped[str] = mapped_column(String(255), nullable=False)
    website: Mapped[str | None] = mapped_column(String(500), nullable=True, default=None)
    latest_decision: Mapped[str | None] = mapped_column(
        String(50), nullable=True, default=None
    )
    latest_confidence: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    latest_composite_score: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    snapshot_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    first_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    last_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    user_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True, default=None
    )


class CompanySnapshot(Base):
    """Append-only snapshot of one completed analysis for a company.

    Snapshots are immutable history: existing rows are never overwritten.
    The primary key is deterministic — a SHA-256 digest of
    ``(company_id, analysis_id)`` — so re-ingesting the same analysis ID is
    a no-op instead of a duplicate row. A unique constraint on
    ``analysis_id`` provides a second, explicit guard.
    """

    __tablename__ = "company_snapshots"
    __table_args__ = (
        Index("ix_company_snapshots_company_created", "company_id", "created_at"),
        Index("uq_company_snapshots_analysis_id", "analysis_id", unique=True),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    company_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("companies.company_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    analysis_id: Mapped[str] = mapped_column(String(36), nullable=False)
    report_id: Mapped[str] = mapped_column(String(36), nullable=False)
    decision: Mapped[str | None] = mapped_column(String(50), nullable=True, default=None)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    composite_score: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    readiness_score: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    dimension_scores: Mapped[dict[str, float]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
