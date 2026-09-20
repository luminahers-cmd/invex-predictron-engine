"""Company Intelligence Hub (CIH) — Phase 4 validity-loop persistence models.

Additive tables that record real-world outcomes for companies and the
append-only prediction evaluations derived from them. The rows store the
exact shapes of the engine's benchmark models (:class:`StartupOutcome` and
:class:`PredictionSummary`) so the existing evaluation machinery in
:mod:`predictron_engine.dataset` can be reused verbatim — no evaluation
logic is duplicated here.

Design rules (additive only):
- ``companies`` / ``company_snapshots`` are untouched; snapshots stay
  immutable.
- Outcomes are observed facts; the verdict is derived, never predicted.
- Evaluations are append-only rows with deterministic ids, so re-computing
  an evaluation (for example when a later snapshot becomes applicable to an
  earlier outcome) is an idempotent no-op.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def utc_now() -> datetime:
    """Current UTC timestamp used as a safe Python-side column default."""
    return datetime.now(UTC)


class CompanyOutcome(Base):
    """One recorded real-world outcome for a company.

    ``id`` is a deterministic SHA-256 digest of ``(company_id, snapshot_id,
    source, occurred_at, canonical outcome payload)`` so re-submitting the
    same outcome is an idempotent no-op — mirroring the deterministic-id
    pattern used by the Phase 1 registry and its snapshot history.
    """

    __tablename__ = "company_outcomes"
    __table_args__ = (
        Index("ix_company_outcomes_company_created", "company_id", "created_at"),
        Index("ix_company_outcomes_snapshot_id", "snapshot_id"),
        Index("ix_company_outcomes_verdict", "verdict"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    company_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("companies.company_id", ondelete="CASCADE"),
        nullable=False,
    )
    snapshot_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("company_snapshots.id", ondelete="SET NULL"),
        nullable=True,
        default=None,
    )
    source: Mapped[str] = mapped_column(
        String(255), nullable=False, default="manual", server_default="manual"
    )
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    time_horizon_days: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=None
    )
    outcome_data: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    verdict: Mapped[str] = mapped_column(String(50), nullable=False)
    verdict_reasoning: Mapped[str] = mapped_column(
        Text, nullable=False, default="", server_default=""
    )
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="unknown", server_default="unknown"
    )
    notes: Mapped[str] = mapped_column(
        Text, nullable=False, default="", server_default=""
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class CompanyEvaluation(Base):
    """Append-only evaluation of one prediction snapshot against one outcome.

    ``id`` is a deterministic SHA-256 digest of ``(company_id, snapshot_id,
    outcome_id)`` so re-computing an evaluation is an idempotent no-op; a
    unique ``(snapshot_id, outcome_id)`` index is a second, explicit guard.

    ``prediction`` is the frozen :class:`PredictionSummary` captured from the
    immutable snapshot at evaluation time and is never modified afterwards.
    """

    __tablename__ = "company_evaluations"
    __table_args__ = (
        Index("ix_company_evaluations_company_created", "company_id", "created_at"),
        Index("ix_company_evaluations_company_verdict", "company_id", "verdict"),
        Index("ix_company_evaluations_outcome_id", "outcome_id"),
        Index(
            "uq_company_evaluations_snapshot_outcome",
            "snapshot_id",
            "outcome_id",
            unique=True,
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    company_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("companies.company_id", ondelete="CASCADE"),
        nullable=False,
    )
    snapshot_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("company_snapshots.id", ondelete="CASCADE"),
        nullable=False,
    )
    outcome_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("company_outcomes.id", ondelete="CASCADE"),
        nullable=False,
    )
    prediction: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    verdict: Mapped[str] = mapped_column(String(50), nullable=False)
    alignment: Mapped[str] = mapped_column(
        String(50), nullable=False, default="neutral", server_default="neutral"
    )
    outcome_verdict: Mapped[str] = mapped_column(String(50), nullable=False)
    decision_match: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True, default=None
    )
    snapshot_confidence: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    snapshot_composite_score: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=None
    )
    snapshot_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


__all__ = ["CompanyEvaluation", "CompanyOutcome", "utc_now"]
