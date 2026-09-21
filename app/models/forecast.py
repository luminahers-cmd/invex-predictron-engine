"""Live Prediction Ledger models (CIH Phase 5).

Additive persistence for the deterministic forecast lifecycle. Two tables:

* ``forecasts`` — one frozen forecast per ``(snapshot, horizon)``. The
  primary key is a deterministic SHA-256 of ``(company_id, snapshot_id,
  horizon_days)`` so re-registration is an idempotent no-op. The
  ``prediction`` column stores a frozen :class:`PredictionSummary` JSON
  snapshot that is never modified after registration; ``engine_version`` and
  ``schema_version`` are immutable row metadata pinned at registration.
* ``forecast_events`` — an append-only lifecycle event log
  (``registered`` / ``due`` / ``resolved``) with deterministic event IDs, so
  re-deriving a status never emits a duplicate event.

``status`` is service-owned: it is always derived by
:func:`~predictron_engine.dataset.forecast_lifecycle.derive_forecast_status`
and then persisted — the ledger never trusts a caller-supplied status.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def utc_now() -> datetime:
    """Current UTC timestamp used as a safe Python-side column default."""
    return datetime.now(UTC)


class Forecast(Base):
    """One frozen prediction awaited against an evaluation horizon."""

    __tablename__ = "forecasts"
    __table_args__ = (
        Index("ix_forecasts_company_created", "company_id", "created_at"),
        Index("ix_forecasts_status", "status"),
        Index("ix_forecasts_company_status", "company_id", "status"),
        Index("ix_forecasts_due_at", "due_at"),
        Index("ix_forecasts_outcome_id", "outcome_id"),
        Index(
            "uq_forecasts_snapshot_horizon",
            "snapshot_id",
            "horizon_days",
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
    outcome_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("company_outcomes.id", ondelete="SET NULL"),
        nullable=True,
        default=None,
    )
    analysis_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    decision: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False)
    prediction: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    engine_version: Mapped[str] = mapped_column(String(50), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(50), nullable=False)
    reconciled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
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


class ForecastEvent(Base):
    """Append-only lifecycle event against one forecast."""

    __tablename__ = "forecast_events"
    __table_args__ = (
        Index(
            "ix_forecast_events_forecast_occurred",
            "forecast_id",
            "occurred_at",
        ),
        Index("ix_forecast_events_type", "event_type"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    forecast_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("forecasts.id", ondelete="CASCADE"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    payload: Mapped[dict[str, object] | None] = mapped_column(
        JSON, nullable=True, default=None
    )


__all__ = ["Forecast", "ForecastEvent", "utc_now"]
