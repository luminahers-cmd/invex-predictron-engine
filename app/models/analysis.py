"""SQLAlchemy models for analysis request and report persistence."""

from __future__ import annotations

import uuid

from sqlalchemy import JSON, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import TimestampMixin
from app.db.session import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class AnalysisRequest(Base, TimestampMixin):
    """Persisted analysis request — the input to the engine pipeline."""

    __tablename__ = "analysis_requests"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    startup_name: Mapped[str] = mapped_column(String(255), nullable=False)
    website: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    pitch_deck_url: Mapped[str | None] = mapped_column(
        String(500), nullable=True, default=None
    )
    founder_linkedin_urls: Mapped[list] = mapped_column(
        JSON, nullable=False, default=list
    )

    report: Mapped[AnalysisReport | None] = relationship(
        back_populates="request", uselist=False
    )


class AnalysisReport(Base, TimestampMixin):
    """Persisted analysis report — the completed engine output.

    Only written after the engine pipeline completes successfully.
    Partial or failed analyses are never persisted.
    """

    __tablename__ = "analysis_reports"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("analysis_requests.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    startup_name: Mapped[str] = mapped_column(String(255), nullable=False)
    venture_score: Mapped[float] = mapped_column(Float, nullable=False)
    market_score: Mapped[float] = mapped_column(Float, nullable=False)
    founder_score: Mapped[float] = mapped_column(Float, nullable=False)
    traction_score: Mapped[float] = mapped_column(Float, nullable=False)
    recommendations: Mapped[list] = mapped_column(
        JSON, nullable=False, default=list
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    engine_version: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )
    processing_time_ms: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    full_report: Mapped[dict] = mapped_column(JSON, nullable=False)

    request: Mapped[AnalysisRequest] = relationship(back_populates="report")
