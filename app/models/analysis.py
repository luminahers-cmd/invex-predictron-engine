from __future__ import annotations

import uuid

from sqlalchemy import JSON, Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import TimestampMixin
from app.db.session import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class AnalysisRequest(Base, TimestampMixin):
    __tablename__ = "analysis_requests"
    __table_args__ = (
        Index("ix_analysis_requests_created_at", "created_at"),
        Index("ix_analysis_requests_user_created", "user_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    user_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True, default=None, index=True
    )
    startup_name: Mapped[str] = mapped_column(String(255), nullable=False)
    website: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    pitch_deck_url: Mapped[str | None] = mapped_column(
        String(500), nullable=True, default=None
    )
    founder_linkedin_urls: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )

    report: Mapped[AnalysisReport | None] = relationship(
        back_populates="request", uselist=False
    )


class AnalysisReport(Base, TimestampMixin):
    __tablename__ = "analysis_reports"
    __table_args__ = (
        Index("ix_analysis_reports_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )
    request_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("analysis_requests.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    startup_name: Mapped[str] = mapped_column(String(255), nullable=False)
    venture_score: Mapped[float] = mapped_column(Float, nullable=False)
    market_score: Mapped[float] = mapped_column(Float, nullable=False)
    founder_score: Mapped[float] = mapped_column(Float, nullable=False)
    traction_score: Mapped[float] = mapped_column(Float, nullable=False)
    recommendations: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    engine_version: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )
    processing_time_ms: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    full_report: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)

    request: Mapped[AnalysisRequest] = relationship(back_populates="report")
