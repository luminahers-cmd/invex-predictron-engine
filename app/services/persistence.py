"""Persistence service — stores and retrieves completed analyses.

This module is the single source of truth for database operations related
to analysis requests and reports. It is separated from both the API layer
and the PredictronEngine. All functions accept an explicit AsyncSession
for testability and composability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import desc, select

from app.models.analysis import AnalysisReport, AnalysisRequest
from app.schemas.analysis import (
    AnalysisDetailResponse,
    AnalysisListResponse,
    AnalysisSummaryResponse,
    StartupAnalysisRequest,
    StartupAnalysisResponse,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from predictron_engine.models.report import Report

logger = logging.getLogger(__name__)


async def persist_analysis(
    session: AsyncSession,
    request: StartupAnalysisRequest,
    report: Report,
    response: StartupAnalysisResponse,
    user_id: str | None = None,
) -> AnalysisRequest:
    """Persist a successfully completed analysis.

    Both the request and the report are written atomically.
    This function should ONLY be called after the engine pipeline
    completes without error — partial or failed analyses are never stored.

    Args:
        user_id: Optional authenticated user identifier to associate with the analysis.

    Returns:
        The persisted AnalysisRequest record.
    """
    db_request = AnalysisRequest(
        user_id=user_id,
        startup_name=request.startup_name,
        website=str(request.website) if request.website else "",
        description=request.description,
        pitch_deck_url=str(request.pitch_deck_url) if request.pitch_deck_url else None,
        founder_linkedin_urls=[str(u) for u in request.founder_linkedin_urls],
    )
    session.add(db_request)
    await session.flush()

    db_report = AnalysisReport(
        request_id=db_request.id,
        startup_name=response.startup_name,
        venture_score=response.venture_score,
        market_score=response.market_score,
        founder_score=response.founder_score,
        traction_score=response.traction_score,
        recommendations=response.recommendations,
        confidence=response.confidence,
        engine_version=report.analysis_metadata.engine_version,
        processing_time_ms=report.analysis_metadata.processing_time_ms,
        full_report=report.model_dump(mode="json"),
    )
    session.add(db_report)
    await session.flush()

    logger.info("Persisted analysis %s for '%s'", db_request.id, request.startup_name)
    return db_request


async def get_analysis(
    session: AsyncSession,
    analysis_id: str,
    user_id: str | None = None,
) -> AnalysisDetailResponse | None:
    """Retrieve a single completed analysis by ID.

    When user_id is provided, only returns the analysis if it belongs
    to that user. Admin/superuser flows are out of scope — this is a
    strict ownership gate.

    Returns:
        An AnalysisDetailResponse if found, otherwise None.
    """
    stmt = (
        select(AnalysisRequest, AnalysisReport)
        .join(AnalysisReport, AnalysisRequest.id == AnalysisReport.request_id)
        .where(AnalysisRequest.id == analysis_id)
    )
    if user_id is None:
        stmt = stmt.where(AnalysisRequest.user_id.is_(None))
    else:
        stmt = stmt.where(AnalysisRequest.user_id == user_id)
    result = await session.execute(stmt)
    row = result.one_or_none()

    if row is None:
        return None

    db_request, db_report = row
    return AnalysisDetailResponse(
        id=db_request.id,
        startup_name=db_request.startup_name,
        website=db_request.website,
        description=db_request.description,
        pitch_deck_url=db_request.pitch_deck_url,
        founder_linkedin_urls=db_request.founder_linkedin_urls,
        venture_score=db_report.venture_score,
        market_score=db_report.market_score,
        founder_score=db_report.founder_score,
        traction_score=db_report.traction_score,
        recommendations=db_report.recommendations,
        confidence=db_report.confidence,
        engine_version=db_report.engine_version,
        processing_time_ms=db_report.processing_time_ms,
        full_report=db_report.full_report,
        created_at=db_request.created_at,
    )


async def list_analyses(
    session: AsyncSession,
    user_id: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> AnalysisListResponse:
    from sqlalchemy import func

    count_over = func.count().over().label("total_count")

    from sqlalchemy.orm import load_only

    stmt = (
        select(AnalysisRequest, AnalysisReport, count_over)
        .join(AnalysisReport, AnalysisRequest.id == AnalysisReport.request_id)
        .options(
            load_only(
                AnalysisReport.id,
                AnalysisReport.startup_name,
                AnalysisReport.venture_score,
                AnalysisReport.market_score,
                AnalysisReport.founder_score,
                AnalysisReport.traction_score,
                AnalysisReport.confidence,
                AnalysisReport.engine_version,
                AnalysisReport.processing_time_ms,
                AnalysisReport.created_at,
                AnalysisReport.request_id,
            )
        )
        .order_by(desc(AnalysisRequest.created_at))
        .offset(offset)
        .limit(limit)
    )
    if user_id is not None:
        stmt = stmt.where(AnalysisRequest.user_id == user_id)

    result = await session.execute(stmt)
    rows = result.all()

    total = 0
    summaries = []
    for db_request, db_report, total_count in rows:
        total = total_count
        summaries.append(
            AnalysisSummaryResponse(
                id=db_request.id,
                startup_name=db_request.startup_name,
                venture_score=db_report.venture_score,
                market_score=db_report.market_score,
                founder_score=db_report.founder_score,
                traction_score=db_report.traction_score,
                confidence=db_report.confidence,
                engine_version=db_report.engine_version,
                processing_time_ms=db_report.processing_time_ms,
                created_at=db_request.created_at,
            )
        )

    return AnalysisListResponse(analyses=summaries, total=total)
