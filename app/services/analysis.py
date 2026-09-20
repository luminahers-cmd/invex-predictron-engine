"""Analysis service — bridges the API layer and PredictronEngine.

This module owns the conversion between the API request/response schemas
and the internal PredictronEngine. It runs the synchronous engine in a
thread pool via asyncio.to_thread() to avoid blocking the event loop.

After successful engine execution, completed analyses are persisted to the
database via the persistence service. Persistence failures are logged but
never propagated to the caller — a successful analysis always returns its
response regardless of storage status.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from app.models.analysis import AnalysisRequest
from app.schemas.analysis import StartupAnalysisRequest, StartupAnalysisResponse

if TYPE_CHECKING:
    from predictron_engine.engine import PredictronEngine
    from predictron_engine.models.report import Report

logger = logging.getLogger(__name__)

async def run_analysis(
    engine: PredictronEngine,
    request: StartupAnalysisRequest,
    user_id: str | None = None,
) -> StartupAnalysisResponse:
    """Execute the analysis pipeline and return an API response.

    The synchronous PredictronEngine.analyze() is offloaded to a thread
    so the asyncio event loop remains unblocked.

    After successful execution, the completed analysis is persisted in a
    background fire-and-forget fashion. Persistence errors are logged
    but do not affect the API response.

    Args:
        user_id: Optional authenticated user to associate with the persisted analysis.
    """
    report = await asyncio.to_thread(engine.analyze, request)
    response = _report_to_response(request.startup_name, report)

    persisted: AnalysisRequest | None = None
    try:
        persisted = await _persist_async(request, report, response, user_id=user_id)
        if persisted is not None and isinstance(getattr(persisted, "id", None), str):
            response.id = persisted.id
    except Exception:
        logger.warning("Persistence layer error (analysis still returned)", exc_info=True)

    if persisted is not None:
        await _ingest_company_hook(persisted, request, report, response, user_id=user_id)

    return response


async def _persist_async(
    request: StartupAnalysisRequest,
    report: Report,
    response: StartupAnalysisResponse,
    user_id: str | None = None,
) -> AnalysisRequest | None:
    """Persist a completed analysis. Errors are logged and swallowed.

    Returns:
        The persisted AnalysisRequest when successful, otherwise None.
    """
    from app.db.session import AsyncSessionLocal
    from app.services.persistence import persist_analysis

    try:
        async with AsyncSessionLocal() as session:
            db_request = await persist_analysis(
                session, request, report, response, user_id=user_id
            )
            await session.commit()
        logger.debug("Analysis persisted successfully")
        return db_request
    except Exception:
        logger.warning("Failed to persist analysis", exc_info=True)
        return None


async def _ingest_company_hook(
    analysis: AnalysisRequest,
    request: StartupAnalysisRequest,
    report: Report,
    response: StartupAnalysisResponse,
    user_id: str | None = None,
) -> None:
    """Company Intelligence Hub hook — runs after analysis persistence.

    This is the single integration point for CIH Phase 1. It is deliberately
    small and additive: it resolves the company identity, upserts the
    company, and appends one immutable snapshot. Failures are logged and
    swallowed so a registry hiccup never masks a successful analysis.
    """
    from app.db.session import AsyncSessionLocal
    from app.services.companies import (
        CompanyIngestService,
        build_snapshot_inputs,
    )

    try:
        inputs = build_snapshot_inputs(request, report, response)
        service = CompanyIngestService()
        async with AsyncSessionLocal() as session:
            await service.ingest_after_persist(
                session, analysis=analysis, inputs=inputs, user_id=user_id
            )
            await session.commit()
        logger.debug("Company registry ingest completed for analysis %s", analysis.id)
    except Exception:
        logger.warning(
            "Company registry ingest failed (analysis still persisted)",
            exc_info=True,
        )


def _report_to_response(
    startup_name: str,
    report: object,
) -> StartupAnalysisResponse:
    """Convert a PredictronEngine Report into an API response."""
    from predictron_engine.models.report import Report

    assert isinstance(report, Report)

    scores_by_dim = {s.dimension: s.score for s in report.scores}

    return StartupAnalysisResponse(
        startup_name=startup_name,
        venture_score=report.overall_score,
        market_score=scores_by_dim.get("market_opportunity", 0.0),
        founder_score=scores_by_dim.get("founder_quality", 0.0),
        traction_score=scores_by_dim.get("traction_signals", 0.0),
        recommendations=[r.action for r in report.recommendations],
        confidence=report.overall_confidence,
    )
