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

from app.schemas.analysis import StartupAnalysisRequest, StartupAnalysisResponse

if TYPE_CHECKING:
    from predictron_engine.engine import PredictronEngine

logger = logging.getLogger(__name__)

_SCORE_DIMENSION_MAP: dict[str, str] = {
    "market_score": "market_opportunity",
    "founder_score": "founder_quality",
    "traction_score": "traction_signals",
}


async def run_analysis(
    engine: PredictronEngine,
    request: StartupAnalysisRequest,
) -> StartupAnalysisResponse:
    """Execute the analysis pipeline and return an API response.

    The synchronous PredictronEngine.analyze() is offloaded to a thread
    so the asyncio event loop remains unblocked.

    After successful execution, the completed analysis is persisted in a
    background fire-and-forget fashion. Persistence errors are logged
    but do not affect the API response.
    """
    report = await asyncio.to_thread(engine.analyze, request)
    response = _report_to_response(request.startup_name, report)

    try:
        await _persist_async(request, report, response)
    except Exception:
        logger.warning("Persistence layer error (analysis still returned)", exc_info=True)

    return response


async def _persist_async(
    request: StartupAnalysisRequest,
    report: object,
    response: StartupAnalysisResponse,
) -> None:
    """Persist a completed analysis. Errors are logged and swallowed."""
    from app.db.session import AsyncSessionLocal
    from app.services.persistence import persist_analysis

    try:
        async with AsyncSessionLocal() as session:
            await persist_analysis(session, request, report, response)
            await session.commit()
        logger.debug("Analysis persisted successfully")
    except Exception:
        logger.warning("Failed to persist analysis", exc_info=True)


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
