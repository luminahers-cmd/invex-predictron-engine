"""Retrieve persisted analyses."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.analysis import (
    AnalysisDetailResponse,
    AnalysisListResponse,
    StartupAnalysisRequest,
    StartupAnalysisResponse,
)
from app.services.analysis import run_analysis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analysis"])


@router.post(
    "",
    response_model=StartupAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze a startup venture",
    description="Submit startup info and receive a venture analysis.",
)
async def analyze_startup(
    request: StartupAnalysisRequest,
    raw_request: Request,
) -> StartupAnalysisResponse:
    """Delegate analysis to the PredictronEngine singleton via the service layer."""
    engine = raw_request.app.state.predictron_engine
    return await run_analysis(engine, request)


@router.get(
    "",
    response_model=AnalysisListResponse,
    status_code=status.HTTP_200_OK,
    summary="List persisted analyses",
    description="Retrieve a paginated list of completed analyses.",
)
async def list_analyses(
    db: AsyncSession = Depends(get_db),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return"),
) -> AnalysisListResponse:
    from app.services.persistence import list_analyses as _list

    return await _list(db, offset=offset, limit=limit)


@router.get(
    "/{analysis_id}",
    response_model=AnalysisDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a persisted analysis",
    description="Retrieve the full detail of a completed analysis by ID.",
)
async def get_analysis(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
) -> AnalysisDetailResponse:
    from app.services.persistence import get_analysis as _get

    result = await _get(db, analysis_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis {analysis_id} not found",
        )
    return result
