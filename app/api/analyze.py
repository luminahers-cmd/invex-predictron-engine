"""Analysis endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt import get_current_user, get_current_user_optional
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
    description="Submit a startup, receive a venture analysis from Predictron. Auth optional.",
    responses={
        200: {"description": "Analysis completed successfully", "model": StartupAnalysisResponse},
        422: {"description": "Validation error — check request body fields"},
        429: {"description": "Rate limit exceeded"},
    },

)
async def analyze_startup(
    request: StartupAnalysisRequest,
    raw_request: Request,
    current_user: dict | None = Depends(get_current_user_optional),
) -> StartupAnalysisResponse:
    """Delegate analysis to the PredictronEngine singleton via the service layer.

    Authenticated users have their analyses associated with their user account.
    Anonymous users can still run analyses without persistence association.
    """
    engine = raw_request.app.state.predictron_engine
    user_id = current_user.get("sub") if current_user else None
    return await run_analysis(engine, request, user_id=user_id)


@router.get(
    "",
    response_model=AnalysisListResponse,
    status_code=status.HTTP_200_OK,
    summary="List persisted analyses",
    description="List your completed analyses. Requires authentication — only yours are returned.",
    responses={
        200: {"description": "Paginated list of analyses", "model": AnalysisListResponse},
        401: {"description": "Not authenticated — missing or invalid token"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def list_analyses(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return"),
) -> AnalysisListResponse:
    from app.services.persistence import list_analyses as _list

    user_id = current_user.get("sub")
    return await _list(db, user_id=user_id, offset=offset, limit=limit)


@router.get(
    "/{analysis_id}",
    response_model=AnalysisDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a persisted analysis",
    description="Get a completed analysis by ID. Requires authentication — owner only.",
    responses={
        200: {"description": "Full analysis details", "model": AnalysisDetailResponse},
        401: {"description": "Not authenticated — missing or invalid token"},
        404: {"description": "Analysis not found or access denied"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_analysis(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
) -> AnalysisDetailResponse:
    from app.services.persistence import get_analysis as _get

    user_id = current_user.get("sub")
    result = await _get(db, analysis_id, user_id=user_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis {analysis_id} not found",
        )
    return result
