"""Company Comparison API — compare two or more startups across all dimensions."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, status

from app.auth.jwt import get_current_user_optional
from app.schemas.comparison import ComparisonRequest, FullComparisonResponse
from app.services.comparison import compare_companies

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compare", tags=["company-comparison"])


@router.post(
    "",
    response_model=FullComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare companies",
    description=(
        "Compare two or more startups across features, decisions, "
        "contributions, signals, knowledge graph, and benchmarks."
    ),
    responses={
        200: {"description": "Comparison completed"},
        422: {"description": "Validation error — need at least 2 companies"},
    },
)
async def compare_startups(
    request: ComparisonRequest,
    raw_request: Request,
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> FullComparisonResponse:
    engine = raw_request.app.state.predictron_engine
    return await compare_companies(
        engine=engine,
        request=request,
    )
