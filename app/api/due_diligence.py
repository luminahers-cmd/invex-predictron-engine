"""Due Diligence Report API — structured due diligence report generation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, status

from app.auth.jwt import get_current_user_optional
from app.schemas.due_diligence import (
    DueDiligenceReportResponse,
    DueDiligenceRequest,
)
from app.services.due_diligence import generate_due_diligence

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/due-diligence", tags=["due-diligence"])


@router.post(
    "",
    response_model=DueDiligenceReportResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate due diligence report",
    description=(
        "Generate a complete due diligence report containing: executive summary, "
        "strengths, weaknesses, opportunities, risks, evidence, decision trace, "
        "confidence, supporting features, and benchmark context."
    ),
    responses={
        200: {"description": "Report generated"},
        422: {"description": "Validation error"},
    },
)
async def due_diligence_generate(
    request: DueDiligenceRequest,
    raw_request: Request,
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> DueDiligenceReportResponse:
    engine = raw_request.app.state.predictron_engine
    return await generate_due_diligence(
        engine=engine,
        startup_name=request.startup_name,
        website_url=request.website_url,
        description=request.description,
        pitch_deck_url=request.pitch_deck_url,
        founder_linkedin_urls=request.founder_linkedin_urls,
    )
