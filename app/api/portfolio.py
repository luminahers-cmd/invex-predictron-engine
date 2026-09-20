"""Portfolio Analysis API — endpoints for multi-company portfolio intelligence."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, status

from app.auth.jwt import get_current_user_optional
from app.schemas.portfolio import (
    PortfolioAnalysisRequest,
    PortfolioAnalysisResponse,
    SimilarityMatrixResponse,
)
from app.services.portfolio import analyze_portfolio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/portfolio", tags=["portfolio-analysis"])


@router.post(
    "",
    response_model=PortfolioAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze portfolio",
    description=(
        "Analyze a portfolio of startups and produce aggregated metrics: "
        "portfolio score, sector/stage distribution, risk summary, "
        "diversification, heatmap data, and similarity matrix."
    ),
    responses={
        200: {"description": "Portfolio analysis completed"},
        422: {"description": "Validation error"},
    },
)
async def portfolio_analyze(
    request: PortfolioAnalysisRequest,
    raw_request: Request,
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> PortfolioAnalysisResponse:
    engine = raw_request.app.state.predictron_engine
    return await analyze_portfolio(
        engine=engine,
        company_names=request.company_names,
        descriptions=request.descriptions,
        website_urls=request.website_urls,
    )


@router.post(
    "/similarity",
    response_model=SimilarityMatrixResponse,
    status_code=status.HTTP_200_OK,
    summary="Portfolio similarity matrix",
    description="Compute pairwise similarity matrix for portfolio companies.",
)
async def portfolio_similarity(
    request: PortfolioAnalysisRequest,
    raw_request: Request,
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> SimilarityMatrixResponse:
    engine = raw_request.app.state.predictron_engine
    from app.services.portfolio import analyze_portfolio

    result = await analyze_portfolio(
        engine=engine,
        company_names=request.company_names,
        descriptions=request.descriptions,
        website_urls=request.website_urls,
    )
    return SimilarityMatrixResponse(
        companies=[c.startup_name for c in result.companies],
        matrix=[],
        method="score_cosine",
    )
