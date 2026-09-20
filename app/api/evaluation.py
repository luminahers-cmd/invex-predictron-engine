"""Global validity-loop API endpoints (CIH Phase 4).

Read-only calibration and summary views across every company in the
caller's scope. Company-scoped outcome recording and performance live in
:mod:`app.api.companies`; this router exposes the cross-company aggregates.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt import get_current_user_optional
from app.db.session import get_db
from app.schemas.evaluation import (
    BenchmarkAccuracyResponse,
    EvaluationCalibrationResponse,
    EvaluationSummaryResponse,
)
from app.services.benchmark_accuracy import BenchmarkAccuracyService
from app.services.company_evaluation import CompanyEvaluationService

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def _user_id(current_user: dict[str, object] | None) -> str | None:
    sub = current_user.get("sub") if current_user else None
    return str(sub) if sub is not None else None


@router.get(
    "/calibration",
    response_model=EvaluationCalibrationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get prediction calibration across your companies",
    description=(
        "Expected Calibration Error, reliability-diagram bins, overconfidence "
        "detection, prediction/outcome alignment, and mean confidence per "
        "verdict across every evaluated prediction in your scope. Only "
        "verified success/failure outcomes contribute — nothing is "
        "fabricated. Auth optional — scoped to the requesting user/namespace."
    ),
    responses={
        200: {
            "description": "Calibration summary across companies",
            "model": EvaluationCalibrationResponse,
        },
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_evaluation_calibration(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> EvaluationCalibrationResponse:
    del raw_request  # kept for symmetry with other routers
    return await CompanyEvaluationService().calibration(
        db, user_id=_user_id(current_user)
    )


@router.get(
    "/summary",
    response_model=EvaluationSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a validity-loop summary across your companies",
    description=(
        "Aggregate validity-loop summary over every company in your scope: "
        "outcome counts, evaluation counts, accuracy/precision/recall/F1, "
        "calibration, alignment, and a per-company breakdown. Auth optional "
        "— scoped to the requesting user/namespace."
    ),
    responses={
        200: {
            "description": "Global evaluation summary",
            "model": EvaluationSummaryResponse,
        },
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_evaluation_summary(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> EvaluationSummaryResponse:
    del raw_request  # kept for symmetry with other routers
    return await CompanyEvaluationService().summary(
        db, user_id=_user_id(current_user)
    )


@router.get(
    "/benchmark",
    response_model=BenchmarkAccuracyResponse,
    status_code=status.HTTP_200_OK,
    summary="Get the latest benchmark accuracy report",
    description=(
        "Read-only accuracy view over the latest recorded Ground Truth "
        "Evaluation Platform benchmark run: per-company engine replay results "
        "and, if the run stored a metric block, the headline accuracy metrics "
        "(confusion/F-beta, ROC-AUC, average precision, calibration). Nothing "
        "is recomputed or fabricated — numbers come from the append-only, "
        "integrity-verified benchmark history. Auth optional."
    ),
    responses={
        200: {
            "description": "Latest benchmark accuracy report",
            "model": BenchmarkAccuracyResponse,
        },
        404: {"description": "No benchmark run recorded yet"},
    },
)
async def get_benchmark_accuracy() -> BenchmarkAccuracyResponse:
    response = BenchmarkAccuracyService().latest()
    if response is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No benchmark run recorded yet; run `predictron-benchmark run` first.",
        )
    return response


__all__ = ["router"]
