"""Continuous Intelligence & Drift Detection API (CIH Phase 6).

Read-only monitoring views over the live prediction ledger (summary, health,
stale, overdue, re-analysis) plus time-series trends and drift over the
recorded snapshot history.  Auth optional — live views are scoped to the
requesting user/namespace; history views are repository-wide artifacts
recorded by the ``predictron-monitor snapshot`` CLI.

Static routes are declared before any path-parameter routes so nothing is
shadowed.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt import get_current_user_optional
from app.db.session import get_db
from app.schemas.monitoring import (
    MonitorDriftResponse,
    MonitorHealthListResponse,
    MonitorReanalysisResponse,
    MonitorSummaryResponse,
    MonitorTrendsResponse,
)
from app.services.monitoring import MonitoringService
from predictron_engine.monitoring.models import MonitorPeriodKind

router = APIRouter(prefix="/monitor", tags=["monitor"])


def _user_id(current_user: dict[str, object] | None) -> str | None:
    sub = current_user.get("sub") if current_user else None
    return str(sub) if sub is not None else None


def _normalize_period(value: str) -> MonitorPeriodKind:
    try:
        return MonitorPeriodKind(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"period must be one of {[p.value for p in MonitorPeriodKind]}",
        ) from exc


@router.get(
    "/summary",
    response_model=MonitorSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a live monitoring summary for your companies",
    description=(
        "Live repository-wide monitoring summary over your forecasts and "
        "evaluations: counts, canonical aggregate metrics (accuracy/"
        "precision/recall/F1/coverage), calibration (ECE/MCE), confidence "
        "mean, verdict/outcome/decision distributions, and horizon / sector "
        "breakdowns. Auth optional — scoped to the requesting user/namespace."
    ),
    responses={
        200: {"description": "Live monitoring summary", "model": MonitorSummaryResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_monitor_summary(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> MonitorSummaryResponse:
    del raw_request  # kept for symmetry with other routers
    return await MonitoringService().summary(db, user_id=_user_id(current_user))


@router.get(
    "/rollup",
    response_model=MonitorSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a rolling-window monitoring rollup",
    description=(
        "Same live summary as ``/monitor/summary`` but with rolling-window "
        "aggregates over the most recent evaluations (default window 30). "
        "Auth optional — scoped to the requesting user/namespace."
    ),
    responses={
        200: {"description": "Rolling-window monitoring rollup", "model": MonitorSummaryResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_monitor_rollup(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> MonitorSummaryResponse:
    del raw_request  # kept for symmetry with other routers
    return await MonitoringService().rolling(db, user_id=_user_id(current_user))


@router.get(
    "/health",
    response_model=MonitorHealthListResponse,
    status_code=status.HTTP_200_OK,
    summary="Get derived forecast-health rows",
    description=(
        "Per-forecast health projection (active / due / overdue / stale / "
        "resolved) derived deterministically from the stored lifecycle. "
        "Auth optional — scoped to the requesting user/namespace."
    ),
    responses={
        200: {"description": "Forecast-health rows", "model": MonitorHealthListResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_monitor_health(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> MonitorHealthListResponse:
    del raw_request
    return await MonitoringService().health(db, user_id=_user_id(current_user))


@router.get(
    "/stale",
    response_model=MonitorHealthListResponse,
    status_code=status.HTTP_200_OK,
    summary="List stale forecasts in your scope",
    description=(
        "Forecasts whose frozen snapshot is older than 365 days (the named "
        "staleness threshold) and still unresolved.  ``distribution`` covers "
        "the full population. Auth optional — scoped to your namespace."
    ),
    responses={
        200: {"description": "Stale forecasts", "model": MonitorHealthListResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_monitor_stale(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> MonitorHealthListResponse:
    del raw_request
    return await MonitoringService().health(
        db, user_id=_user_id(current_user), health_filter="stale"
    )


@router.get(
    "/overdue",
    response_model=MonitorHealthListResponse,
    status_code=status.HTTP_200_OK,
    summary="List overdue forecasts in your scope",
    description=(
        "Forecasts whose due time has passed with no outcome attached. "
        "``distribution`` covers the full population. Auth optional — scoped "
        "to your namespace."
    ),
    responses={
        200: {"description": "Overdue forecasts", "model": MonitorHealthListResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_monitor_overdue(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> MonitorHealthListResponse:
    del raw_request
    return await MonitoringService().health(
        db, user_id=_user_id(current_user), health_filter="overdue"
    )


@router.get(
    "/reanalysis",
    response_model=MonitorReanalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="List re-analysis recommendations",
    description=(
        "Deterministic re-analysis recommendations (new outcome recorded, "
        "evidence changed, prediction expired, snapshot age exceeded) for "
        "the frozen forecasts in your scope. Auth optional — scoped to your "
        "namespace."
    ),
    responses={
        200: {"description": "Re-analysis recommendations", "model": MonitorReanalysisResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_monitor_reanalysis(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> MonitorReanalysisResponse:
    del raw_request
    return await MonitoringService().reanalysis(db, user_id=_user_id(current_user))


@router.get(
    "/trends",
    response_model=MonitorTrendsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get dashboard time-series over recorded snapshots",
    description=(
        "Anchor-ordered time-series (accuracy, precision, recall, F1, "
        "coverage, ECE, mean confidence) computed from the recorded "
        "monitor-snapshot history. Record snapshots with the "
        "``predictron-monitor snapshot`` CLI. Repository-wide, read-only."
    ),
    responses={
        200: {"description": "Metric time-series", "model": MonitorTrendsResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_monitor_trends(
    period: str = "daily",
) -> MonitorTrendsResponse:
    period_kind = _normalize_period(period)
    return MonitoringService().trends(period_kind=period_kind)


@router.get(
    "/drift",
    response_model=MonitorDriftResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare drift between two recorded snapshots",
    description=(
        "Deterministic five-dimension drift report (calibration, confidence, "
        "prediction distribution, outcome distribution, horizon performance) "
        "between two recorded snapshots. Without ids, the latest two daily "
        "snapshots are compared. Repository-wide, read-only."
    ),
    responses={
        200: {"description": "Drift report", "model": MonitorDriftResponse},
        404: {"description": "Fewer than two snapshots recorded"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_monitor_drift(
    before_id: str | None = None,
    after_id: str | None = None,
    period: str = "daily",
) -> MonitorDriftResponse:
    period_kind = _normalize_period(period)
    try:
        return MonitoringService().drift(
            before_id=before_id,
            after_id=after_id,
            period_kind=period_kind,
        )
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


__all__ = ["router"]
