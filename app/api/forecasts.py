"""Live Prediction Ledger API endpoints (CIH Phase 5).

Deterministic forecast lifecycle: registration (idempotent), listing,
detail, reconciliation (per forecast and batch), and a cross-scope summary.
Auth optional — scoped to the requesting user/namespace.

Static routes (``/forecasts/summary`` and ``/forecasts/reconcile``) are
declared before ``/forecasts/{forecast_id}`` so the path parameters never
shadow them.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt import get_current_user_optional
from app.db.session import get_db
from app.models.forecast import Forecast, ForecastEvent
from app.schemas.forecast import (
    ForecastBatchReconcileResponse,
    ForecastCreateRequest,
    ForecastDetailResponse,
    ForecastEventResponse,
    ForecastListResponse,
    ForecastReconcileResponse,
    ForecastResponse,
    ForecastSummaryResponse,
    ForecastWriteResponse,
)
from app.services.forecasts import ForecastService, ForecastSnapshotNotFoundError

router = APIRouter(prefix="/forecasts", tags=["forecasts"])


def _user_id(current_user: dict[str, object] | None) -> str | None:
    sub = current_user.get("sub") if current_user else None
    return str(sub) if sub is not None else None


@router.post(
    "",
    response_model=ForecastWriteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a forecast for a frozen snapshot",
    description=(
        "Register one deterministic forecast for a company snapshot. The "
        "forecast ID is derived from (company, snapshot, horizon), so "
        "re-submitting the same registration is an idempotent no-op "
        "(``created=false``). The frozen prediction, engine version, and "
        "schema version are persisted once and never changed afterwards. "
        "Auth optional — scoped to the requesting user/namespace."
    ),
    responses={
        201: {
            "description": "Forecast registered (or already present)",
            "model": ForecastWriteResponse,
        },
        404: {"description": "Company or snapshot not found or access denied"},
        422: {"description": "Invalid forecast payload"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def register_forecast(
    payload: ForecastCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> ForecastWriteResponse:
    service = ForecastService()
    try:
        result = await service.register(
            db,
            company_id=payload.company_id,
            snapshot_id=payload.snapshot_id,
            horizon_days=payload.horizon_days,
            notes=payload.notes,
            user_id=_user_id(current_user),
        )
    except ForecastSnapshotNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company {payload.company_id} not found",
        )
    return ForecastWriteResponse(
        forecast=forecast_response(result.forecast),
        created=result.created,
    )


@router.get(
    "",
    response_model=ForecastListResponse,
    status_code=status.HTTP_200_OK,
    summary="List forecasts",
    description=(
        "List forecasts newest-first, scoped to the requesting "
        "user/namespace, optionally filtered by company and lifecycle "
        "status. Auth optional."
    ),
    responses={
        200: {
            "description": "Paginated list of forecasts",
            "model": ForecastListResponse,
        },
        429: {"description": "Rate limit exceeded"},
    },
)
async def list_forecasts(
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
    company_id: str | None = Query(
        default=None, description="Filter by owning company"
    ),
    forecast_status: str | None = Query(
        default=None, alias="status", description="Filter by lifecycle status"
    ),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return"),
) -> ForecastListResponse:
    page = await ForecastService().list_forecasts(
        db,
        user_id=_user_id(current_user),
        company_id=company_id,
        status_filter=forecast_status,
        offset=offset,
        limit=limit,
    )
    return ForecastListResponse(
        total=page.total,
        forecasts=[forecast_response(row) for row in page.forecasts],
        offset=offset,
        limit=limit,
    )


@router.get(
    "/summary",
    response_model=ForecastSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a live prediction ledger summary",
    description=(
        "Ledger counts for the requesting user/namespace: total forecasts, "
        "per-status distribution, outcome-linked count, and the most recent "
        "forecasts. Auth optional."
    ),
    responses={
        200: {
            "description": "Forecast ledger summary",
            "model": ForecastSummaryResponse,
        },
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_forecast_summary(
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> ForecastSummaryResponse:
    total, by_status, outcome_linked, recent = await ForecastService().summary(
        db, user_id=_user_id(current_user)
    )
    return ForecastSummaryResponse(
        total=total,
        by_status=by_status,
        active=by_status.get("active", 0),
        due=by_status.get("due", 0),
        resolved=by_status.get("resolved", 0),
        outcome_linked=outcome_linked,
        recent=[forecast_response(row) for row in recent],
        generated_at=datetime.now(UTC),
    )


@router.post(
    "/reconcile",
    response_model=ForecastBatchReconcileResponse,
    status_code=status.HTTP_200_OK,
    summary="Reconcile every forecast in scope against recorded outcomes",
    description=(
        "Match every forecast in the requesting user/namespace against its "
        "earliest eligible recorded outcome, derive and persist the lifecycle "
        "status, and replay the existing prediction-evaluation machinery. "
        "Idempotent — a second pass changes nothing. Auth optional."
    ),
    responses={
        200: {
            "description": "Batch reconciliation result",
            "model": ForecastBatchReconcileResponse,
        },
        429: {"description": "Rate limit exceeded"},
    },
)
async def reconcile_all_forecasts(
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> ForecastBatchReconcileResponse:
    forecasts, reconciled_count = await ForecastService().reconcile_all(
        db, user_id=_user_id(current_user)
    )
    return ForecastBatchReconcileResponse(
        forecasts=[forecast_response(row) for row in forecasts],
        reconciled_count=reconciled_count,
    )


@router.get(
    "/{forecast_id}",
    response_model=ForecastDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a forecast with its frozen prediction and event log",
    description=(
        "Fetch one forecast with its frozen prediction payload and "
        "append-only lifecycle event log. Auth optional — a forecast outside "
        "your scope is treated as not found."
    ),
    responses={
        200: {
            "description": "Forecast detail",
            "model": ForecastDetailResponse,
        },
        404: {"description": "Forecast not found or access denied"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_forecast_detail(
    forecast_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> ForecastDetailResponse:
    forecast = await ForecastService().get(
        db, forecast_id, user_id=_user_id(current_user)
    )
    if forecast is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Forecast {forecast_id} not found",
        )
    events = await ForecastService().list_events(db, forecast_id)
    return forecast_detail_response(forecast, events)


@router.post(
    "/{forecast_id}/reconcile",
    response_model=ForecastReconcileResponse,
    status_code=status.HTTP_200_OK,
    summary="Reconcile one forecast against recorded outcomes",
    description=(
        "Match one forecast against its earliest eligible recorded outcome, "
        "derive and persist the lifecycle status, and replay the existing "
        "prediction-evaluation machinery. Idempotent — re-attaching the same "
        "outcome changes nothing. Auth optional."
    ),
    responses={
        200: {
            "description": "Reconciliation result",
            "model": ForecastReconcileResponse,
        },
        404: {"description": "Forecast not found or access denied"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def reconcile_forecast(
    forecast_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> ForecastReconcileResponse:
    result = await ForecastService().reconcile(
        db, forecast_id, user_id=_user_id(current_user)
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Forecast {forecast_id} not found",
        )
    return ForecastReconcileResponse(
        forecast=forecast_response(result.forecast),
        outcome_id=result.outcome_id,
        outcome_verdict=result.outcome_verdict,
        evaluations_created=result.evaluations_created,
        reconciled=result.reconciled,
    )


def forecast_response(row: Forecast) -> ForecastResponse:
    return ForecastResponse(
        id=row.id,
        company_id=row.company_id,
        snapshot_id=row.snapshot_id,
        outcome_id=row.outcome_id,
        analysis_timestamp=row.analysis_timestamp,
        due_at=row.due_at,
        horizon_days=row.horizon_days,
        status=row.status,
        decision=row.decision,
        confidence=row.confidence,
        composite_score=row.composite_score,
        engine_version=row.engine_version,
        schema_version=row.schema_version,
        reconciled_at=row.reconciled_at,
        notes=row.notes,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def forecast_event_response(event: ForecastEvent) -> ForecastEventResponse:
    return ForecastEventResponse(
        id=event.id,
        forecast_id=event.forecast_id,
        event_type=event.event_type,
        occurred_at=event.occurred_at,
        payload=event.payload,
    )


def forecast_detail_response(
    row: Forecast, events: list[ForecastEvent]
) -> ForecastDetailResponse:
    base = forecast_response(row)
    return ForecastDetailResponse(
        **base.model_dump(),
        prediction=row.prediction,
        events=[forecast_event_response(event) for event in events],
    )


__all__ = ["router"]
