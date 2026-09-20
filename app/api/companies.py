"""Company Intelligence Hub API endpoints.

Read-only for Phase 1: companies are written exclusively by the ingest
hook that runs after successful analysis persistence. Only company
identity and analysis history are exposed — graph, signals, features, and
benchmarks belong to future phases.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt import get_current_user, get_current_user_optional
from app.db.session import get_db
from app.models.company_outcome import CompanyOutcome
from app.schemas.company import (
    CompanyDetailResponse,
    CompanyHistoryResponse,
    CompanyListResponse,
    CompanySnapshotResponse,
    CompanySummaryResponse,
)
from app.schemas.company_delta import CompanyDeltaListResponse
from app.schemas.company_history import (
    CompanyHistoryLatestResponse,
    CompanyTimelineEntry,
    CompanyTimelineResponse,
)
from app.schemas.company_profile import CompanyProfileResponse
from app.schemas.evaluation import CompanyPerformanceResponse
from app.schemas.outcome import (
    CompanyOutcomeListResponse,
    CompanyOutcomeResponse,
    CompanyOutcomeWriteResponse,
    OutcomeCreateRequest,
)
from app.services.company_deltas import CompanyDeltaService
from app.services.company_evaluation import CompanyEvaluationService
from app.services.company_history import (
    CompanyHistoryService,
    SnapshotView,
    default_dataset,
)
from app.services.company_outcomes import (
    CompanyOutcomeService,
    OutcomeSnapshotNotFoundError,
)
from app.services.company_postgres import PostgresCompanyStore
from app.services.company_protocols import CompanySnapshotRecord
from app.services.company_read import CompanyReadService
from predictron_engine.dataset.outcomes import StartupOutcome

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/companies", tags=["companies"])


def _user_id(current_user: dict[str, object] | None) -> str | None:
    sub = current_user.get("sub") if current_user else None
    return str(sub) if sub is not None else None


@router.get(
    "",
    response_model=CompanyListResponse,
    status_code=status.HTTP_200_OK,
    summary="List companies",
    description=(
        "List your persisted companies, newest first. Requires "
        "authentication — only companies your account created are returned."
    ),
    responses={
        200: {"description": "Paginated list of companies", "model": CompanyListResponse},
        401: {"description": "Not authenticated — missing or invalid token"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def list_companies(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] = Depends(get_current_user),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return"),
) -> CompanyListResponse:
    del raw_request  # kept for symmetry with other routers
    store = PostgresCompanyStore()
    page = await store.list_companies(
        db, user_id=_user_id(current_user), offset=offset, limit=limit
    )
    return CompanyListResponse(
        companies=[
            CompanySummaryResponse(
                company_id=c.company_id,
                canonical_name=c.canonical_name,
                canonical_domain=c.canonical_domain,
                primary_name=c.primary_name,
                website=c.website,
                latest_decision=c.latest_decision,
                latest_confidence=c.latest_confidence,
                latest_composite_score=c.latest_composite_score,
                snapshot_count=c.snapshot_count,
                first_seen=c.first_seen,
                last_seen=c.last_seen,
            )
            for c in page.companies
        ],
        total=page.total,
    )


@router.get(
    "/resolve",
    response_model=CompanyProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Resolve a company profile from identity signals",
    description=(
        "Build a unified intelligence profile from a company name and optional "
        "website/domain. Prefers the live registry when the resolved identity "
        "is registered; otherwise falls back to the grounded offline dataset so "
        "companies that predate the registry still surface a profile. Auth "
        "optional — the live lookup is scoped to the requesting user/namespace."
    ),
    responses={
        200: {
            "description": "Unified company intelligence profile",
            "model": CompanyProfileResponse,
        },
        404: {"description": "No company data found for the supplied signals"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def resolve_company_profile(
    name: str = Query(..., min_length=1, description="Company name"),
    website: str | None = Query(default=None, description="Company website URL"),
    domain: str | None = Query(default=None, description="Canonical company domain"),
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
    offset: int = Query(default=0, ge=0, description="Number of snapshots to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Max snapshots to summarize"),
) -> CompanyProfileResponse:
    service = CompanyReadService()
    profile = await service.resolve(
        db,
        name=name,
        website=website,
        domain=domain,
        user_id=_user_id(current_user),
        offset=offset,
        limit=limit,
    )
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No company data found for {name!r}",
        )
    return profile


@router.get(
    "/{company_id}",
    response_model=CompanyDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a company",
    description=(
        "Get company metadata, its latest analysis snapshot, and the total "
        "snapshot count. Auth optional — anonymous and authenticated users "
        "can only see companies in their own scope."
    ),
    responses={
        200: {"description": "Company details", "model": CompanyDetailResponse},
        404: {"description": "Company not found or access denied"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_company(
    company_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> CompanyDetailResponse:
    store = PostgresCompanyStore()
    uid = _user_id(current_user)

    company = await store.get_company(db, company_id, user_id=uid)
    if company is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company {company_id} not found",
        )

    page = await store.list_snapshots(db, company_id, user_id=uid, offset=0, limit=1)
    latest = page.snapshots[0] if page.snapshots else None

    return CompanyDetailResponse(
        company_id=company.company_id,
        canonical_name=company.canonical_name,
        canonical_domain=company.canonical_domain,
        primary_name=company.primary_name,
        website=company.website,
        latest_decision=company.latest_decision,
        latest_confidence=company.latest_confidence,
        latest_composite_score=company.latest_composite_score,
        snapshot_count=company.snapshot_count,
        first_seen=company.first_seen,
        last_seen=company.last_seen,
        latest_snapshot=_snapshot_response(latest) if latest is not None else None,
    )


@router.get(
    "/{company_id}/history",
    response_model=CompanyHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a company's analysis history",
    description=(
        "Get ordered (newest-first) analysis snapshots for a company. "
        "Auth optional — scoped to the requesting user/namespace."
    ),
    responses={
        200: {"description": "Ordered snapshot history", "model": CompanyHistoryResponse},
        404: {"description": "Company not found or access denied"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_company_history(
    company_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return"),
) -> CompanyHistoryResponse:
    store = PostgresCompanyStore()
    uid = _user_id(current_user)

    company = await store.get_company(db, company_id, user_id=uid)
    if company is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company {company_id} not found",
        )

    page = await store.list_snapshots(
        db, company_id, user_id=uid, offset=offset, limit=limit
    )
    return CompanyHistoryResponse(
        company_id=company_id,
        snapshots=[_snapshot_response(s) for s in page.snapshots],
        total=page.total,
    )


def _snapshot_response(snapshot: CompanySnapshotRecord) -> CompanySnapshotResponse:
    return CompanySnapshotResponse(
        id=snapshot.id,
        company_id=snapshot.company_id,
        analysis_id=snapshot.analysis_id,
        report_id=snapshot.report_id,
        decision=snapshot.decision,
        confidence=snapshot.confidence,
        composite_score=snapshot.composite_score,
        readiness_score=snapshot.readiness_score,
        dimension_scores=snapshot.dimension_scores,
        created_at=snapshot.created_at,
    )


@router.get(
    "/{company_id}/profile",
    response_model=CompanyProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a company's unified intelligence profile",
    description=(
        "Get the unified read-only profile for a company: live analysis "
        "history from the registry merged with grounded offline dataset "
        "intelligence (records, knowledge graph, signals, features). "
        "Coverage gating reports placeholder offline data explicitly — "
        "placeholders are never presented as live intelligence. Auth "
        "optional — scoped to the requesting user/namespace."
    ),
    responses={
        200: {
            "description": "Unified company intelligence profile",
            "model": CompanyProfileResponse,
        },
        404: {"description": "Company not found or access denied"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_company_profile(
    company_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
    offset: int = Query(default=0, ge=0, description="Number of snapshots to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Max snapshots to summarize"),
) -> CompanyProfileResponse:
    service = CompanyReadService()
    profile = await service.profile(
        db, company_id, user_id=_user_id(current_user), offset=offset, limit=limit
    )
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company {company_id} not found",
        )
    return profile


@router.get(
    "/{company_id}/history/latest",
    response_model=CompanyHistoryLatestResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a company's latest enriched snapshot",
    description=(
        "Get the most recent snapshot enriched with derived temporal "
        "intelligence (benchmark placement, decision explanation, key facts, "
        "evidence count) plus the deterministic trend summary over the full "
        "history. Auth optional — scoped to the requesting user/namespace."
    ),
    responses={
        200: {
            "description": "Latest enriched snapshot and trend",
            "model": CompanyHistoryLatestResponse,
        },
        404: {"description": "Company not found or access denied"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_company_history_latest(
    company_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> CompanyHistoryLatestResponse:
    service = CompanyHistoryService(dataset=default_dataset())
    result = await service.latest(db, company_id, user_id=_user_id(current_user))
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company {company_id} not found",
        )
    latest = result.entries[0] if result.entries else None
    return CompanyHistoryLatestResponse(
        company_id=company_id,
        total=result.total,
        entry=_timeline_entry(latest) if latest is not None else None,
        trend=result.trend,
    )


@router.get(
    "/{company_id}/timeline",
    response_model=CompanyTimelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a company's enriched temporal timeline",
    description=(
        "Get the enriched snapshot timeline (newest-first) with per-snapshot "
        "benchmark placement, decision explanation, key facts, and evidence "
        "count, plus the deterministic trend summary over the full history. "
        "Auth optional — scoped to the requesting user/namespace."
    ),
    responses={
        200: {"description": "Enriched snapshot timeline", "model": CompanyTimelineResponse},
        404: {"description": "Company not found or access denied"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_company_timeline(
    company_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return"),
) -> CompanyTimelineResponse:
    service = CompanyHistoryService(dataset=default_dataset())
    result = await service.timeline(
        db, company_id, user_id=_user_id(current_user), offset=offset, limit=limit
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company {company_id} not found",
        )
    return CompanyTimelineResponse(
        company_id=company_id,
        total=result.total,
        entries=[_timeline_entry(view) for view in result.entries],
        trend=result.trend,
    )


@router.get(
    "/{company_id}/deltas",
    response_model=CompanyDeltaListResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a company's snapshot deltas",
    description=(
        "Get structured comparisons between consecutive snapshots "
        "(newest-first), reporting per-field changes for decision, readiness, "
        "confidence, composite score, benchmark placement, decision "
        "explanation, key facts, and evidence count. Auth optional — scoped "
        "to the requesting user/namespace."
    ),
    responses={
        200: {"description": "Structured snapshot deltas", "model": CompanyDeltaListResponse},
        404: {"description": "Company not found or access denied"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_company_deltas(
    company_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return"),
) -> CompanyDeltaListResponse:
    service = CompanyDeltaService(
        history=CompanyHistoryService(dataset=default_dataset())
    )
    result = await service.deltas(
        db, company_id, user_id=_user_id(current_user), offset=offset, limit=limit
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company {company_id} not found",
        )
    return CompanyDeltaListResponse(
        company_id=company_id,
        total=result.total,
        deltas=result.deltas,
        trend=result.trend,
    )


@router.post(
    "/{company_id}/outcomes",
    response_model=CompanyOutcomeWriteResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record a company outcome",
    description=(
        "Record an observed real-world outcome for a company. Requires "
        "authentication — a company outside your scope is treated as not "
        "found. The outcome verdict is derived from the observed facts, never "
        "predicted. Submitting the same observation twice is an idempotent "
        "no-op. Recording also generates the newly applicable prediction "
        "evaluations."
    ),
    responses={
        201: {
            "description": "Outcome recorded (or already present)",
            "model": CompanyOutcomeWriteResponse,
        },
        404: {"description": "Company or snapshot not found or access denied"},
        422: {"description": "Invalid outcome payload"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def record_company_outcome(
    company_id: str,
    payload: OutcomeCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] = Depends(get_current_user),
) -> CompanyOutcomeWriteResponse:
    service = CompanyOutcomeService()
    try:
        result = await service.record(
            db, company_id, payload, user_id=_user_id(current_user)
        )
    except OutcomeSnapshotNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company {company_id} not found",
        )
    return CompanyOutcomeWriteResponse(
        outcome=_outcome_response(result.outcome),
        created=result.created,
        evaluations_created=result.evaluations_created,
    )


@router.get(
    "/{company_id}/outcomes",
    response_model=CompanyOutcomeListResponse,
    status_code=status.HTTP_200_OK,
    summary="List a company's outcomes",
    description=(
        "List a company's recorded real-world outcomes, newest-first. Auth "
        "optional — scoped to the requesting user/namespace."
    ),
    responses={
        200: {
            "description": "Paginated list of outcomes",
            "model": CompanyOutcomeListResponse,
        },
        404: {"description": "Company not found or access denied"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def list_company_outcomes(
    company_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return"),
) -> CompanyOutcomeListResponse:
    service = CompanyOutcomeService()
    page = await service.list(
        db, company_id, user_id=_user_id(current_user), offset=offset, limit=limit
    )
    if page is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company {company_id} not found",
        )
    return CompanyOutcomeListResponse(
        company_id=company_id,
        total=page.total,
        outcomes=[_outcome_response(row) for row in page.outcomes],
    )


@router.get(
    "/{company_id}/performance",
    response_model=CompanyPerformanceResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a company's prediction-evaluation performance",
    description=(
        "Get a company's validity-loop performance: recorded outcomes, "
        "evaluated predictions, aggregate accuracy/precision/recall/F1, "
        "calibration error, prediction/outcome alignment, and the "
        "append-only evaluation history. Auth optional — scoped to the "
        "requesting user/namespace."
    ),
    responses={
        200: {
            "description": "Company evaluation performance",
            "model": CompanyPerformanceResponse,
        },
        404: {"description": "Company not found or access denied"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_company_performance(
    company_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return"),
) -> CompanyPerformanceResponse:
    service = CompanyEvaluationService()
    result = await service.performance(
        db, company_id, user_id=_user_id(current_user), offset=offset, limit=limit
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company {company_id} not found",
        )
    return result


def _outcome_response(row: CompanyOutcome) -> CompanyOutcomeResponse:
    return CompanyOutcomeResponse(
        id=row.id,
        company_id=row.company_id,
        snapshot_id=row.snapshot_id,
        source=row.source,
        occurred_at=row.occurred_at,
        time_horizon_days=row.time_horizon_days,
        outcome=StartupOutcome.model_validate(row.outcome_data),
        verdict=row.verdict,
        verdict_reasoning=row.verdict_reasoning,
        status=row.status,
        notes=row.notes,
        created_at=row.created_at,
    )


def _timeline_entry(view: SnapshotView) -> CompanyTimelineEntry:
    snapshot = view.snapshot
    return CompanyTimelineEntry(
        snapshot_id=snapshot.id,
        company_id=snapshot.company_id,
        analysis_id=snapshot.analysis_id,
        report_id=snapshot.report_id,
        created_at=snapshot.created_at,
        decision=snapshot.decision,
        confidence=snapshot.confidence,
        readiness_score=snapshot.readiness_score,
        composite_score=snapshot.composite_score,
        dimension_scores=snapshot.dimension_scores,
        benchmark=view.benchmark,
        decision_explanation=list(view.decision_explanation),
        key_facts=list(view.key_facts),
        evidence_count=view.evidence_count,
    )
