"""Continuous Learning Intelligence API (CIH Phase 7).

Read-only learning views over the live evaluated prediction ledger —
summary, patterns, observations, confidence, bias, recommendations — plus
recorded learning-report history.  Live views are auth-optional and scoped
to the requesting user/namespace; reports are repository-wide artifacts
recorded by the ``predictron-learning snapshot`` CLI.

Static routes are declared before any path-parameter routes so nothing is
shadowed.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt import get_current_user_optional
from app.db.session import get_db
from app.schemas.learning import (
    LearningBiasResponse,
    LearningConfidenceResponse,
    LearningKnowledgeResponse,
    LearningObservationResponse,
    LearningPatternResponse,
    LearningRecommendationsResponse,
    LearningReportDetailResponse,
    LearningReportsResponse,
    LearningSummaryResponse,
)
from app.services.learning import LearningService
from predictron_engine.learning.models import LearningPeriodKind

router = APIRouter(prefix="/learning", tags=["learning"])


def _user_id(current_user: dict[str, object] | None) -> str | None:
    sub = current_user.get("sub") if current_user else None
    return str(sub) if sub is not None else None


def _normalize_period(value: str) -> LearningPeriodKind:
    try:
        return LearningPeriodKind(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"period must be one of {[p.value for p in LearningPeriodKind]}",
        ) from exc


def _normalize_dimension(value: str) -> str:
    from predictron_engine.learning.models import LearningDimension

    try:
        return LearningDimension(value).value
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"dimension must be one of "
                f"{sorted(d.value for d in LearningDimension)}"
            ),
        ) from exc


@router.get(
    "/summary",
    response_model=LearningSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a live learning summary over your evaluated population",
    description=(
        "Live learning summary over your evaluated prediction ledger: "
        "counts, canonical aggregate metrics (accuracy/precision/recall/"
        "false-positive/false-negative rate), calibration digest (ECE/over-"
        "confidence), confidence statistics, per-dimension knowledge, "
        "patterns, observations, and rule-based recommendations. Auth "
        "optional — scoped to the requesting user/namespace."
    ),
    responses={
        200: {"description": "Live learning summary", "model": LearningSummaryResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_learning_summary(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> LearningSummaryResponse:
    del raw_request
    return await LearningService().summary(db, user_id=_user_id(current_user))


@router.get(
    "/patterns",
    response_model=LearningPatternResponse,
    status_code=status.HTTP_200_OK,
    summary="Get per-dimension cohort patterns over your evaluated population",
    description=(
        "Per-dimension cohort patterns (sector/stage/country/technology/"
        "business-model/founder) with accuracy, confusion counts, confidence "
        "bias, and deterministic rule-engine guidance per bucket. Auth "
        "optional — scoped to the requesting user/namespace."
    ),
    responses={
        200: {"description": "Cohort patterns", "model": LearningPatternResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_learning_patterns(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> LearningPatternResponse:
    del raw_request
    return await LearningService().patterns(db, user_id=_user_id(current_user))


@router.get(
    "/observations",
    response_model=LearningObservationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get canonical observations about your evaluated population",
    description=(
        "Canonical, deterministic observations (accuracy excess/deficit, "
        "over/under-confidence, low support, strong accuracy, high "
        "false-positive/negative rate) emitted from the live evaluated "
        "population. Auth optional — scoped to the requesting user/namespace."
    ),
    responses={
        200: {"description": "Canonical observations", "model": LearningObservationResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_learning_observations(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> LearningObservationResponse:
    del raw_request
    return await LearningService().observations(db, user_id=_user_id(current_user))


@router.get(
    "/confidence",
    response_model=LearningConfidenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Get confidence statistics and calibration digest",
    description=(
        "Population confidence statistics (count, mean, standard deviation, "
        "bias, calibrated flag) and the equal-width calibration digest for "
        "the live evaluated population. Auth optional — scoped to your "
        "namespace."
    ),
    responses={
        200: {"description": "Confidence statistics", "model": LearningConfidenceResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_learning_confidence(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> LearningConfidenceResponse:
    del raw_request
    return await LearningService().confidence(db, user_id=_user_id(current_user))


@router.get(
    "/bias",
    response_model=LearningBiasResponse,
    status_code=status.HTTP_200_OK,
    summary="Get confidence-bias view plus population metric surface",
    description=(
        "Confidence-bias view across the attribute dimensions plus the full "
        "population metric surface for the live evaluated ledger. Auth "
        "optional — scoped to the requesting user/namespace."
    ),
    responses={
        200: {"description": "Confidence-bias view", "model": LearningBiasResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_learning_bias(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> LearningBiasResponse:
    del raw_request
    return await LearningService().bias(db, user_id=_user_id(current_user))


@router.get(
    "/recommendations",
    response_model=LearningRecommendationsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get deterministic rule-based recommendations",
    description=(
        "Canned, deterministic recommendations (confidence recalibration, "
        "optimism reduction, weighting maintenance, concentration watch) "
        "derived from the live evaluated population — never LLM-generated "
        "and never fabricated. Auth optional — scoped to your namespace."
    ),
    responses={
        200: {
            "description": "Rule-based recommendations",
            "model": LearningRecommendationsResponse,
        },
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_learning_recommendations(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> LearningRecommendationsResponse:
    del raw_request
    return await LearningService().recommendations(db, user_id=_user_id(current_user))


@router.get(
    "/knowledge/{dimension}",
    response_model=LearningKnowledgeResponse,
    status_code=status.HTTP_200_OK,
    summary="Get per-dimension knowledge for one learning dimension",
    description=(
        "Per-dimension knowledge view (sector/stage/country/technology/"
        "business-model/founder): value buckets with accuracy, confidence "
        "bias, false-positive/negative rates, and deterministic guidance. "
        "Auth optional — scoped to the requesting user/namespace."
    ),
    responses={
        200: {"description": "Per-dimension knowledge", "model": LearningKnowledgeResponse},
        422: {"description": "Unknown learning dimension"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_learning_knowledge(
    dimension: str,
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> LearningKnowledgeResponse:
    del raw_request
    return await LearningService().dimension_intelligence(
        db, _normalize_dimension(dimension), user_id=_user_id(current_user)
    )


@router.get(
    "/sector",
    response_model=LearningKnowledgeResponse,
    status_code=status.HTTP_200_OK,
    summary="Get sector knowledge",
    description="Shortcut for ``/learning/knowledge/sector``.",
    responses={
        200: {"description": "Sector knowledge", "model": LearningKnowledgeResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_learning_sector(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> LearningKnowledgeResponse:
    del raw_request
    return await LearningService().dimension_intelligence(
        db, "sector", user_id=_user_id(current_user)
    )


@router.get(
    "/technology",
    response_model=LearningKnowledgeResponse,
    status_code=status.HTTP_200_OK,
    summary="Get technology knowledge",
    description="Shortcut for ``/learning/knowledge/technology``.",
    responses={
        200: {"description": "Technology knowledge", "model": LearningKnowledgeResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_learning_technology(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> LearningKnowledgeResponse:
    del raw_request
    return await LearningService().dimension_intelligence(
        db, "technology", user_id=_user_id(current_user)
    )


@router.get(
    "/country",
    response_model=LearningKnowledgeResponse,
    status_code=status.HTTP_200_OK,
    summary="Get country knowledge",
    description="Shortcut for ``/learning/knowledge/country``.",
    responses={
        200: {"description": "Country knowledge", "model": LearningKnowledgeResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_learning_country(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> LearningKnowledgeResponse:
    del raw_request
    return await LearningService().dimension_intelligence(
        db, "country", user_id=_user_id(current_user)
    )


@router.get(
    "/stage",
    response_model=LearningKnowledgeResponse,
    status_code=status.HTTP_200_OK,
    summary="Get stage knowledge",
    description="Shortcut for ``/learning/knowledge/stage``.",
    responses={
        200: {"description": "Stage knowledge", "model": LearningKnowledgeResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_learning_stage(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> LearningKnowledgeResponse:
    del raw_request
    return await LearningService().dimension_intelligence(
        db, "stage", user_id=_user_id(current_user)
    )


@router.get(
    "/founders",
    response_model=LearningKnowledgeResponse,
    status_code=status.HTTP_200_OK,
    summary="Get founder knowledge",
    description="Shortcut for ``/learning/knowledge/founder``.",
    responses={
        200: {"description": "Founder knowledge", "model": LearningKnowledgeResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_learning_founders(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> LearningKnowledgeResponse:
    del raw_request
    return await LearningService().dimension_intelligence(
        db, "founder", user_id=_user_id(current_user)
    )


@router.get(
    "/business-model",
    response_model=LearningKnowledgeResponse,
    status_code=status.HTTP_200_OK,
    summary="Get business-model knowledge",
    description="Shortcut for ``/learning/knowledge/business_model``.",
    responses={
        200: {"description": "Business-model knowledge", "model": LearningKnowledgeResponse},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_learning_business_model(
    raw_request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: dict[str, object] | None = Depends(get_current_user_optional),
) -> LearningKnowledgeResponse:
    del raw_request
    return await LearningService().dimension_intelligence(
        db, "business_model", user_id=_user_id(current_user)
    )


@router.get(
    "/reports",
    response_model=LearningReportsResponse,
    status_code=status.HTTP_200_OK,
    summary="List recorded learning snapshots (headers only)",
    description=(
        "Header summaries of recorded learning snapshots (latest first). "
        "Record snapshots with the ``predictron-learning snapshot`` CLI. "
        "Repository-wide, read-only."
    ),
    responses={
        200: {
            "description": "Recorded learning snapshot headers",
            "model": LearningReportsResponse,
        },
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_learning_reports(
    period: str = "daily",
) -> LearningReportsResponse:
    period_kind = _normalize_period(period)
    from app.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        return await LearningService().reports(session, period_kind=period_kind)


@router.get(
    "/reports/{snapshot_id}",
    response_model=LearningReportDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a full recorded learning report",
    description=(
        "Full recorded learning-report payload plus its integrity hash for "
        "one snapshot id. ``verify`` returns ``True`` when the stored hash "
        "matches the recomputed payload. Repository-wide, read-only."
    ),
    responses={
        200: {
            "description": "Full recorded learning report",
            "model": LearningReportDetailResponse,
        },
        404: {"description": "No report recorded for this snapshot id"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def get_learning_report_detail(
    snapshot_id: str,
) -> LearningReportDetailResponse:
    from app.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        report = await LearningService().report_detail(session, snapshot_id=snapshot_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no learning report recorded for snapshot id {snapshot_id!r}",
        )
    return report


__all__ = ["router"]
