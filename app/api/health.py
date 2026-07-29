import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import AsyncSessionLocal
from app.schemas.analysis import HealthResponse, ReadinessResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])
settings = get_settings()


async def _check_db_health() -> bool:
    """Return True if the database connection is alive."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.warning("Database health check failed")
        return False


def _check_engine_ready(app_request: Request) -> bool:
    """Return True if the PredictronEngine singleton is initialized."""
    return getattr(app_request.app.state, "predictron_engine", None) is not None


def _startup_state(app_request: Request) -> str:
    """Return the application startup state."""
    return getattr(app_request.app.state, "startup_state", "unknown")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Current health status including DB, engine, and startup state.",
    responses={
        200: {"description": "Application is healthy", "model": HealthResponse},
        503: {"description": "Application is not healthy"},
    },
)
async def health_check(request: Request) -> HealthResponse:
    engine_ready = _check_engine_ready(request)
    db_healthy = await _check_db_health()
    startup = _startup_state(request)
    overall_status = "ok" if (engine_ready and db_healthy) else "degraded"
    return HealthResponse(
        status=overall_status,
        version=settings.APP_VERSION,
        engine_reachable=engine_ready,
        db_healthy=db_healthy,
        startup_state=startup,
    )


@router.get(
    "/health/readiness",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Whether the app is ready for traffic. Checks DB, engine, and startup.",
    responses={
        200: {"description": "Application is ready"},
        503: {"description": "Application is not ready"},
    },
)
async def readiness_check(request: Request) -> JSONResponse:
    db_healthy = await _check_db_health()
    engine_ready = _check_engine_ready(request)
    startup_complete = getattr(request.app.state, "startup_state", None) == "ready"
    all_ready = db_healthy and engine_ready and startup_complete
    status_code = 200 if all_ready else 503
    body = ReadinessResponse(
        status="ready" if all_ready else "not_ready",
        db_healthy=db_healthy,
        engine_ready=engine_ready,
        startup_complete=startup_complete,
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())
