import logging

from fastapi import APIRouter, Request
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import AsyncSessionLocal
from app.schemas.analysis import HealthResponse

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


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check(request: Request) -> HealthResponse:
    engine_ready = _check_engine_ready(request)
    db_healthy = await _check_db_health()
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        engine_reachable=engine_ready,
        db_healthy=db_healthy,
    )
