from fastapi import APIRouter

from app.adapters.predictron_adapter import check_engine_health
from app.core.config import get_settings
from app.schemas.analysis import HealthResponse

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    engine_reachable = await check_engine_health()
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        engine_reachable=engine_reachable,
    )
