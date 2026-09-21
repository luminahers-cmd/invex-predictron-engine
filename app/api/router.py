from fastapi import APIRouter

from app.api.analyze import router as analyze_router
from app.api.batch import router as batch_router
from app.api.companies import router as companies_router
from app.api.comparison import router as comparison_router
from app.api.due_diligence import router as due_diligence_router
from app.api.evaluation import router as evaluation_router
from app.api.forecasts import router as forecasts_router
from app.api.health import router as health_router
from app.api.monitoring import router as monitoring_router
from app.api.portfolio import router as portfolio_router
from app.api.search import router as search_router
from app.api.venture import router as venture_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health_router)
api_router.include_router(analyze_router)
api_router.include_router(companies_router)
api_router.include_router(evaluation_router)
api_router.include_router(forecasts_router)
api_router.include_router(monitoring_router)
api_router.include_router(venture_router)
api_router.include_router(portfolio_router)
api_router.include_router(comparison_router)
api_router.include_router(due_diligence_router)
api_router.include_router(search_router)
api_router.include_router(batch_router)
