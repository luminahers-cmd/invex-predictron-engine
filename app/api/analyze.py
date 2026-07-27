import logging

from fastapi import APIRouter, Request, status

from app.schemas.analysis import StartupAnalysisRequest, StartupAnalysisResponse
from app.services.analysis import run_analysis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analysis"])


@router.post(
    "",
    response_model=StartupAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze a startup venture",
    description="Submit startup info and receive a venture analysis.",
)
async def analyze_startup(
    request: StartupAnalysisRequest,
    raw_request: Request,
) -> StartupAnalysisResponse:
    """Delegate analysis to the PredictronEngine singleton via the service layer."""
    engine = raw_request.app.state.predictron_engine
    return await run_analysis(engine, request)
