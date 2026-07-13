from fastapi import APIRouter, HTTPException, status

from app.adapters.predictron_adapter import PredictronError, analyze
from app.schemas.analysis import StartupAnalysisRequest, StartupAnalysisResponse

router = APIRouter(prefix="/analyze", tags=["analysis"])


@router.post(
    "",
    response_model=StartupAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze a startup venture",
    description="Submit startup info and receive a venture analysis.",
)
async def analyze_startup(request: StartupAnalysisRequest) -> StartupAnalysisResponse:
    """Validate the incoming request and delegate to the Predictron Engine."""
    try:
        result = await analyze(request)
    except PredictronError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Prediction engine error: {exc}",
        )
    return result
