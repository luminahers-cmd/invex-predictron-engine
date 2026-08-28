from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class StartupAnalysisRequest(BaseModel):
    """Request payload for venture analysis."""

    model_config = ConfigDict(populate_by_name=True)

    startup_name: str = Field(
        ..., min_length=1, max_length=255, description="Name of the startup"
    )
    website: HttpUrl | None = Field(
        default=None,
        alias="website_url",
        description="Startup website URL (optional; frontend sends website_url)",
    )
    description: str = Field(
        ..., min_length=10, max_length=5000, description="Brief description of the startup"
    )
    pitch_deck_url: HttpUrl | None = Field(
        default=None, description="URL to the pitch deck"
    )
    founder_linkedin_urls: list[HttpUrl] = Field(
        default_factory=list,
        description="List of founder LinkedIn profile URLs",
        max_length=10,
    )


class AnalysisScores(BaseModel):
    """Individual score breakdown from analysis."""

    venture_score: float = Field(..., ge=0, le=100, description="Overall venture score")
    market_score: float = Field(..., ge=0, le=100, description="Market opportunity score")
    founder_score: float = Field(..., ge=0, le=100, description="Founder strength score")
    traction_score: float = Field(..., ge=0, le=100, description="Traction indicator score")


class StartupAnalysisResponse(BaseModel):
    """Response payload containing analysis results."""

    id: str | None = Field(
        default=None,
        description="Persisted analysis ID. Present when persistence succeeds.",
    )
    startup_name: str
    venture_score: float = Field(..., ge=0, le=100)
    market_score: float = Field(..., ge=0, le=100)
    founder_score: float = Field(..., ge=0, le=100)
    traction_score: float = Field(..., ge=0, le=100)
    recommendations: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1, description="Model confidence level")


class AnalysisSummaryResponse(BaseModel):
    """Summary of a persisted analysis for list views."""

    id: str
    startup_name: str
    venture_score: float = Field(..., ge=0, le=100)
    market_score: float = Field(..., ge=0, le=100)
    founder_score: float = Field(..., ge=0, le=100)
    traction_score: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    engine_version: str | None = None
    processing_time_ms: float | None = None
    created_at: datetime


class AnalysisDetailResponse(BaseModel):
    """Full detail of a persisted analysis."""

    id: str
    startup_name: str
    website: str
    description: str
    pitch_deck_url: str | None = None
    founder_linkedin_urls: list[str] = Field(default_factory=list)
    venture_score: float = Field(..., ge=0, le=100)
    market_score: float = Field(..., ge=0, le=100)
    founder_score: float = Field(..., ge=0, le=100)
    traction_score: float = Field(..., ge=0, le=100)
    recommendations: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)
    engine_version: str | None = None
    processing_time_ms: float | None = None
    full_report: dict[str, object] = Field(default_factory=dict)
    created_at: datetime


class AnalysisListResponse(BaseModel):
    """Paginated list of persisted analyses."""

    analyses: list[AnalysisSummaryResponse] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str
    engine_reachable: bool = False
    db_healthy: bool = False
    startup_state: str = "unknown"


class ReadinessResponse(BaseModel):
    """Readiness check response.

    Returns 200 only when all dependencies are healthy and the application
    has finished its startup sequence.
    """

    status: str = "ok"
    db_healthy: bool = False
    engine_ready: bool = False
    startup_complete: bool = False
