from pydantic import BaseModel, Field, HttpUrl


class StartupAnalysisRequest(BaseModel):
    """Request payload for venture analysis."""

    startup_name: str = Field(
        ..., min_length=1, max_length=255, description="Name of the startup"
    )
    website: HttpUrl = Field(..., description="Startup website URL")
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

    startup_name: str
    venture_score: float = Field(..., ge=0, le=100)
    market_score: float = Field(..., ge=0, le=100)
    founder_score: float = Field(..., ge=0, le=100)
    traction_score: float = Field(..., ge=0, le=100)
    recommendations: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1, description="Model confidence level")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str
    engine_reachable: bool = False
