"""Portfolio Analysis API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class PortfolioAnalysisRequest(BaseModel):
    """Request to analyze a portfolio of startups."""

    company_names: list[str] = Field(
        ...,
        min_length=2,
        max_length=50,
        description="List of startup names to analyze as a portfolio",
    )
    descriptions: dict[str, str] = Field(
        default_factory=dict,
        description="Optional descriptions keyed by company name",
    )
    website_urls: dict[str, str | None] = Field(
        default_factory=dict,
        description="Optional website URLs keyed by company name",
    )


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class PortfolioCompanyScore(BaseModel):
    """Score summary for a single company in the portfolio."""

    startup_name: str
    overall_score: float = Field(..., ge=0.0, le=100.0)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    decision_category: str | None = None
    conviction_level: str | None = None


class SectorDistribution(BaseModel):
    """Sector breakdown within the portfolio."""

    sector: str
    count: int
    percentage: float = Field(..., ge=0.0, le=100.0)
    avg_score: float = Field(default=0.0, ge=0.0, le=100.0)


class StageDistribution(BaseModel):
    """Stage breakdown within the portfolio."""

    stage: str
    count: int
    percentage: float = Field(..., ge=0.0, le=100.0)


class RiskSummary(BaseModel):
    """Aggregated risk information for the portfolio."""

    high_risk_count: int = 0
    medium_risk_count: int = 0
    low_risk_count: int = 0
    average_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_factors: list[str] = Field(default_factory=list)


class ConcentrationRisk(BaseModel):
    """Concentration risk analysis."""

    sector_concentration: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="HHI-like concentration: 1.0 = fully concentrated",
    )
    stage_concentration: float = Field(default=0.0, ge=0.0, le=1.0)
    score_variance: float = Field(default=0.0, ge=0.0)
    most_concentrated_sector: str | None = None
    recommendations: list[str] = Field(default_factory=list)


class DiversificationScore(BaseModel):
    """Portfolio diversification metrics."""

    sector_diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    stage_diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    geography_diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_diversification: float = Field(default=0.0, ge=0.0, le=1.0)
    recommendation_count: int = 0


class HeatmapCell(BaseModel):
    """Single cell in the portfolio heatmap."""

    row_label: str
    col_label: str
    value: float
    label: str = ""


class PortfolioAnalysisResponse(BaseModel):
    """Complete portfolio analysis response."""

    company_count: int = 0
    companies: list[PortfolioCompanyScore] = Field(default_factory=list)
    portfolio_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Weighted average portfolio score"
    )
    sector_distribution: list[SectorDistribution] = Field(default_factory=list)
    stage_distribution: list[StageDistribution] = Field(default_factory=list)
    risk_summary: RiskSummary = Field(default_factory=RiskSummary)
    diversification: DiversificationScore = Field(
        default_factory=DiversificationScore
    )
    concentration_risk: ConcentrationRisk = Field(
        default_factory=ConcentrationRisk
    )
    heatmap_data: list[HeatmapCell] = Field(default_factory=list)
    similarity_matrix: list[dict[str, object]] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time_ms: float | None = None


class PortfolioComparisonItem(BaseModel):
    """Single company in a comparison matrix."""

    startup_name: str
    overall_score: float = Field(default=0.0)
    confidence: float = Field(default=0.0)
    decision: str = ""
    dimension_scores: dict[str, float] = Field(default_factory=dict)


class SimilarityMatrixResponse(BaseModel):
    """Pairwise similarity matrix for portfolio companies."""

    companies: list[str] = Field(default_factory=list)
    matrix: list[list[float]] = Field(default_factory=list)
    method: str = "score_cosine"
