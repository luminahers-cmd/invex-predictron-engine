"""Due Diligence Report API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class DueDiligenceRequest(BaseModel):
    """Request to generate a due diligence report."""

    startup_name: str = Field(
        ..., min_length=1, max_length=255, description="Startup name"
    )
    website_url: str | None = Field(
        default=None, description="Startup website URL"
    )
    description: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Brief description of the startup",
    )
    pitch_deck_url: str | None = Field(
        default=None, description="URL to the pitch deck"
    )
    founder_linkedin_urls: list[str] = Field(
        default_factory=list, max_length=10
    )


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class ExecutiveSummary(BaseModel):
    """Executive summary section of the due diligence report."""

    headline: str = ""
    overview: str = ""
    key_findings: list[str] = Field(default_factory=list)
    confidence_level: float = Field(default=0.0, ge=0.0, le=1.0)


class StrengthItem(BaseModel):
    """Single strength entry."""

    title: str
    description: str = ""
    dimension: str = ""
    severity: str = "moderate"
    evidence: list[str] = Field(default_factory=list)


class WeaknessItem(BaseModel):
    """Single weakness entry."""

    title: str
    description: str = ""
    dimension: str = ""
    severity: str = "moderate"
    evidence: list[str] = Field(default_factory=list)


class OpportunityItemResponse(BaseModel):
    """Single opportunity entry."""

    title: str
    description: str = ""
    dimension: str = ""
    impact: str = "moderate"
    evidence: list[str] = Field(default_factory=list)


class RiskItemResponse(BaseModel):
    """Single risk entry."""

    title: str
    description: str = ""
    dimension: str = ""
    severity: str = "low"
    mitigation: str = ""
    evidence: list[str] = Field(default_factory=list)


class EvidenceEntry(BaseModel):
    """Evidence supporting a finding."""

    claim: str
    domain: str = ""
    category: str = ""
    source: str = ""
    trust_score: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)


class DecisionTraceEntry(BaseModel):
    """Decision trace from the analysis."""

    category: str = ""
    conviction: str = ""
    composite_score: float = Field(default=0.0)
    margin_to_next_category: float = 0.0
    rationale_for: list[str] = Field(default_factory=list)
    rationale_against: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class BenchmarkContext(BaseModel):
    """Benchmark context for the due diligence report."""

    composite_score: float = Field(default=0.0)
    benchmark_mean: float = Field(default=0.0)
    benchmark_std_dev: float = Field(default=0.0)
    percentile_rank: float = Field(default=0.0)
    sample_size: int = 0
    category_distribution: dict[str, int] = Field(default_factory=dict)


class DueDiligenceReportResponse(BaseModel):
    """Complete due diligence report."""

    report_id: str | None = None
    startup_name: str
    executive_summary: ExecutiveSummary = Field(
        default_factory=ExecutiveSummary
    )
    strengths: list[StrengthItem] = Field(default_factory=list)
    weaknesses: list[WeaknessItem] = Field(default_factory=list)
    opportunities: list[OpportunityItemResponse] = Field(default_factory=list)
    risks: list[RiskItemResponse] = Field(default_factory=list)
    evidence: list[EvidenceEntry] = Field(default_factory=list)
    decision_trace: DecisionTraceEntry = Field(
        default_factory=DecisionTraceEntry
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_features: dict[str, object] = Field(default_factory=dict)
    benchmark_context: BenchmarkContext = Field(
        default_factory=BenchmarkContext
    )
    processing_time_ms: float | None = None
    engine_version: str | None = None
