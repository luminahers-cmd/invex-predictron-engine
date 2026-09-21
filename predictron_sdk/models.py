"""Strongly-typed request and response models for the Predictron SDK.

Every API response is deserialized into one of these pydantic models — the
SDK never exposes raw ``dict`` payloads. Request models serialize into the
exact JSON shape the Predictron API expects.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "to_payload",
    # Shared / health
    "Health",
    "Readiness",
    "ErrorBody",
    # Analyze
    "AnalyzeRequest",
    "AnalysisScores",
    "AnalyzeResponse",
    "AnalysisSummary",
    "AnalysisDetail",
    "AnalysisList",
    # Venture
    "VentureRequest",
    "VentureAnalysis",
    "DimensionScore",
    "Explanation",
    "Contribution",
    "DecisionExplain",
    "TraceNode",
    "DecisionTrace",
    "FeatureSnapshot",
    "FeatureList",
    "GraphNode",
    "GraphEdge",
    "KnowledgeGraphSummary",
    "Signal",
    "SignalTimeline",
    "TrendResult",
    "SignalTrends",
    "SignalAggregation",
    "BenchmarkComparison",
    "GroundTruthEntry",
    "GroundTruthHistory",
    # Portfolio
    "PortfolioRequest",
    "PortfolioCompanyScore",
    "SectorDistribution",
    "StageDistribution",
    "RiskSummary",
    "ConcentrationRisk",
    "DiversificationScore",
    "HeatmapCell",
    "PortfolioAnalysis",
    "PortfolioComparisonItem",
    "SimilarityMatrix",
    # Comparison
    "ComparisonRequest",
    "FeatureDiff",
    "FeatureDiffs",
    "DecisionDiffs",
    "ContributionDiffItem",
    "ContributionDiffs",
    "SignalDiffs",
    "KnowledgeGraphDiffs",
    "BenchmarkComparisonDetail",
    "FullComparison",
    # Due diligence
    "DueDiligenceRequest",
    "ExecutiveSummary",
    "Strength",
    "Weakness",
    "Opportunity",
    "RiskItem",
    "EvidenceEntry",
    "DecisionTraceEntry",
    "BenchmarkContext",
    "DueDiligenceReport",
    # Search
    "SearchRequest",
    "SearchResultItem",
    "SearchResponse",
    "CompanySearchResult",
    "SignalSearchResult",
    "GraphNodeSearchResult",
    "SearchByType",
    # Company Intelligence Hub
    "CompanySnapshot",
    "CompanyHistory",
    "CompanyCompleteness",
    "CompanyOfflineSummary",
    "CompanyGraphNeighbor",
    "CompanyGraphSummary",
    "CompanyFeature",
    "CompanyFeatureSummary",
    "CompanySignalSummary",
    "CompanyBenchmarkSummary",
    "CompanyDecisionSummary",
    "CompanyProfile",
    # Batch
    "JobStatus",
    "BatchSubmission",
    "BatchItem",
    "BatchJobSummary",
    "BatchJobList",
    "BatchJobDetail",
    "BatchJobResultItem",
    "BatchJobProgress",
    "BatchJobCancelled",
    "BatchEvent",
    # Monitor (Phase 6)
    "MonitorDistribution",
    "MonitorHorizonPerformance",
    "MonitorSectorPerformance",
    "MonitorTimePoint",
    "MonitorMetricTrend",
    "MonitorHealthEntry",
    "MonitorDriftSignal",
    "MonitorReanalysisRecommendation",
    "MonitorSummary",
    "MonitorHealthList",
    "MonitorDrift",
    "MonitorTrends",
    "MonitorReanalysis",
]


def to_payload(model: BaseModel) -> dict[str, Any]:
    """Serialize a request model into the JSON body sent to the API."""
    return model.model_dump(mode="json", exclude_none=True)


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Health / errors
# ---------------------------------------------------------------------------


class Health(_Base):
    """Health check response."""

    status: str = "ok"
    version: str = Field(default="")
    engine_reachable: bool = False
    db_healthy: bool = False
    startup_state: str = "unknown"


class Readiness(_Base):
    """Readiness check response."""

    status: str = "ok"
    db_healthy: bool = False
    engine_ready: bool = False
    startup_complete: bool = False


class ErrorBody(_Base):
    """Standard API error body."""

    detail: str = ""
    code: str | None = None


# ---------------------------------------------------------------------------
# Analyze (legacy single-startup analysis)
# ---------------------------------------------------------------------------


class AnalyzeRequest(_Base):
    """Request payload for ``POST /analyze``."""

    startup_name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=10, max_length=5000)
    website_url: str | None = None
    pitch_deck_url: str | None = None
    founder_linkedin_urls: list[str] = Field(default_factory=list, max_length=10)


class AnalysisScores(_Base):
    """Individual score breakdown."""

    venture_score: float = Field(..., ge=0, le=100)
    market_score: float = Field(..., ge=0, le=100)
    founder_score: float = Field(..., ge=0, le=100)
    traction_score: float = Field(..., ge=0, le=100)


class AnalyzeResponse(_Base):
    """Response payload for ``POST /analyze``."""

    id: str | None = None
    startup_name: str
    venture_score: float = Field(..., ge=0, le=100)
    market_score: float = Field(..., ge=0, le=100)
    founder_score: float = Field(..., ge=0, le=100)
    traction_score: float = Field(..., ge=0, le=100)
    recommendations: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)


class AnalysisSummary(_Base):
    """Summary of a persisted analysis."""

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


class AnalysisDetail(_Base):
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
    full_report: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class AnalysisList(_Base):
    """Paginated list of persisted analyses."""

    analyses: list[AnalysisSummary] = Field(default_factory=list)
    total: int = 0


# ---------------------------------------------------------------------------
# Venture analysis
# ---------------------------------------------------------------------------


class VentureRequest(_Base):
    """Request payload for the venture analysis endpoints."""

    startup_name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=10, max_length=5000)
    website_url: str | None = None
    pitch_deck_url: str | None = None
    founder_linkedin_urls: list[str] = Field(default_factory=list, max_length=10)


class DimensionScore(_Base):
    """Score for a single analysis dimension."""

    dimension: str
    score: float = Field(..., ge=0.0, le=100.0)
    rationale: str = ""


class VentureAnalysis(_Base):
    """Full venture analysis response."""

    id: str | None = None
    startup_name: str
    overall_score: float = Field(..., ge=0.0, le=100.0)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    dimension_scores: list[DimensionScore] = Field(default_factory=list)
    decision_category: str | None = None
    conviction_level: str | None = None
    recommendation_count: int = Field(default=0, ge=0)
    key_recommendations: list[str] = Field(default_factory=list)
    processing_time_ms: float | None = None
    engine_version: str | None = None
    created_at: datetime | None = None


class Explanation(_Base):
    """Human-readable explanation of an investment decision."""

    company_id: str
    headline: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    neutral_factors: list[str] = Field(default_factory=list)
    confidence_factors: list[str] = Field(default_factory=list)
    evidence_summary: str = ""
    recommendation: str = ""


class Contribution(_Base):
    """Single feature contribution to a decision."""

    feature_id: str
    feature_name: str
    category: str
    contribution_type: str
    raw_value: Any = None
    normalized_value: float = 0.0
    weight: float = 0.0
    computed_contribution: float = 0.0
    human_explanation: str = ""
    supporting_evidence: str = ""


class DecisionExplain(_Base):
    """Full decision explainability response."""

    company_id: str
    overall_score: float = 0.0
    verdict: str = ""
    confidence: float = 0.0
    explanation: Explanation | None = None
    positive_contributions: list[Contribution] = Field(default_factory=list)
    negative_contributions: list[Contribution] = Field(default_factory=list)
    top_strengths: list[str] = Field(default_factory=list)
    top_weaknesses: list[str] = Field(default_factory=list)


class TraceNode(_Base):
    """Single node in the decision reasoning graph."""

    node_id: str
    node_type: str
    feature_id: str | None = None
    label: str = ""
    value: Any = None
    value_type: str = ""
    rule: str | None = None
    input_node_ids: list[str] = Field(default_factory=list)
    output_node_ids: list[str] = Field(default_factory=list)


class DecisionTrace(_Base):
    """Complete decision reasoning trace."""

    company_id: str
    trace_id: str
    overall_score: float = 0.0
    verdict: str = ""
    confidence: float = 0.0
    nodes: list[TraceNode] = Field(default_factory=list)
    edges: list[list[str]] = Field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0


class FeatureSnapshot(_Base):
    """Single feature snapshot."""

    feature_id: str
    feature_name: str
    category: str
    value: Any = None
    value_type: str = ""
    status: str = ""
    computed_at: str | None = None


class FeatureList(_Base):
    """List of feature snapshots for a company."""

    company_id: str
    feature_count: int = 0
    features: list[FeatureSnapshot] = Field(default_factory=list)


class GraphNode(_Base):
    """Knowledge graph node."""

    node_id: str
    node_type: str
    label: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)


class GraphEdge(_Base):
    """Knowledge graph edge."""

    edge_type: str
    source_id: str
    target_id: str
    properties: dict[str, Any] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)


class KnowledgeGraphSummary(_Base):
    """Knowledge graph summary for the dataset."""

    company_id: str | None = None
    node_count: int = 0
    edge_count: int = 0
    node_type_counts: dict[str, int] = Field(default_factory=dict)
    edge_type_counts: dict[str, int] = Field(default_factory=dict)
    connected_components: int = 0
    density: float = 0.0
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    top_connected_nodes: list[dict[str, Any]] = Field(default_factory=list)


class Signal(_Base):
    """Single company signal."""

    signal_id: str = ""
    company_id: str = ""
    signal_type: str = ""
    timestamp: str = ""
    source: str = ""
    provenance: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SignalTimeline(_Base):
    """Signal timeline for a company."""

    company_id: str
    signal_count: int = 0
    signals: list[Signal] = Field(default_factory=list)
    type_counts: dict[str, int] = Field(default_factory=dict)
    span_days: float = 0.0
    first_signal: str | None = None
    last_signal: str | None = None


class TrendResult(_Base):
    """Single trend analysis result."""

    name: str
    value: float
    direction: str
    window_days: int
    explanation: str
    available: bool = True


class SignalTrends(_Base):
    """All trend results for a company."""

    company_id: str
    trends: dict[str, TrendResult] = Field(default_factory=dict)


class SignalAggregation(_Base):
    """Signal aggregation metrics."""

    company_id: str
    recent_activity: dict[str, Any] = Field(default_factory=dict)
    momentum_score: float = 0.0
    signal_freshness: dict[str, Any] = Field(default_factory=dict)
    funding_cadence: dict[str, Any] = Field(default_factory=dict)


class GroundTruthEntry(_Base):
    """Single ground truth evaluation entry."""

    evaluation_id: str = ""
    record_id: str = ""
    startup_name: str = ""
    verdict: str = ""
    alignment: str = ""
    decision_match: bool | None = None
    confidence_accuracy: float | None = None
    created_at: str | None = None


class GroundTruthHistory(_Base):
    """Historical ground truth evaluations."""

    startup_name: str
    total_evaluations: int = 0
    evaluations: list[GroundTruthEntry] = Field(default_factory=list)


class BenchmarkComparison(_Base):
    """Compare an analysis against historical benchmarks."""

    startup_name: str
    composite_score: float = 0.0
    benchmark_mean: float = 0.0
    benchmark_std_dev: float = 0.0
    percentile_rank: float = 0.0
    score_z_score: float = 0.0
    dimension_comparisons: dict[str, dict[str, float]] = Field(default_factory=dict)
    category_distribution: dict[str, int] = Field(default_factory=dict)
    sample_size: int = 0


# ---------------------------------------------------------------------------
# Portfolio analysis
# ---------------------------------------------------------------------------


class PortfolioRequest(_Base):
    """Request payload for portfolio endpoints."""

    company_names: list[str] = Field(..., min_length=2, max_length=50)
    descriptions: dict[str, str] = Field(default_factory=dict)
    website_urls: dict[str, str | None] = Field(default_factory=dict)


class PortfolioCompanyScore(_Base):
    """Score summary for a single portfolio company."""

    startup_name: str
    overall_score: float = Field(..., ge=0.0, le=100.0)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    decision_category: str | None = None
    conviction_level: str | None = None


class SectorDistribution(_Base):
    """Sector breakdown within a portfolio."""

    sector: str
    count: int
    percentage: float = Field(..., ge=0.0, le=100.0)
    avg_score: float = Field(default=0.0, ge=0.0, le=100.0)


class StageDistribution(_Base):
    """Stage breakdown within a portfolio."""

    stage: str
    count: int
    percentage: float = Field(..., ge=0.0, le=100.0)


class RiskSummary(_Base):
    """Aggregated risk information for a portfolio."""

    high_risk_count: int = 0
    medium_risk_count: int = 0
    low_risk_count: int = 0
    average_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_factors: list[str] = Field(default_factory=list)


class ConcentrationRisk(_Base):
    """Concentration risk analysis."""

    sector_concentration: float = Field(default=0.0, ge=0.0, le=1.0)
    stage_concentration: float = Field(default=0.0, ge=0.0, le=1.0)
    score_variance: float = Field(default=0.0, ge=0.0)
    most_concentrated_sector: str | None = None
    recommendations: list[str] = Field(default_factory=list)


class DiversificationScore(_Base):
    """Portfolio diversification metrics."""

    sector_diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    stage_diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    geography_diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_diversification: float = Field(default=0.0, ge=0.0, le=1.0)
    recommendation_count: int = 0


class HeatmapCell(_Base):
    """Single cell in the portfolio heatmap."""

    row_label: str
    col_label: str
    value: float
    label: str = ""


class PortfolioAnalysis(_Base):
    """Complete portfolio analysis response."""

    company_count: int = 0
    companies: list[PortfolioCompanyScore] = Field(default_factory=list)
    portfolio_score: float = Field(default=0.0, ge=0.0, le=100.0)
    sector_distribution: list[SectorDistribution] = Field(default_factory=list)
    stage_distribution: list[StageDistribution] = Field(default_factory=list)
    risk_summary: RiskSummary = Field(default_factory=RiskSummary)
    diversification: DiversificationScore = Field(default_factory=DiversificationScore)
    concentration_risk: ConcentrationRisk = Field(default_factory=ConcentrationRisk)
    heatmap_data: list[HeatmapCell] = Field(default_factory=list)
    similarity_matrix: list[dict[str, Any]] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time_ms: float | None = None


class PortfolioComparisonItem(_Base):
    """Single company in a comparison matrix."""

    startup_name: str
    overall_score: float = 0.0
    confidence: float = 0.0
    decision: str = ""
    dimension_scores: dict[str, float] = Field(default_factory=dict)


class SimilarityMatrix(_Base):
    """Pairwise similarity matrix for portfolio companies."""

    companies: list[str] = Field(default_factory=list)
    matrix: list[list[float]] = Field(default_factory=list)
    method: str = "score_cosine"


# ---------------------------------------------------------------------------
# Company comparison
# ---------------------------------------------------------------------------


class ComparisonRequest(_Base):
    """Request to compare two or more startups."""

    company_names: list[str] = Field(..., min_length=2, max_length=10)
    descriptions: dict[str, str] = Field(default_factory=dict)
    website_urls: dict[str, str | None] = Field(default_factory=dict)


class FeatureDiff(_Base):
    """Feature difference between two companies."""

    feature: str
    company_a: Any = None
    company_b: Any = None
    difference: float = 0.0
    direction: str = "equal"


class FeatureDiffs(_Base):
    """Feature comparison between all companies."""

    companies: list[str] = Field(default_factory=list)
    feature_diffs: list[FeatureDiff] = Field(default_factory=list)


class DecisionDiffs(_Base):
    """Decision comparison between companies."""

    companies: list[str] = Field(default_factory=list)
    decision_comparison: list[dict[str, Any]] = Field(default_factory=list)


class ContributionDiffItem(_Base):
    """Contribution difference for a single feature."""

    feature: str
    contributions: dict[str, float] = Field(default_factory=dict)
    max_contribution: float = 0.0
    min_contribution: float = 0.0


class ContributionDiffs(_Base):
    """Contribution comparison across companies."""

    companies: list[str] = Field(default_factory=list)
    diffs: list[ContributionDiffItem] = Field(default_factory=list)


class SignalDiffs(_Base):
    """Signal comparison across companies."""

    companies: list[str] = Field(default_factory=list)
    signal_comparison: list[dict[str, Any]] = Field(default_factory=list)


class KnowledgeGraphDiffs(_Base):
    """Knowledge graph comparison across companies."""

    companies: list[str] = Field(default_factory=list)
    graph_summary: dict[str, dict[str, Any]] = Field(default_factory=dict)


class BenchmarkComparisonDetail(_Base):
    """Benchmark comparison detail for each company."""

    companies: list[str] = Field(default_factory=list)
    benchmark_metrics: list[dict[str, Any]] = Field(default_factory=list)


class FullComparison(_Base):
    """Complete comparison response across all dimensions."""

    companies: list[str] = Field(default_factory=list)
    feature_diffs: FeatureDiffs = Field(default_factory=FeatureDiffs)
    decision_diffs: DecisionDiffs = Field(default_factory=DecisionDiffs)
    contribution_diffs: ContributionDiffs = Field(default_factory=ContributionDiffs)
    signal_diffs: SignalDiffs = Field(default_factory=SignalDiffs)
    knowledge_graph_diffs: KnowledgeGraphDiffs = Field(
        default_factory=KnowledgeGraphDiffs
    )
    benchmark_comparison: BenchmarkComparisonDetail = Field(
        default_factory=BenchmarkComparisonDetail
    )
    overall_summary: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Due diligence
# ---------------------------------------------------------------------------


class DueDiligenceRequest(_Base):
    """Request to generate a due diligence report."""

    startup_name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=10, max_length=5000)
    website_url: str | None = None
    pitch_deck_url: str | None = None
    founder_linkedin_urls: list[str] = Field(default_factory=list, max_length=10)


class ExecutiveSummary(_Base):
    """Executive summary section of a due diligence report."""

    headline: str = ""
    overview: str = ""
    key_findings: list[str] = Field(default_factory=list)
    confidence_level: float = Field(default=0.0, ge=0.0, le=1.0)


class Strength(_Base):
    """Single strength entry."""

    title: str
    description: str = ""
    dimension: str = ""
    severity: str = "moderate"
    evidence: list[str] = Field(default_factory=list)


class Weakness(_Base):
    """Single weakness entry."""

    title: str
    description: str = ""
    dimension: str = ""
    severity: str = "moderate"
    evidence: list[str] = Field(default_factory=list)


class Opportunity(_Base):
    """Single opportunity entry."""

    title: str
    description: str = ""
    dimension: str = ""
    impact: str = "moderate"
    evidence: list[str] = Field(default_factory=list)


class RiskItem(_Base):
    """Single risk entry."""

    title: str
    description: str = ""
    dimension: str = ""
    severity: str = "low"
    mitigation: str = ""
    evidence: list[str] = Field(default_factory=list)


class EvidenceEntry(_Base):
    """Evidence supporting a finding."""

    claim: str
    domain: str = ""
    category: str = ""
    source: str = ""
    trust_score: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)


class DecisionTraceEntry(_Base):
    """Decision trace from the analysis."""

    category: str = ""
    conviction: str = ""
    composite_score: float = 0.0
    margin_to_next_category: float = 0.0
    rationale_for: list[str] = Field(default_factory=list)
    rationale_against: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class BenchmarkContext(_Base):
    """Benchmark context for a due diligence report."""

    composite_score: float = 0.0
    benchmark_mean: float = 0.0
    benchmark_std_dev: float = 0.0
    percentile_rank: float = 0.0
    sample_size: int = 0
    category_distribution: dict[str, int] = Field(default_factory=dict)


class DueDiligenceReport(_Base):
    """Complete due diligence report."""

    report_id: str | None = None
    startup_name: str
    executive_summary: ExecutiveSummary = Field(default_factory=ExecutiveSummary)
    strengths: list[Strength] = Field(default_factory=list)
    weaknesses: list[Weakness] = Field(default_factory=list)
    opportunities: list[Opportunity] = Field(default_factory=list)
    risks: list[RiskItem] = Field(default_factory=list)
    evidence: list[EvidenceEntry] = Field(default_factory=list)
    decision_trace: DecisionTraceEntry = Field(default_factory=DecisionTraceEntry)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_features: dict[str, Any] = Field(default_factory=dict)
    benchmark_context: BenchmarkContext = Field(default_factory=BenchmarkContext)
    processing_time_ms: float | None = None
    engine_version: str | None = None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchRequest(_Base):
    """Unified search request."""

    query: str = Field(..., min_length=1, max_length=500)
    search_type: str = "all"
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)


class SearchResultItem(_Base):
    """Single search result."""

    result_type: str
    id: str
    name: str = ""
    description: str = ""
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(_Base):
    """Paginated search response."""

    query: str = ""
    search_type: str = "all"
    total: int = Field(default=0, ge=0)
    offset: int = 0
    limit: int = 20
    results: list[SearchResultItem] = Field(default_factory=list)


class CompanySearchResult(_Base):
    """Company search result."""

    record_id: str = ""
    startup_name: str = ""
    website: str = ""
    industries: list[str] = Field(default_factory=list)
    country_code: str | None = None
    headquarters: str | None = None
    founded_year: int | None = None
    decision: str = ""
    composite_score: float = 0.0
    confidence: float = 0.0


class SignalSearchResult(_Base):
    """Signal search result."""

    signal_id: str = ""
    company_id: str = ""
    signal_type: str = ""
    timestamp: str = ""
    source: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphNodeSearchResult(_Base):
    """Knowledge graph node search result."""

    node_id: str = ""
    node_type: str = ""
    label: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    connected_companies: list[str] = Field(default_factory=list)


class SearchByType(_Base):
    """Type-specific search response."""

    search_type: str = ""
    total: int = 0
    offset: int = 0
    limit: int = 20
    companies: list[CompanySearchResult] = Field(default_factory=list)
    signals: list[SignalSearchResult] = Field(default_factory=list)
    graph_nodes: list[GraphNodeSearchResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Company Intelligence Hub
# ---------------------------------------------------------------------------


class CompanySnapshot(_Base):
    """One stored company analysis snapshot."""

    id: str
    company_id: str = ""
    analysis_id: str = ""
    report_id: str = ""
    decision: str | None = None
    confidence: float | None = None
    composite_score: float | None = None
    readiness_score: float | None = None
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    created_at: datetime | None = None


class CompanyHistory(_Base):
    """Aggregated history of a company's stored snapshots."""

    snapshot_count: int = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    latest_decision: str | None = None
    latest_confidence: float | None = None
    latest_composite_score: float | None = None


class CompanyCompleteness(_Base):
    """Field-completeness of a company identity."""

    total_fields: int = 0
    populated_fields: int = 0
    fraction: float = 0.0
    missing_fields: list[str] = Field(default_factory=list)


class CompanyOfflineSummary(_Base):
    """Summary of the matching historical dataset records."""

    record_count: int = 0
    sources: list[str] = Field(default_factory=list)
    placeholder: bool = False
    placeholder_count: int = 0
    placeholder_reasons: list[str] = Field(default_factory=list)
    first_analysis_date: datetime | None = None
    last_analysis_date: datetime | None = None


class CompanyGraphNeighbor(_Base):
    """An adjacent knowledge-graph node."""

    node_id: str = ""
    node_type: str = ""
    label: str = ""
    edge_types: list[str] = Field(default_factory=list)


class CompanyGraphSummary(_Base):
    """Knowledge-graph summary of a company."""

    node_id: str | None = None
    degree: int = 0
    neighbor_count: int = 0
    neighbors: list[CompanyGraphNeighbor] = Field(default_factory=list)


class CompanyFeature(_Base):
    """One computed feature exposed in the profile."""

    feature_id: str = ""
    feature_name: str = ""
    category: str = ""
    value: Any = None
    value_type: str = ""
    status: str = ""


class CompanyFeatureSummary(_Base):
    """Feature summary of a company."""

    feature_count: int = 0
    computed_count: int = 0
    failed_count: int = 0
    by_category: dict[str, int] = Field(default_factory=dict)
    features: list[CompanyFeature] = Field(default_factory=list)


class CompanySignalSummary(_Base):
    """Signal-timeline summary of a company."""

    signal_count: int = 0
    type_counts: dict[str, int] = Field(default_factory=dict)
    span_days: float = 0.0
    first_signal_at: datetime | None = None
    last_signal_at: datetime | None = None
    momentum_score: float = 0.0
    is_active: bool = False
    highlights: list[str] = Field(default_factory=list)


class CompanyProfile(_Base):
    """Unified intelligence profile of one company."""

    company_id: str
    canonical_name: str = ""
    primary_name: str = ""
    canonical_domain: str | None = None
    website: str | None = None
    coverage: str = "none"
    coverage_reasons: list[str] = Field(default_factory=list)
    completeness: CompanyCompleteness = Field(default_factory=CompanyCompleteness)
    latest_snapshot: CompanySnapshot | None = None
    history: CompanyHistory = Field(default_factory=CompanyHistory)
    offline: CompanyOfflineSummary | None = None
    graph_summary: CompanyGraphSummary | None = None
    feature_summary: CompanyFeatureSummary | None = None
    signal_summary: CompanySignalSummary | None = None
    benchmark_summary: CompanyBenchmarkSummary | None = None
    decision_summary: CompanyDecisionSummary | None = None
    generated_at: datetime | None = None


class CompanyBenchmarkSummary(_Base):
    """Benchmark placement of the company's composite score."""

    composite_score: float = 0.0
    benchmark_mean: float = 0.0
    benchmark_std_dev: float = 0.0
    percentile_rank: float = 0.0
    z_score: float = 0.0
    sample_size: int = 0


class CompanyDecisionSummary(_Base):
    """Decision-trace summary derived from the offline feature snapshot."""

    verdict: str = ""
    confidence: float = 0.0
    overall_score: float = 0.0
    feature_count: int = 0
    positive_factor_count: int = 0
    negative_factor_count: int = 0
    node_count: int = 0
    edge_count: int = 0
    trace_id: str | None = None
    top_strengths: list[str] = Field(default_factory=list)
    top_weaknesses: list[str] = Field(default_factory=list)
    source_record_id: str | None = None


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    """Batch job status values."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchItem(_Base):
    """A single item within a batch submission."""

    startup_name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(default=None, min_length=10, max_length=5000)
    website_url: str | None = None
    pitch_deck_url: str | None = None
    founder_linkedin_urls: list[str] = Field(default_factory=list, max_length=10)

    def to_payload(self) -> dict[str, Any]:
        return to_payload(self)


class BatchSubmission(_Base):
    """Request to submit a batch of analyses."""

    job_name: str = Field(default="", max_length=255)
    items: list[dict[str, Any]] = Field(..., min_length=1, max_length=1000)


class BatchJobSummary(_Base):
    """Summary of a batch job."""

    job_id: str
    job_name: str = ""
    status: JobStatus = JobStatus.PENDING
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None


class BatchJobList(_Base):
    """List of batch jobs."""

    jobs: list[BatchJobSummary] = Field(default_factory=list)
    total: int = 0


class BatchJobResultItem(_Base):
    """Single item result within a batch job."""

    index: int = 0
    status: JobStatus = JobStatus.PENDING
    startup_name: str = ""
    result_id: str | None = None
    error: str | None = None
    processing_time_ms: float | None = None


class BatchJobDetail(_Base):
    """Detailed batch job status with results."""

    job_id: str
    job_name: str = ""
    status: JobStatus = JobStatus.PENDING
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
    results: list[BatchJobResultItem] = Field(default_factory=list)
    cursor: str | None = None


class BatchJobProgress(_Base):
    """Lightweight progress-only response."""

    job_id: str
    status: JobStatus = JobStatus.PENDING
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    progress_pct: float = 0.0
    current_item: str | None = None


class BatchJobCancelled(_Base):
    """Response body returned when a batch job is cancelled."""

    status: str = "cancelled"
    job_id: str = ""


class BatchEvent(_Base):
    """Event emitted during batch processing."""

    event_type: str = "progress"
    job_id: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str | None = None


# ---------------------------------------------------------------------------
# Monitoring (CIH Phase 6)
# ---------------------------------------------------------------------------


class MonitorDistribution(_Base):
    """Label-universe frequency distribution of one analytic dimension."""

    label: str = ""
    universe: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


class MonitorHorizonPerformance(_Base):
    """Per-horizon bucket performance."""

    horizon_days: int = 0
    forecasts: int = 0
    evaluated: int = 0
    scoreable: int = 0
    accuracy: float | None = None


class MonitorSectorPerformance(_Base):
    """Per-sector bucket performance."""

    sector: str = ""
    evaluations: int = 0
    scoreable: int = 0
    accuracy: float | None = None


class MonitorTimePoint(_Base):
    """One anchor-ordered point in a metric time-series."""

    anchor: date = Field(...)
    value: float | None = None


class MonitorMetricTrend(_Base):
    """Deterministic trend over a metric time-series."""

    metric: str = ""
    direction: str = "flat"
    from_value: float | None = None
    to_value: float | None = None
    series: list[MonitorTimePoint] = Field(default_factory=list)


class MonitorHealthEntry(_Base):
    """Derived health projection of one forecast."""

    forecast_id: str = ""
    company_id: str = ""
    snapshot_id: str = ""
    decision: str = ""
    confidence: float = 0.0
    status: str = ""
    health: str = ""
    analysis_timestamp: str | None = None
    due_at: str | None = None
    as_of: str | None = None
    age_days: int = 0
    days_until_due: float = 0.0
    days_overdue: float = 0.0
    outcome_id: str | None = None
    outcome_verdict: str | None = None
    evaluation_status: str = "pending"
    evaluation_verdict: str | None = None


class MonitorDriftSignal(_Base):
    """One dimension of the deterministic drift report."""

    signal: str = ""
    from_value: float = 0.0
    to_value: float = 0.0
    delta: float = 0.0
    magnitude: float = 0.0
    direction: str = "flat"
    severity: str = "low"
    affected: bool = False
    detail: dict[str, Any] = Field(default_factory=dict)


class MonitorReanalysisRecommendation(_Base):
    """Deterministic re-analysis recommendation for one frozen forecast."""

    company_id: str = ""
    forecast_id: str = ""
    snapshot_id: str = ""
    reasons: list[str] = Field(default_factory=list)
    latest_outcome_at: str | None = None
    latest_snapshot_at: str | None = None


class MonitorSummary(_Base):
    """Live monitoring summary over the caller's scope."""

    scope: str = ""
    period_kind: str = "daily"
    anchor_date: date | None = None
    generated_at: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    distributions: dict[str, MonitorDistribution] = Field(default_factory=dict)
    horizon_breakdown: dict[str, MonitorHorizonPerformance] = Field(
        default_factory=dict
    )
    sector_breakdown: dict[str, MonitorSectorPerformance] = Field(
        default_factory=dict
    )
    health: dict[str, int] = Field(default_factory=dict)


class MonitorHealthList(_Base):
    """Derived forecast-health rows plus their distribution."""

    as_of: str | None = None
    generated_at: str | None = None
    scope: str = ""
    entries: list[MonitorHealthEntry] = Field(default_factory=list)
    distribution: dict[str, int] = Field(default_factory=dict)


class MonitorDrift(_Base):
    """Deterministic drift comparison between two snapshot populations."""

    baseline_id: str = ""
    comparison_id: str = ""
    baseline_period: date | None = None
    comparison_period: date | None = None
    signals: list[MonitorDriftSignal] = Field(default_factory=list)


class MonitorTrends(_Base):
    """Dashboard-consumable time-series for the requested metrics."""

    period_kind: str = "daily"
    as_of: date | None = None
    trends: list[MonitorMetricTrend] = Field(default_factory=list)


class MonitorReanalysis(_Base):
    """Deterministic re-analysis recommendations for the caller's scope."""

    as_of: str | None = None
    generated_at: str | None = None
    scope: str = ""
    recommendations: list[MonitorReanalysisRecommendation] = Field(
        default_factory=list
    )
