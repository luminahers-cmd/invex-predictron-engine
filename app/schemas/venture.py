"""Venture Analysis API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class VentureAnalysisRequest(BaseModel):
    """Request payload for a full venture intelligence analysis."""

    startup_name: str = Field(
        ..., min_length=1, max_length=255, description="Startup name"
    )
    website_url: str | None = Field(
        default=None, description="Startup website URL (optional)"
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
        default_factory=list,
        max_length=10,
        description="Founder LinkedIn profile URLs",
    )


# ---------------------------------------------------------------------------
# Response — Venture Score breakdown
# ---------------------------------------------------------------------------


class DimensionScore(BaseModel):
    """Score for a single analysis dimension."""

    dimension: str = Field(..., description="Dimension identifier")
    score: float = Field(..., ge=0.0, le=100.0, description="Score (0-100)")
    rationale: str = Field(default="", description="Score rationale")


class VentureAnalysisResponse(BaseModel):
    """Full venture analysis response."""

    id: str | None = Field(
        default=None, description="Persisted analysis ID"
    )
    startup_name: str
    overall_score: float = Field(..., ge=0.0, le=100.0)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    dimension_scores: list[DimensionScore] = Field(default_factory=list)
    decision_category: str | None = Field(
        default=None, description="Investment decision category"
    )
    conviction_level: str | None = Field(
        default=None, description="Conviction level"
    )
    recommendation_count: int = Field(default=0, ge=0)
    key_recommendations: list[str] = Field(default_factory=list)
    processing_time_ms: float | None = None
    engine_version: str | None = None
    created_at: datetime | None = None


# ---------------------------------------------------------------------------
# Decision Explainability
# ---------------------------------------------------------------------------


class ExplanationResponse(BaseModel):
    """Human-readable explanation of an investment decision."""

    company_id: str
    headline: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    neutral_factors: list[str] = Field(default_factory=list)
    confidence_factors: list[str] = Field(default_factory=list)
    evidence_summary: str = Field(default="")
    recommendation: str = Field(default="")


class ContributionResponse(BaseModel):
    """Single feature contribution to the decision."""

    feature_id: str
    feature_name: str
    category: str
    contribution_type: str
    raw_value: object = None
    normalized_value: float = Field(default=0.0)
    weight: float = Field(default=0.0)
    computed_contribution: float = Field(default=0.0)
    human_explanation: str = Field(default="")
    supporting_evidence: str = Field(default="")


class DecisionExplainResponse(BaseModel):
    """Full decision explainability response."""

    company_id: str
    overall_score: float = Field(default=0.0)
    verdict: str = Field(default="")
    confidence: float = Field(default=0.0)
    explanation: ExplanationResponse | None = None
    positive_contributions: list[ContributionResponse] = Field(default_factory=list)
    negative_contributions: list[ContributionResponse] = Field(default_factory=list)
    top_strengths: list[str] = Field(default_factory=list)
    top_weaknesses: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Decision Trace
# ---------------------------------------------------------------------------


class TraceNodeResponse(BaseModel):
    """Single node in the decision reasoning graph."""

    node_id: str
    node_type: str
    feature_id: str | None = None
    label: str = ""
    value: object = None
    value_type: str = ""
    rule: str | None = None
    input_node_ids: list[str] = Field(default_factory=list)
    output_node_ids: list[str] = Field(default_factory=list)


class DecisionTraceResponse(BaseModel):
    """Complete decision reasoning trace."""

    company_id: str
    trace_id: str
    overall_score: float = 0.0
    verdict: str = ""
    confidence: float = 0.0
    nodes: list[TraceNodeResponse] = Field(default_factory=list)
    edges: list[list[str]] = Field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0


# ---------------------------------------------------------------------------
# Feature Snapshot
# ---------------------------------------------------------------------------


class FeatureSnapshotResponse(BaseModel):
    """Single feature snapshot."""

    feature_id: str
    feature_name: str
    category: str
    value: object = None
    value_type: str = ""
    status: str = ""
    computed_at: str | None = None


class FeatureSnapshotListResponse(BaseModel):
    """List of feature snapshots for a company."""

    company_id: str
    feature_count: int = 0
    features: list[FeatureSnapshotResponse] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Knowledge Graph Summary
# ---------------------------------------------------------------------------


class GraphNodeResponse(BaseModel):
    """Knowledge graph node."""

    node_id: str
    node_type: str
    label: str = ""
    properties: dict[str, object] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)


class GraphEdgeResponse(BaseModel):
    """Knowledge graph edge."""

    edge_type: str
    source_id: str
    target_id: str
    properties: dict[str, object] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)


class KnowledgeGraphSummaryResponse(BaseModel):
    """Knowledge graph summary for a company."""

    company_id: str | None = None
    node_count: int = 0
    edge_count: int = 0
    node_type_counts: dict[str, int] = Field(default_factory=dict)
    edge_type_counts: dict[str, int] = Field(default_factory=dict)
    connected_components: int = 0
    density: float = 0.0
    nodes: list[GraphNodeResponse] = Field(default_factory=list)
    edges: list[GraphEdgeResponse] = Field(default_factory=list)
    top_connected_nodes: list[dict[str, object]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Signals Timeline
# ---------------------------------------------------------------------------


class SignalResponse(BaseModel):
    """Single company signal."""

    signal_id: str = ""
    company_id: str = ""
    signal_type: str = ""
    timestamp: str = ""
    source: str = ""
    provenance: str = ""
    confidence: float = 1.0
    metadata: dict[str, object] = Field(default_factory=dict)


class SignalTimelineResponse(BaseModel):
    """Signal timeline for a company."""

    company_id: str
    signal_count: int = 0
    signals: list[SignalResponse] = Field(default_factory=list)
    type_counts: dict[str, int] = Field(default_factory=dict)
    span_days: float = 0.0
    first_signal: str | None = None
    last_signal: str | None = None


class TrendResultResponse(BaseModel):
    """Single trend analysis result."""

    name: str
    value: float
    direction: str
    window_days: int
    explanation: str
    available: bool = True


class SignalTrendsResponse(BaseModel):
    """All trend results for a company."""

    company_id: str
    trends: dict[str, TrendResultResponse] = Field(default_factory=dict)


class SignalAggregationResponse(BaseModel):
    """Signal aggregation metrics."""

    company_id: str
    recent_activity: dict[str, object] = Field(default_factory=dict)
    momentum_score: float = 0.0
    signal_freshness: dict[str, object] = Field(default_factory=dict)
    funding_cadence: dict[str, object] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Ground Truth History
# ---------------------------------------------------------------------------


class GroundTruthEntryResponse(BaseModel):
    """Single ground truth evaluation entry."""

    evaluation_id: str = ""
    record_id: str = ""
    startup_name: str = ""
    verdict: str = ""
    alignment: str = ""
    decision_match: bool | None = None
    confidence_accuracy: float | None = None
    created_at: str | None = None


class GroundTruthHistoryResponse(BaseModel):
    """Historical ground truth evaluations."""

    startup_name: str
    total_evaluations: int = 0
    evaluations: list[GroundTruthEntryResponse] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Benchmark Comparison
# ---------------------------------------------------------------------------


class BenchmarkComparisonResponse(BaseModel):
    """Compare analysis against historical benchmarks."""

    startup_name: str
    composite_score: float = 0.0
    benchmark_mean: float = 0.0
    benchmark_std_dev: float = 0.0
    percentile_rank: float = 0.0
    score_z_score: float = 0.0
    dimension_comparisons: dict[str, dict[str, float]] = Field(default_factory=dict)
    category_distribution: dict[str, int] = Field(default_factory=dict)
    sample_size: int = 0
