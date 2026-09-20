"""Company Comparison API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ComparisonRequest(BaseModel):
    """Request to compare two or more startups."""

    company_names: list[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Startup names to compare",
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


class FeatureDiff(BaseModel):
    """Feature difference between two companies."""

    feature: str
    company_a: object = None
    company_b: object = None
    difference: float = Field(default=0.0)
    direction: str = Field(default="equal")


class FeatureDiffResponse(BaseModel):
    """Feature comparison between all companies."""

    companies: list[str] = Field(default_factory=list)
    feature_diffs: list[FeatureDiff] = Field(default_factory=list)


class DecisionDiffResponse(BaseModel):
    """Decision comparison between companies."""

    companies: list[str] = Field(default_factory=list)
    decision_comparison: list[dict[str, object]] = Field(default_factory=list)


class ContributionDiffItem(BaseModel):
    """Contribution difference for a single feature."""

    feature: str
    contributions: dict[str, float] = Field(default_factory=dict)
    max_contribution: float = Field(default=0.0)
    min_contribution: float = Field(default=0.0)


class ContributionDiffResponse(BaseModel):
    """Contribution comparison across companies."""

    companies: list[str] = Field(default_factory=list)
    diffs: list[ContributionDiffItem] = Field(default_factory=list)


class SignalDiffResponse(BaseModel):
    """Signal comparison across companies."""

    companies: list[str] = Field(default_factory=list)
    signal_comparison: list[dict[str, object]] = Field(default_factory=list)


class KnowledgeGraphDiffResponse(BaseModel):
    """Knowledge graph comparison across companies."""

    companies: list[str] = Field(default_factory=list)
    graph_summary: dict[str, dict[str, object]] = Field(default_factory=dict)


class BenchmarkComparisonDetail(BaseModel):
    """Benchmark comparison detail for each company."""

    companies: list[str] = Field(default_factory=list)
    benchmark_metrics: list[dict[str, object]] = Field(default_factory=list)


class FullComparisonResponse(BaseModel):
    """Complete comparison response across all dimensions."""

    companies: list[str] = Field(default_factory=list)
    feature_diffs: FeatureDiffResponse = Field(
        default_factory=FeatureDiffResponse
    )
    decision_diffs: DecisionDiffResponse = Field(
        default_factory=DecisionDiffResponse
    )
    contribution_diffs: ContributionDiffResponse = Field(
        default_factory=ContributionDiffResponse
    )
    signal_diffs: SignalDiffResponse = Field(
        default_factory=SignalDiffResponse
    )
    knowledge_graph_diffs: KnowledgeGraphDiffResponse = Field(
        default_factory=KnowledgeGraphDiffResponse
    )
    benchmark_comparison: BenchmarkComparisonDetail = Field(
        default_factory=BenchmarkComparisonDetail
    )
    overall_summary: dict[str, object] = Field(default_factory=dict)
