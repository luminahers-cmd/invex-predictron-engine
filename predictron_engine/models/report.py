"""Output models for the analysis pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field

from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup


class DecisionCategory(str, Enum):
    """Deterministic investment decision categories.

    Categories are ordered from most positive to most negative.
    Each category maps to a deterministic score range derived from
    the composite investment score.
    """

    STRONG_INVEST = "strong_invest"
    INVEST = "invest"
    WATCH = "watch"
    INVESTIGATE_FURTHER = "investigate_further"
    PASS = "pass"


class ConvictionLevel(str, Enum):
    """Investment conviction levels.

    Conviction measures investment attractiveness independently of
    evidence confidence. A startup can have high confidence (reliable
    data) but low conviction (unattractive opportunity), or vice versa.
    """

    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


class EvidenceItem(BaseModel):
    """A single piece of contextual evidence about a startup's domain.

    Evidence items are objective facts retrieved from the knowledge base.
    They are NOT conclusions about the startup — they are observations
    about the domain the startup operates in.
    """

    domain: str = Field(
        ..., description="Feature domain this evidence relates to"
    )
    category: str = Field(
        ..., description="Evidence sub-category (e.g. sales_cycle, regulatory)"
    )
    statement: str = Field(
        ..., description="Objective factual statement about the domain"
    )
    source: str = Field(
        ..., description="Knowledge source that produced this evidence"
    )
    relevance_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How relevant this evidence is to the specific startup",
    )


class Observation(BaseModel):
    """An explainable observation derived from extracted features and evidence.

    Each observation links a conclusion to the specific features and
    evidence that support it, enabling transparent reasoning auditability.
    Every observation carries enough metadata to trace its origin back
    to the specific reasoning rule that produced it.
    """

    dimension: str = Field(
        ..., description="Analysis dimension this observation relates to"
    )
    category: str = Field(
        ..., description="Reasoning category (market_context, team_assessment, etc.)"
    )
    statement: str = Field(..., description="Human-readable observation text")
    evidence: list[str] = Field(
        default_factory=list,
        description="Supporting evidence and feature references",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How strongly the evidence supports this observation",
    )
    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative importance of this observation (0.0-1.0)",
    )
    source_rule: str = Field(
        default="",
        description="Name of the reasoning rule that produced this observation",
    )


class ScoreResult(BaseModel):
    """A numerical score for a single analysis dimension."""

    dimension: str = Field(..., description="Analysis dimension being scored")
    score: float = Field(
        ..., ge=0.0, le=100.0, description="Score value on a 0-100 scale"
    )
    rationale: str = Field(
        default="", description="Explanation of why this score was assigned"
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Observations that contributed to this score",
    )


class Recommendation(BaseModel):
    """An actionable recommendation for the investor.

    v0.6 extended model with rich recommendation fields. All new fields
    have defaults to maintain backward compatibility with v0.5 code
    that creates Recommendation with only category, action, priority,
    and rationale.
    """

    category: str = Field(
        ...,
        description="Recommendation category (e.g. due_diligence, risk, opportunity)",
    )
    action: str = Field(..., description="The specific recommended action")
    priority: str = Field(
        default="medium", description="Priority level (high, medium, low)"
    )
    rationale: str = Field(
        default="", description="Why this recommendation was generated"
    )
    title: str = Field(
        default="", description="Short, descriptive title for the recommendation"
    )
    description: str = Field(
        default="", description="Detailed description of the recommendation"
    )
    supporting_observations: list[Observation] = Field(
        default_factory=list,
        description="Observations that support this recommendation",
    )
    supporting_assessments: list[DimensionAssessment] = Field(
        default_factory=list,
        description="Dimension assessments that support this recommendation",
    )
    expected_impact: str = Field(
        default="",
        description="Expected outcome if this recommendation is followed",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this recommendation (0.0-1.0)",
    )
    action_items: list[str] = Field(
        default_factory=list,
        description="Specific actionable steps to follow this recommendation",
    )
    metadata: dict[str, str | int | float | bool | list[str]] = Field(
        default_factory=dict,
        description="Additional metadata about the recommendation",
    )


class ConfidenceAssessment(BaseModel):
    """Confidence level for a specific scoring dimension."""

    dimension: str = Field(
        ..., description="The scoring dimension this assessment applies to"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence level from 0.0 to 1.0"
    )
    factors: list[str] = Field(
        default_factory=list,
        description="Factors influencing this confidence level",
    )
    data_completeness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of relevant data that was available",
    )


class DimensionAssessment(BaseModel):
    """Explainable assessment for a single analysis dimension.

    Each assessment provides a structured summary of what the observations
    and evidence collectively tell us about a specific venture dimension.
    This is NOT a score — it's an interpretive synthesis that explains
    the dimension's status, strengths, weaknesses, and supporting evidence.
    """

    dimension: str = Field(
        ..., description="Analysis dimension being assessed"
    )
    summary: str = Field(
        ..., description="High-level summary of the dimension's status"
    )
    rationale: str = Field(
        ..., description="Detailed explanation of the assessment reasoning"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this assessment (0.0-1.0)",
    )
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Placeholder score (0-100) for backward compatibility",
    )
    supporting_observations: list[Observation] = Field(
        default_factory=list,
        description="Observations that contributed to this assessment",
    )
    supporting_evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence items that support this assessment",
    )
    metadata: dict[str, str | int | float | bool | list[str]] = Field(
        default_factory=dict,
        description="Additional metadata about the assessment",
    )


class EvaluationResult(BaseModel):
    """Composite result containing all dimension assessments.

    This is the output of the evaluation layer. It aggregates all
    individual dimension assessments into a single result that
    downstream layers (recommendations, confidence, report) consume.
    """

    assessments: list[DimensionAssessment] = Field(
        default_factory=list,
        description="Individual dimension assessments",
    )
    overall_summary: str = Field(
        default="",
        description="High-level summary across all dimensions",
    )
    overall_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average confidence across all assessments",
    )
    dimensions_assessed: int = Field(
        default=0,
        description="Number of dimensions assessed",
    )
    metadata: dict[str, str | int | float | bool | list[str]] = Field(
        default_factory=dict,
        description="Additional metadata about the evaluation",
    )


class SignalRelationship(BaseModel):
    """Describes the relationship between signals from different dimensions."""

    source_dimension: str = Field(
        ..., description="The originating dimension of the first signal"
    )
    target_dimension: str = Field(
        ..., description="The related dimension of the second signal"
    )
    relationship_type: str = Field(
        ...,
        description=(
            "Type of relationship: reinforcing, conflicting, or contextual"
        ),
    )
    description: str = Field(
        ..., description="Human-readable explanation of the relationship"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in this signal relationship",
    )
    source_observations: list[str] = Field(
        default_factory=list,
        description="Category values of the observations that form this relationship",
    )


class InvestmentReadiness(BaseModel):
    """Investment readiness assessment synthesizing all dimensions.

    Provides a holistic view of how ready a startup is for investment,
    combining evidence from all analysis dimensions into a structured
    assessment with strengths, concerns, and gaps.
    """

    readiness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall investment readiness score (0-100)",
    )
    readiness_level: str = Field(
        default="undetermined",
        description=(
            "Qualitative readiness level"
            " (needs_data, early, developing, moderate, strong, investment_ready)"
        ),
    )
    key_strengths: list[str] = Field(
        default_factory=list,
        description="Top strengths identified across all dimensions",
    )
    key_concerns: list[str] = Field(
        default_factory=list,
        description="Top concerns identified across all dimensions",
    )
    dimension_contributions: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-dimension contribution to the readiness score (0-100)"
        ),
    )
    signal_relationships: list[SignalRelationship] = Field(
        default_factory=list,
        description="Detected reinforcing and conflicting signal patterns",
    )
    reinforcing_count: int = Field(
        default=0,
        description="Number of reinforcing cross-signal relationships detected",
    )
    conflicting_count: int = Field(
        default=0,
        description="Number of conflicting cross-signal relationships detected",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Information gaps that would improve the assessment",
    )
    summary: str = Field(
        default="",
        description="Executive summary of the investment readiness assessment",
    )


class AnalysisMetadata(BaseModel):
    """Metadata about the analysis execution itself."""

    engine_version: str = Field(default="0.1.0", description="Engine version")
    pipeline_stages_completed: list[str] = Field(
        default_factory=list,
        description="Pipeline stages that executed successfully",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Total wall-clock processing time in milliseconds",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of analysis completion",
    )
    data_completeness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall data completeness ratio",
    )


class DecisionRationale(BaseModel):
    """Structured rationale explaining an investment decision.

    Every statement must be traceable to evidence from the analysis
    pipeline. The rationale answers:
      - Why this decision?
      - Why not the next higher category?
      - What evidence most influenced the outcome?
      - What additional information could change the decision?
    """

    primary_reasons_for: list[str] = Field(
        default_factory=list,
        description="Primary reasons supporting the decision",
    )
    primary_reasons_against: list[str] = Field(
        default_factory=list,
        description="Primary reasons reducing conviction",
    )
    highest_impact_positive: list[str] = Field(
        default_factory=list,
        description="Highest-impact positive signals",
    )
    highest_impact_negative: list[str] = Field(
        default_factory=list,
        description="Highest-impact negative signals",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="Missing information preventing stronger conviction",
    )
    confidence_explanation: str = Field(
        default="",
        description="Explanation of overall confidence in the decision",
    )
    why_not_higher: str = Field(
        default="",
        description="Why the decision is not the next higher category",
    )
    key_evidence_summary: str = Field(
        default="",
        description="Summary of evidence most influencing the outcome",
    )
    information_that_could_change_decision: list[str] = Field(
        default_factory=list,
        description="Specific information that could change the decision",
    )


class InvestmentDecision(BaseModel):
    """Deterministic investment decision synthesizing all pipeline signals.

    Combines dimension scores, confidence, risk, evidence quality,
    cross-signal reasoning, and recommendations into a single
    actionable decision with full explainability.

    The decision is fully deterministic — the same inputs always
    produce the same decision, category, conviction, and rationale.
    """

    category: DecisionCategory = Field(
        ..., description="Investment decision category"
    )
    conviction: ConvictionLevel = Field(
        ..., description="Investment conviction level"
    )
    composite_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite score used for decision threshold determination",
    )
    rationale: DecisionRationale = Field(
        default_factory=DecisionRationale,
        description="Structured rationale explaining the decision",
    )
    decision_factors: dict[str, float] = Field(
        default_factory=dict,
        description="Quantitative factors contributing to the decision",
    )
    next_category_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Score threshold needed to reach the next higher category",
    )
    margin_to_next_category: float = Field(
        default=0.0,
        description="How far the composite score is from the next higher category",
    )
    data_quality_modifier: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Data quality modifier applied to the composite score (0-1)",
    )
    risk_modifier: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Risk modifier applied to the composite score (0-2)",
    )


class Report(BaseModel):
    """Complete analysis report assembled from all pipeline stages.

    This is the final output of the Predictron Engine. It aggregates
    every stage of the pipeline into a single structured object that
    the adapter layer converts into an API response.
    """

    startup: Startup = Field(..., description="The normalized startup data")
    features: ExtractedFeatures = Field(
        ..., description="Extracted factual features"
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Contextual domain evidence gathered from knowledge base",
    )
    observations: list[Observation] = Field(
        default_factory=list, description="Reasoning layer observations"
    )
    dimension_assessments: list[DimensionAssessment] = Field(
        default_factory=list,
        description="Structured assessments for each analysis dimension",
    )
    scores: list[ScoreResult] = Field(
        default_factory=list, description="Dimension scores from the scoring layer"
    )
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Aggregated overall score",
    )
    recommendations: list[Recommendation] = Field(
        default_factory=list, description="Actionable recommendations"
    )
    confidence: list[ConfidenceAssessment] = Field(
        default_factory=list,
        description="Per-dimension confidence assessments",
    )
    overall_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Aggregated overall confidence level",
    )
    investment_readiness: InvestmentReadiness | None = Field(
        default=None,
        description="Investment readiness assessment (Sprint 8)",
    )
    investment_decision: InvestmentDecision | None = Field(
        default=None,
        description="Deterministic investment decision (Sprint 13)",
    )
    signal_relationships: list[SignalRelationship] = Field(
        default_factory=list,
        description="Cross-signal relationships detected across dimensions",
    )
    analysis_metadata: AnalysisMetadata = Field(
        default_factory=AnalysisMetadata, description="Execution metadata"
    )
