"""Output models for the analysis pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field

from predictron_engine.decision.models import (
    CalibrationSummary,
    DecisionConfidence,
)
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


class EvidenceCitation(BaseModel):
    """Structured citation linking a claim to its source documents.

    Every citation connects a specific evidence claim (domain, category,
    statement) to the web documents that support it, along with trust
    metadata for ranking and filtering.

    Citations are deterministic — the same inputs always produce the
    same citation.  They are pure value objects with no side effects.
    """

    claim: str = Field(
        ..., description="The evidence statement being cited"
    )
    domain: str = Field(
        ..., description="Feature domain of the cited evidence"
    )
    category: str = Field(
        ..., description="Evidence sub-category"
    )
    source_document_ids: list[str] = Field(
        default_factory=list,
        description="UUIDv5 document ids that support this claim",
    )
    source_urls: list[str] = Field(
        default_factory=list,
        description="Source URLs of the supporting documents",
    )
    best_trust_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Highest trust score among cited source documents",
    )
    citation_text: str = Field(
        default="",
        description="Human-readable formatted citation",
    )
    provider: str = Field(
        default="",
        description="Knowledge provider that produced this evidence item",
    )


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

    # Sprint 5A: Provenance linkage
    provenance_record: str | None = Field(
        default=None,
        description=(
            "JSON-encoded ProvenanceRecord linking this evidence item "
            "back to the source document that provided it"
        ),
    )

    # Sprint 5B: Citation linkage
    citations: list[EvidenceCitation] = Field(
        default_factory=list,
        description=(
            "Structured citations linking this evidence item to "
            "supporting source documents with trust metadata"
        ),
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

    # Sprint 5B: Citation linkage
    citations: list[EvidenceCitation] = Field(
        default_factory=list,
        description=(
            "Structured citations linking this observation to "
            "supporting source documents with trust metadata"
        ),
    )

    # Sprint 6A: Evidence-backed observation metadata
    trust_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Deterministic evidence trust score for this observation: "
            "the highest trust score among cited supporting documents"
        ),
    )
    provenance_document_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Evidence document ids supporting this observation, enabling "
            "traceability Observation -> EvidenceItem -> EvidenceCitation "
            "-> EvidenceDocument -> Provider -> Original URL"
        ),
    )
    evidence_agreement_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the supporting evidence items that are "
            "corroborated by multiple independent sources"
        ),
    )
    evidence_conflict_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of conflicting signals detected among the "
            "supporting evidence items"
        ),
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

    # Sprint 5B: Citation linkage
    citations: list[EvidenceCitation] = Field(
        default_factory=list,
        description=(
            "Structured citations linking this recommendation to "
            "supporting source documents with trust metadata"
        ),
    )

    # Sprint 6B: Decision calibration risk (optional, backward compatible)
    expected_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated confidence expected if this recommendation is "
            "followed (Sprint 6B decision calibration)"
        ),
    )
    expected_uncertainty: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated uncertainty associated with this recommendation "
            "(Sprint 6B decision calibration)"
        ),
    )
    recommended_action: str = Field(
        default="",
        description=(
            "Deterministic action guidance derived from calibrated "
            "confidence (Sprint 6B decision calibration)"
        ),
    )

    # Sprint 6C: Deterministic synthesis rank (optional, backward compatible)
    rank: int | None = Field(
        default=None,
        ge=1,
        description=(
            "1-based deterministic priority rank assigned by the decision "
            "synthesis engine (Sprint 6C). None when not synthesized."
        ),
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

    # Sprint 5B: Citation linkage
    citations: list[EvidenceCitation] = Field(
        default_factory=list,
        description=(
            "Structured citations linking this assessment to "
            "supporting source documents with trust metadata"
        ),
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


class EvidenceCollectionMetadata(BaseModel):
    """Aggregate statistics about website evidence collection for one analysis.

    This is an internal analysis artifact today. A later API version may
    expose these fields publicly; no public endpoint consumes them yet.
    """

    website: str | None = Field(
        default=None, description="Website URL that was collected, if any"
    )
    pages_discovered: int = Field(
        default=0, ge=0, description="Number of candidate pages discovered"
    )
    pages_fetched: int = Field(
        default=0, ge=0, description="Number of pages successfully fetched"
    )
    successful_sources: int = Field(
        default=0, ge=0, description="Number of retrieval sources that succeeded"
    )
    failed_sources: int = Field(
        default=0, ge=0, description="Number of retrieval sources that failed"
    )
    collection_time_ms: int = Field(
        default=0, ge=0, description="Wall-clock time spent collecting evidence"
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
    evidence_collection: EvidenceCollectionMetadata | None = Field(
        default=None,
        description="Website evidence collection statistics, when evidence collection ran",
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


# ---------------------------------------------------------------------------
# Sprint 6C: Decision synthesis models.
#
# These models are containers for artifacts synthesized from ALREADY
# produced pipeline outputs. They never recompute scores, confidence,
# trust, evidence, reasoning, or evaluations — they only aggregate,
# prioritize, explain, and connect existing artifacts. Confidence values
# on every synthesis artifact are propagated (typically the minimum of
# the contributing artifacts), never recomputed.
# ---------------------------------------------------------------------------


class SynthesisSeverity(str, Enum):
    """Deterministic severity/impact bands used by the risk and
    opportunity registers.

    Bands mirror the thresholds already used by the scoring layer
    (TractionScorer) and the investment readiness assessment so that no
    new judgment scale is introduced.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


_SEVERITY_ORDER: dict[str, int] = {
    SynthesisSeverity.CRITICAL.value: 0,
    SynthesisSeverity.HIGH.value: 1,
    SynthesisSeverity.MODERATE.value: 2,
    SynthesisSeverity.LOW.value: 3,
}


def severity_order(severity: str | SynthesisSeverity) -> int:
    """Deterministic sort key for a severity value (lower sorts first)."""
    return _SEVERITY_ORDER.get(
        severity.value if isinstance(severity, SynthesisSeverity) else str(severity),
        len(_SEVERITY_ORDER),
    )


class TradeOff(BaseModel):
    """An explicit tension between a strength and a concern.

    Both sides must come from existing evaluator outputs (dimension
    scores and supporting observations). Confidence is the minimum of
    the contributing confidences — propagated, never recomputed.
    """

    dimension: str = Field(..., description="Analysis dimension the trade-off belongs to")
    strength: str = Field(default="", description="The positive side of the tension")
    concern: str = Field(default="", description="The negative side of the tension")
    strength_score: float | None = Field(
        default=None, ge=0.0, le=100.0, description="Dimension score supporting the strength"
    )
    concern_score: float | None = Field(
        default=None, ge=0.0, le=100.0, description="Dimension score driving the concern"
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Statements from existing observations backing this trade-off",
    )
    supporting_citations: list[EvidenceCitation] = Field(
        default_factory=list,
        description="Citations carried over from the supporting observations",
    )
    net_assessment: str = Field(
        default="balanced",
        description=(
            "Deterministic label: 'strength_dominant', 'concern_dominant', "
            "or 'balanced'"
        ),
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence among contributing observations",
    )


class AlternativeScenario(BaseModel):
    """A deterministic what-if scenario derived from decision margins.

    Scenarios are generated from templates only (no generative text).
    Every numeric projection is a bounded arithmetic derivation of an
    already-emitted scalar such as margin_to_next_category or the
    data_quality_modifier recovery headroom.
    """

    direction: str = Field(
        ...,
        description=(
            "Scenario direction: 'improve', 'worsen', or 'information'"
        ),
    )
    title: str = Field(..., description="Deterministic template title")
    condition: str = Field(
        ..., description="What would have to happen for this scenario"
    )
    description: str = Field(
        ..., description="Deterministic template sentence describing the outcome"
    )
    projected_impact: float | None = Field(
        default=None,
        description=(
            "Bounded projected composite-score delta derived from "
            "existing decision scalars, when derivable"
        ),
    )
    plausibility: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Deterministic plausibility in [0, 1] derived from existing "
            "margin, uncertainty, and gap counts"
        ),
    )


class RiskItem(BaseModel):
    """One entry in the unified risk register.

    Aggregated (and deduplicated) from feature risks, consistency
    conflicts, low-confidence assessments, missing evidence, weakening
    calibration factors, and negative signal relationships.
    """

    label: str = Field(..., description="Stable, human-readable risk label")
    dimension: str = Field(
        default="", description="Owning analysis dimension ('' when cross-cutting)"
    )
    severity: SynthesisSeverity = Field(
        default=SynthesisSeverity.LOW,
        description="Deterministic severity band",
    )
    sources: list[str] = Field(
        default_factory=list,
        description=(
            "Traceability tags of contributing artifacts, e.g. "
            "'feature:risk_flags' or 'consistency:conflict'"
        ),
    )
    statements: list[str] = Field(
        default_factory=list,
        description="Verbatim statements carried over from source artifacts",
    )
    citations: list[EvidenceCitation] = Field(
        default_factory=list,
        description="Citations carried over from contributing observations",
    )
    provenance_document_ids: list[str] = Field(
        default_factory=list,
        description="Provenance document ids carried over from contributing observations",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence among contributing artifacts",
    )


class OpportunityItem(BaseModel):
    """One entry in the unified opportunity register.

    Aggregated (and deduplicated) from strengths, high dimension scores,
    reinforcing signals, readiness strengths, positive quantitative
    metrics, and strong evidence.
    """

    label: str = Field(..., description="Stable, human-readable opportunity label")
    dimension: str = Field(
        default="", description="Owning analysis dimension ('' when cross-cutting)"
    )
    impact: SynthesisSeverity = Field(
        default=SynthesisSeverity.LOW,
        description="Deterministic impact band (reuses the severity scale)",
    )
    sources: list[str] = Field(
        default_factory=list,
        description=(
            "Traceability tags of contributing artifacts, e.g. "
            "'score:market_opportunity' or 'relationship:reinforcing'"
        ),
    )
    statements: list[str] = Field(
        default_factory=list,
        description="Verbatim statements carried over from source artifacts",
    )
    citations: list[EvidenceCitation] = Field(
        default_factory=list,
        description="Citations carried over from contributing observations",
    )
    provenance_document_ids: list[str] = Field(
        default_factory=list,
        description="Provenance document ids carried over from contributing observations",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence among contributing artifacts",
    )


class DecisionSynthesis(BaseModel):
    """Single container for every Sprint 6C synthesized artifact.

    Attached to Report as ``decision_synthesis``. All contained data is
    derived deterministically from already-produced pipeline outputs;
    confidence and uncertainty are propagated passthrough values from
    DecisionConfidence / CalibrationSummary, never recomputed.
    """

    executive_summary: str = Field(
        default="",
        description="Deterministic template executive summary paragraph",
    )
    executive_summary_key_points: list[str] = Field(
        default_factory=list,
        description="Deterministic bullet points behind the summary",
    )
    prioritized_recommendations: list[Recommendation] = Field(
        default_factory=list,
        description=(
            "Deduplicated, globally prioritized, capped recommendations "
            "with rank assigned; citations/provenance/confidence preserved"
        ),
    )
    trade_offs: list[TradeOff] = Field(
        default_factory=list,
        description="Explicit strength-vs-concern tensions from evaluator outputs",
    )
    scenarios: list[AlternativeScenario] = Field(
        default_factory=list,
        description="Deterministic improve/worsen/information scenarios",
    )
    risks: list[RiskItem] = Field(
        default_factory=list,
        description="Unified, deduplicated, severity-ranked risk register",
    )
    opportunities: list[OpportunityItem] = Field(
        default_factory=list,
        description="Unified, deduplicated, impact-ranked opportunity register",
    )
    overall_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Passthrough of DecisionConfidence.overall_confidence "
            "(propagated, never recomputed)"
        ),
    )
    uncertainty_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Passthrough of DecisionConfidence.uncertainty_score "
            "(propagated, never recomputed)"
        ),
    )
    confidence_level: str = Field(
        default="",
        description="Passthrough of the calibrated ConfidenceLevel value",
    )
    recommended_action: str = Field(
        default="",
        description="Passthrough of the calibrated action guidance",
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
    decision_confidence: DecisionConfidence | None = Field(
        default=None,
        description=(
            "Calibrated decision confidence and uncertainty (Sprint 6B)"
        ),
    )
    calibration_summary: CalibrationSummary | None = Field(
        default=None,
        description="Compact digest of the decision calibration (Sprint 6B)",
    )
    decision_synthesis: DecisionSynthesis | None = Field(
        default=None,
        description=(
            "Unified decision synthesis: prioritized recommendations, "
            "trade-offs, scenarios, risks, opportunities, and executive "
            "summary (Sprint 6C)"
        ),
    )
    signal_relationships: list[SignalRelationship] = Field(
        default_factory=list,
        description="Cross-signal relationships detected across dimensions",
    )
    analysis_metadata: AnalysisMetadata = Field(
        default_factory=AnalysisMetadata, description="Execution metadata"
    )
