"""Decision Intelligence models — transparent, deterministic investment reasoning.

Value objects for the Decision Intelligence layer. Every model is a plain
pydantic value object with no behaviour beyond pure derivation helpers.

Key principles:
  - Immutable after creation
  - Deterministic serialization
  - Full evidence provenance on every contribution
  - No ML, no LLMs, no embeddings, no probabilistic black boxes
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ContributionType(str, Enum):
    """Classification of how a feature contributes to a decision."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    CONFIDENCE = "confidence"


class DecisionVerdict(str, Enum):
    """Deterministic investment verdict."""

    STRONG_INVEST = "strong_invest"
    INVEST = "invest"
    WATCH = "watch"
    INVESTIGATE_FURTHER = "investigate_further"
    PASS = "pass"


class TraceNodeType(str, Enum):
    """Types of nodes in a decision reasoning graph."""

    RAW_FEATURE = "raw_feature"
    NORMALIZED_FEATURE = "normalized_feature"
    RULE = "rule"
    INTERMEDIATE_SCORE = "intermediate_score"
    DIMENSION_SCORE = "dimension_score"
    OVERALL_SCORE = "overall_score"
    DECISION = "decision"


class Contribution(BaseModel):
    """A single feature contribution to an investment decision.

    Every contribution traces back to a Feature Store snapshot and
    preserves full evidence provenance.
    """

    feature_id: str = Field(..., description="Feature definition ID")
    feature_name: str = Field(..., description="Human-readable feature name")
    category: str = Field(..., description="Feature category")
    contribution_type: ContributionType = Field(
        ..., description="How this feature contributes"
    )
    raw_value: Any = Field(
        default=None, description="Original value from Feature Store"
    )
    normalized_value: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Normalized contribution value in [-1, 1]",
    )
    weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0, description="Weight of this feature in the decision"
    )
    computed_contribution: float = Field(
        default=0.0,
        description="Weighted contribution (normalized_value * weight)",
    )
    evidence_references: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Evidence references from Feature Store snapshot",
    )
    human_explanation: str = Field(
        default="", description="Human-readable explanation of this contribution"
    )
    supporting_evidence: str = Field(
        default="", description="Summary of supporting evidence"
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="How this contribution was computed (deterministic trace)",
    )

    def to_dict(self) -> dict[str, Any]:
        """Deterministic serialization."""
        return {
            "feature_id": self.feature_id,
            "feature_name": self.feature_name,
            "category": self.category,
            "contribution_type": self.contribution_type.value,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "weight": self.weight,
            "computed_contribution": self.computed_contribution,
            "evidence_references": self.evidence_references,
            "human_explanation": self.human_explanation,
            "supporting_evidence": self.supporting_evidence,
            "provenance": self.provenance,
        }


class TraceNode(BaseModel):
    """A single node in a decision reasoning graph.

    Forms part of a directed acyclic graph that traces the complete
    reasoning path from raw feature to final decision.
    """

    node_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique node identifier",
    )
    node_type: TraceNodeType = Field(
        ..., description="Type of reasoning node"
    )
    feature_id: str | None = Field(
        default=None, description="Associated feature ID (if applicable)"
    )
    label: str = Field(..., description="Human-readable node label")
    value: Any = Field(default=None, description="Node value")
    value_type: str = Field(
        default="float", description="Type of the value"
    )
    rule: str | None = Field(
        default=None, description="Rule applied at this node"
    )
    input_node_ids: list[str] = Field(
        default_factory=list, description="Parent node IDs"
    )
    output_node_ids: list[str] = Field(
        default_factory=list, description="Child node IDs"
    )
    evidence_references: list[dict[str, Any]] = Field(
        default_factory=list, description="Supporting evidence"
    )
    computation_metadata: dict[str, Any] = Field(
        default_factory=dict, description="How this node was computed"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of node creation",
    )

    def to_dict(self) -> dict[str, Any]:
        """Deterministic serialization."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "feature_id": self.feature_id,
            "label": self.label,
            "value": self.value,
            "value_type": self.value_type,
            "rule": self.rule,
            "input_node_ids": self.input_node_ids,
            "output_node_ids": self.output_node_ids,
            "evidence_references": self.evidence_references,
            "computation_metadata": self.computation_metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class DecisionTrace(BaseModel):
    """Complete deterministic reasoning graph for a decision.

    Contains the full trace from raw features through normalization,
    rules, intermediate scores, dimension scores, overall score,
    and final decision.
    """

    trace_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique trace identifier",
    )
    company_id: str = Field(..., description="Company this trace is about")
    nodes: list[TraceNode] = Field(
        default_factory=list,
        description="All nodes in the reasoning graph (deterministic order)",
    )
    edges: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Directed edges as (source_id, target_id) pairs",
    )
    overall_score: float = Field(
        default=0.0, description="Final composite score"
    )
    verdict: DecisionVerdict = Field(
        default=DecisionVerdict.PASS,
        description="Final investment verdict",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall decision confidence",
    )
    computation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of trace computation",
    )
    engine_version: str = Field(
        default="0.13.0", description="Engine version that produced this trace"
    )

    def get_nodes_by_type(self, node_type: TraceNodeType) -> list[TraceNode]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes if n.node_type == node_type]

    def get_node_by_id(self, node_id: str) -> TraceNode | None:
        """Get a node by its ID."""
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def to_dict(self) -> dict[str, Any]:
        """Deterministic serialization."""
        return {
            "trace_id": self.trace_id,
            "company_id": self.company_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": self.edges,
            "overall_score": self.overall_score,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "computation_timestamp": self.computation_timestamp.isoformat(),
            "engine_version": self.engine_version,
        }


class Explanation(BaseModel):
    """Human-readable explanation of an investment decision.

    Contains structured prose that references actual evidence and
    traces back to specific features and rules.
    """

    company_id: str = Field(..., description="Company this explanation is about")
    headline: str = Field(
        default="", description="One-line summary of the decision"
    )
    strengths: list[str] = Field(
        default_factory=list, description="Top positive factors"
    )
    weaknesses: list[str] = Field(
        default_factory=list, description="Top negative factors"
    )
    neutral_factors: list[str] = Field(
        default_factory=list, description="Neutral or informational factors"
    )
    confidence_factors: list[str] = Field(
        default_factory=list, description="Factors affecting confidence"
    )
    evidence_summary: str = Field(
        default="", description="Summary of evidence quality and provenance"
    )
    full_explanation: str = Field(
        default="", description="Complete prose explanation"
    )
    recommendation: str = Field(
        default="", description="Actionable recommendation"
    )
    supporting_contributions: list[Contribution] = Field(
        default_factory=list,
        description="Contributions supporting this explanation",
    )
    evidence_references: list[dict[str, Any]] = Field(
        default_factory=list,
        description="All evidence references used in this explanation",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional explanation metadata"
    )

    def to_dict(self) -> dict[str, Any]:
        """Deterministic serialization."""
        return {
            "company_id": self.company_id,
            "headline": self.headline,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "neutral_factors": self.neutral_factors,
            "confidence_factors": self.confidence_factors,
            "evidence_summary": self.evidence_summary,
            "full_explanation": self.full_explanation,
            "recommendation": self.recommendation,
            "supporting_contributions": [
                c.to_dict() for c in self.supporting_contributions
            ],
            "evidence_references": self.evidence_references,
            "metadata": self.metadata,
        }


class CalibrationAdjustment(BaseModel):
    """Result of calibration adjustment against benchmark data.

    Consumes Benchmark Platform outputs to adjust confidence calibration
    without modifying feature values.
    """

    benchmark_case_id: str = Field(
        default="", description="Benchmark case used for calibration"
    )
    expected_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Expected confidence from benchmark",
    )
    actual_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Actual confidence computed",
    )
    calibration_delta: float = Field(
        default=0.0,
        description="Difference between expected and actual confidence",
    )
    historical_calibration: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Historical calibration data points",
    )
    adjustment_applied: float = Field(
        default=0.0,
        description="Confidence adjustment applied (never modifies features)",
    )
    adjustment_rationale: str = Field(
        default="", description="Why this adjustment was made"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of calibration",
    )

    def to_dict(self) -> dict[str, Any]:
        """Deterministic serialization."""
        return {
            "benchmark_case_id": self.benchmark_case_id,
            "expected_confidence": self.expected_confidence,
            "actual_confidence": self.actual_confidence,
            "calibration_delta": self.calibration_delta,
            "historical_calibration": self.historical_calibration,
            "adjustment_applied": self.adjustment_applied,
            "adjustment_rationale": self.adjustment_rationale,
            "timestamp": self.timestamp.isoformat(),
        }


class DecisionIntelligenceReport(BaseModel):
    """Complete deterministic decision intelligence report.

    Aggregates contributions, trace, explanation, calibration,
    and evidence into a single report for a company.
    """

    report_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique report identifier",
    )
    company_id: str = Field(..., description="Company this report is about")
    report_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of report generation",
    )
    engine_version: str = Field(
        default="0.13.0", description="Engine version"
    )
    decision_summary: dict[str, Any] = Field(
        default_factory=dict, description="Summary of the decision"
    )
    contributions: list[Contribution] = Field(
        default_factory=list,
        description="All feature contributions (deterministic order)",
    )
    positive_contributions: list[Contribution] = Field(
        default_factory=list, description="Positive contributors"
    )
    negative_contributions: list[Contribution] = Field(
        default_factory=list, description="Negative contributors"
    )
    neutral_contributions: list[Contribution] = Field(
        default_factory=list, description="Neutral contributors"
    )
    confidence_contributions: list[Contribution] = Field(
        default_factory=list, description="Confidence contributors"
    )
    trace: DecisionTrace | None = Field(
        default=None, description="Complete decision reasoning trace"
    )
    explanation: Explanation | None = Field(
        default=None, description="Human-readable explanation"
    )
    calibration: CalibrationAdjustment | None = Field(
        default=None, description="Calibration adjustment data"
    )
    feature_provenance: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Feature provenance chain for all used features",
    )
    evidence_references: list[dict[str, Any]] = Field(
        default_factory=list,
        description="All evidence references in this report",
    )
    top_strengths: list[str] = Field(
        default_factory=list, description="Top investment strengths"
    )
    top_weaknesses: list[str] = Field(
        default_factory=list, description="Top investment weaknesses"
    )
    historical_comparisons: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Historical comparison data",
    )
    confidence_explanation: str = Field(
        default="", description="Why this confidence level"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional report metadata"
    )

    def to_dict(self) -> dict[str, Any]:
        """Deterministic serialization."""
        return {
            "report_id": self.report_id,
            "company_id": self.company_id,
            "report_timestamp": self.report_timestamp.isoformat(),
            "engine_version": self.engine_version,
            "decision_summary": self.decision_summary,
            "contributions": [c.to_dict() for c in self.contributions],
            "positive_contributions": [
                c.to_dict() for c in self.positive_contributions
            ],
            "negative_contributions": [
                c.to_dict() for c in self.negative_contributions
            ],
            "neutral_contributions": [
                c.to_dict() for c in self.neutral_contributions
            ],
            "confidence_contributions": [
                c.to_dict() for c in self.confidence_contributions
            ],
            "trace": self.trace.to_dict() if self.trace else None,
            "explanation": self.explanation.to_dict() if self.explanation else None,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "feature_provenance": self.feature_provenance,
            "evidence_references": self.evidence_references,
            "top_strengths": self.top_strengths,
            "top_weaknesses": self.top_weaknesses,
            "historical_comparisons": self.historical_comparisons,
            "confidence_explanation": self.confidence_explanation,
            "metadata": self.metadata,
        }
