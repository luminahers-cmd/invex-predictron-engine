"""Debug report model — structured output for pipeline debugging and validation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from predictron_engine.validation.explainability import StageExplanation
from predictron_engine.validation.trace import TraceGraph
from predictron_engine.validation.validators.pipeline_validator import (
    ValidationFinding,
)


class PipelineSummary(BaseModel):
    """Summary of the pipeline execution."""

    startup_name: str = Field(
        default="", description="Name of the startup analyzed"
    )
    engine_version: str = Field(
        default="", description="Engine version used"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Total processing time in milliseconds"
    )
    stages_completed: list[str] = Field(
        default_factory=list,
        description="Pipeline stages that completed successfully",
    )
    total_artifacts: int = Field(
        default=0,
        description="Total number of artifacts produced by the pipeline",
    )
    data_completeness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall data completeness",
    )


class StageOutput(BaseModel):
    """Summary of a single pipeline stage's output."""

    stage: str = Field(..., description="Pipeline stage name")
    artifact_type: str = Field(
        ..., description="Type of output produced"
    )
    artifact_count: int = Field(
        default=0, description="Number of artifacts produced"
    )
    explanation: StageExplanation | None = Field(
        default=None, description="Structured explanation of this stage"
    )


class ConfidenceSummary(BaseModel):
    """Summary of confidence across the pipeline."""

    overall_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall pipeline confidence",
    )
    per_dimension: dict[str, float] = Field(
        default_factory=dict,
        description="Confidence per analysis dimension",
    )
    lowest_confidence_dimension: str = Field(
        default="", description="Dimension with lowest confidence"
    )
    highest_confidence_dimension: str = Field(
        default="", description="Dimension with highest confidence"
    )


class MissingInputs(BaseModel):
    """Summary of missing or incomplete inputs."""

    missing_features: list[str] = Field(
        default_factory=list,
        description="Feature fields that could not be determined",
    )
    missing_evidence_domains: list[str] = Field(
        default_factory=list,
        description="Evidence domains with no data",
    )
    low_confidence_dimensions: list[str] = Field(
        default_factory=list,
        description="Dimensions with very low confidence",
    )
    unused_evidence_count: int = Field(
        default=0, description="Number of unused evidence items"
    )
    unused_observation_count: int = Field(
        default=0, description="Number of unused observations"
    )


class DebugReport(BaseModel):
    """Complete debug report for a pipeline execution.

    Aggregates all validation results, traceability information,
    explanations, and diagnostics into a single structured object
    for debugging, auditing, and trust verification.

    Sprint P8D additions (all optional, backward compatible):
      - ``contradiction_graph`` — dict serialization of the reasoning
        contradiction graph when the reasoning layer produced one.
      - ``reasoning_trace`` — dict serialization of the reasoning trace
        when generated (recommendation/debug path only; never changes
        the production Report schema).
      - ``reasoning_budget`` — adaptive-budget diagnostics dict.
    """

    pipeline_summary: PipelineSummary = Field(
        default_factory=PipelineSummary,
        description="Summary of the pipeline execution",
    )
    stage_outputs: list[StageOutput] = Field(
        default_factory=list,
        description="Per-stage output summaries with explanations",
    )
    warnings: list[ValidationFinding] = Field(
        default_factory=list,
        description="All validation warnings and errors",
    )
    validation_results: dict[str, list[ValidationFinding]] = Field(
        default_factory=dict,
        description="Validation results grouped by validator name",
    )
    trace_graph: TraceGraph | None = Field(
        default=None,
        description="Full traceability graph linking all artifacts",
    )
    missing_inputs: MissingInputs = Field(
        default_factory=MissingInputs,
        description="Summary of missing or incomplete inputs",
    )
    confidence_summary: ConfidenceSummary = Field(
        default_factory=ConfidenceSummary,
        description="Confidence summary across the pipeline",
    )
    key_artifacts: dict[str, str] = Field(
        default_factory=dict,
        description="Maps artifact type to its ID in the trace graph",
    )
    contradiction_graph: dict[str, object] | None = Field(
        default=None,
        description=(
            "Serialized reasoning contradiction graph (Sprint P8D)"
        ),
    )
    reasoning_trace: dict[str, object] | None = Field(
        default=None,
        description=(
            "Serialized reasoning trace, when generated (Sprint P8D)"
        ),
    )
    reasoning_budget: dict[str, object] | None = Field(
        default=None,
        description=(
            "Adaptive-budget diagnostics, when computed (Sprint P8D)"
        ),
    )

    @property
    def has_errors(self) -> bool:
        """Check if any error-severity findings exist."""
        return any(f.severity == "error" for f in self.warnings)

    @property
    def warning_count(self) -> int:
        """Count of warning-severity findings."""
        return sum(1 for f in self.warnings if f.severity == "warning")

    @property
    def error_count(self) -> int:
        """Count of error-severity findings."""
        return sum(1 for f in self.warnings if f.severity == "error")

    @property
    def info_count(self) -> int:
        """Count of info-severity findings."""
        return sum(1 for f in self.warnings if f.severity == "info")
