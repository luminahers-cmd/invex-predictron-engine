"""Derived metrics data models — traceability and audit structures.

Every derived metric produced by the inference engine is recorded with
full provenance: which source fields were used, what formula was applied,
and what confidence the derivation carries. This ensures every computed
value is explainable, auditable, and reproducible.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DerivedMetricLog(BaseModel):
    """Audit record for a single derived metric computation.

    Captures the complete provenance of a derived value:
    what inputs were used, what formula produced it,
    and what confidence the derivation carries.

    Attributes:
        metric_name: Name of the derived metric (e.g. "arr_from_mrr").
        derived_value: The computed numeric value.
        source_fields: ExtractedFeatures field names used as inputs.
        source_values: Actual input values used in the computation.
        formula: Deterministic formula description (e.g. "MRR * 12").
        explanation: Human-readable explanation of the derivation.
        confidence: Confidence in the derivation (0.0-1.0).
    """

    metric_name: str = Field(
        description="Identifier for the derived metric"
    )
    derived_value: float = Field(
        description="The computed numeric value"
    )
    source_fields: list[str] = Field(
        description="ExtractedFeatures field names used as inputs"
    )
    source_values: list[float | int] = Field(
        description="Actual input values used in the computation"
    )
    formula: str = Field(
        description="Deterministic formula description"
    )
    explanation: str = Field(
        description="Human-readable explanation of the derivation"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the derivation (0.0-1.0)"
    )
