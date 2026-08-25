"""Decision Calibration models — Sprint 6B.

Minimal, deterministic value objects describing how much a decision can
be trusted.  These models do NOT re-model evidence, trust, provenance,
or citations — they only *combine existing pipeline outputs* into a
calibrated view:

* evidence trust comes from ``TrustScore`` / ``TrustSummary`` (Sprint 5A)
* extraction/evidence confidence comes from ``IntelligenceSummary``
* reasoning confidence comes from ``Observation.confidence``
* evaluator agreement comes from ``DimensionAssessment.confidence``
* feature completeness comes from ``ExtractedFeatures.data_completeness``

Every model is a plain pydantic value object with no behaviour beyond
pure derivation helpers.  Identical inputs always produce identical
outputs.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Calibrated confidence band for an investment decision.

    Deterministic thresholds map a confidence value in [0, 1] to one of
    five ordered bands (see calibration._CONFIDENCE_THRESHOLDS).
    """

    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class ConfidenceFactor(BaseModel):
    """One named component of a confidence or uncertainty computation.

    A single value object shared by both breakdowns so the module does
    not grow parallel factor models.  ``contribution`` is the factor's
    signed impact on the composite (value * weight).
    """

    name: str = Field(..., description="Stable factor identifier")
    value: float = Field(..., ge=0.0, le=1.0, description="Factor value in [0, 1]")
    weight: float = Field(..., ge=0.0, le=1.0, description="Deterministic weight")
    contribution: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Signed contribution to the composite (value * weight)",
    )


class ConfidenceBreakdown(BaseModel):
    """Decomposition of decision confidence into its components.

    Every component is derived from an *existing* pipeline output; this
    model only records the values, their deterministic weights, and the
    resulting weighted composite.
    """

    factors: list[ConfidenceFactor] = Field(
        default_factory=list,
        description="Component factors in deterministic order",
    )
    composite: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted combination of all factor contributions",
    )

    def value_of(self, name: str) -> float | None:
        """Return the raw value of a named factor, or None if absent."""
        for factor in self.factors:
            if factor.name == name:
                return factor.value
        return None


class UncertaintyBreakdown(BaseModel):
    """Decomposition of decision uncertainty into its drivers.

    Each driver is expressed so that higher means *more uncertain*
    (e.g. ``missing_evidence=0.8`` means evidence is largely missing).
    """

    factors: list[ConfidenceFactor] = Field(
        default_factory=list,
        description="Uncertainty drivers in deterministic order",
    )
    composite: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted combination of all driver contributions",
    )

    def value_of(self, name: str) -> float | None:
        """Return the raw value of a named driver, or None if absent."""
        for factor in self.factors:
            if factor.name == name:
                return factor.value
        return None


class DecisionConfidence(BaseModel):
    """Calibrated confidence and uncertainty for one investment decision.

    Produced by :func:`predictron_engine.decision.calibration.compute_decision_confidence`.
    Fully deterministic — identical pipeline outputs produce identical
    calibration.
    """

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall decision confidence in [0, 1]",
    )
    level: ConfidenceLevel = Field(
        ...,
        description="Deterministic confidence band",
    )
    uncertainty_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Decision uncertainty in [0, 1]; "
            "0.0 = fully certain, 1.0 = highly uncertain"
        ),
    )
    breakdown: ConfidenceBreakdown = Field(
        ...,
        description="Per-component confidence decomposition",
    )
    uncertainty_breakdown: UncertaintyBreakdown = Field(
        ...,
        description="Per-driver uncertainty decomposition",
    )
    supporting_factors: list[str] = Field(
        default_factory=list,
        description="Deterministic statements explaining why confidence is high",
    )
    weakening_factors: list[str] = Field(
        default_factory=list,
        description="Deterministic statements explaining why confidence is low",
    )

    @property
    def recommended_action(self) -> str:
        """Deterministic action guidance for this confidence level."""
        # Imported lazily to keep models dependency-light and acyclic.
        from predictron_engine.decision.calibration import (
            action_for_confidence_level,
        )

        return action_for_confidence_level(self.level)


class CalibrationSummary(BaseModel):
    """Compact report-facing digest of the decision calibration.

    Summarizes a :class:`DecisionConfidence` together with the volume
    of pipeline signals behind it, so reports can expose calibration
    without serializing full breakdowns.
    """

    confidence: float = Field(..., ge=0.0, le=1.0)
    level: ConfidenceLevel
    uncertainty_score: float = Field(..., ge=0.0, le=1.0)
    recommended_action: str = Field(
        ..., description="Deterministic action guidance for this level"
    )
    strong_factor_count: int = Field(
        default=0, ge=0, description="Number of supporting statements"
    )
    weak_factor_count: int = Field(
        default=0, ge=0, description="Number of weakening statements"
    )
    top_supporting_factors: list[str] = Field(
        default_factory=list,
        description="First few supporting statements (deterministic order)",
    )
    top_weakening_factors: list[str] = Field(
        default_factory=list,
        description="First few weakening statements (deterministic order)",
    )
    observation_count: int = Field(
        default=0, ge=0, description="Reasoning observations used"
    )
    assessed_dimension_count: int = Field(
        default=0, ge=0, description="Evaluator dimensions used"
    )
    evidence_document_count: int = Field(
        default=0, ge=0, description="Evidence documents used"
    )
