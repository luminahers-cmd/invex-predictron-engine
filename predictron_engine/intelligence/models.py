"""Sprint 9 — Adaptive Intelligence models.

These are *aggregate / analytic value objects* that describe the engine's
own behaviour over time.  They are distinct from the domain models in
``predictron_engine.models`` (which describe a single startup's analysis).

Every value is computed deterministically from already-produced pipeline
outputs — a ``Report``, an ``EvidenceBundle``, or a collection of both.
No functionality is duplicated: this module only *records* analytics and
never recomputes scores, confidence, trust, evidence, or decisions.

Design rules
------------
* Pure pydantic value objects — no side effects, no I/O.
* Identical inputs always produce identical outputs.
* Collections are passed in explicitly (no global state, no hidden caches).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DriftDirection(str, Enum):
    """Deterministic direction of change between two observations."""

    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    REGRESSED = "regressed"


class TrendDirection(str, Enum):
    """Deterministic slope of a metric over a history."""

    UP = "up"
    FLAT = "flat"
    DOWN = "down"


class RuleEffectiveness(BaseModel):
    """Measured effectiveness of a single reasoning rule.

    ``source_rule`` identifies the rule on each :class:`Observation`.
    ``hit_count`` is the number of times the rule produced an observation
    across the analysed reports.  ``importance`` and ``confidence`` are
    averages of the observation fields.  ``contribution`` is the rule's
    share of all observations analysed (0..1).
    """

    source_rule: str = Field(..., description="Name of the reasoning rule")
    hit_count: int = Field(default=0, ge=0, description="Number of observations produced")
    observed_dimensions: list[str] = Field(
        default_factory=list,
        description="Distinct dimensions the rule produced observations for",
    )
    avg_importance: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Mean observation importance"
    )
    avg_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Mean observation confidence"
    )
    contribution: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Share of all observations attributed to this rule",
    )


class RecommendationEffectiveness(BaseModel):
    """Effectiveness metrics for the recommendation layer.

    ``priority_coverage`` is the fraction of recommendations that carry a
    non-empty priority.  ``action_coverage`` is the fraction with a
    non-empty action.  ``confidence_calibration`` measures how strongly
    recommendation confidence tracks the overall calibrated confidence
    (a signed value in [-1, 1], higher is better calibrated).  These are
    *analytics over existing* :class:`Recommendation` objects — no new
    decision logic is introduced.
    """

    total_recommendations: int = Field(
        default=0, ge=0, description="Recommendations analysed"
    )
    categories: list[str] = Field(
        default_factory=list,
        description="Distinct recommendation categories, sorted",
    )
    priority_coverage: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of recommendations with a non-empty priority",
    )
    action_coverage: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of recommendations with a non-empty action",
    )
    avg_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Mean recommendation confidence",
    )
    confidence_calibration: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description=(
            "Pearson-style alignment of recommendation confidence with "
            "overall confidence (signed; higher = better)"
        ),
    )


class CalibrationDrift(BaseModel):
    """Drift of calibrated decision confidence between two runs.

    ``direction`` uses :class:`DriftDirection`.  ``delta`` is the signed
    arithmetic difference (new - old).  ``overconfident`` records whether
    the calibration monitoring (Sprint 8) flagged overconfidence in the
    new run.
    """

    from_confidence: float = Field(..., ge=0.0, le=1.0)
    to_confidence: float = Field(..., ge=0.0, le=1.0)
    delta: float = Field(..., description="Signed delta from -> to")
    direction: DriftDirection = Field(..., description="Deterministic direction")
    overconfident: bool = Field(
        default=False, description="Overconfidence flagged in the new run"
    )


class ConfidenceDrift(BaseModel):
    """Confidence drift analysis across two collections of reports.

    ``decision_drift`` describes calibrated decision confidence change
    (population mean).  ``assessment_drifts`` is a dict keyed by dimension
    describing per-dimension assessed-confidence drift.  ``drifted_dimensions``
    lists dimensions whose change exceeded a caller-defined threshold.
    """

    decision_drift: CalibrationDrift = Field(
        ..., description="Overall calibrated decision confidence drift"
    )
    assessment_drifts: dict[str, CalibrationDrift] = Field(
        default_factory=dict,
        description="Per-dimension assessed confidence drift",
    )
    sample_count_a: int = Field(default=0, ge=0, description="Reports in baseline")
    sample_count_b: int = Field(default=0, ge=0, description="Reports in comparison")


class PerformanceSnapshot(BaseModel):
    """A single point-in-time record of engine performance for one report.

    Encodes the *analytic projection* of a :class:`Report` — never the raw
    report (which would duplicate it).  ``version`` is the source engine
    version label supplied by the caller (defaults to the engine version).
    """

    version: str = Field(..., description="Engine version label for this record")
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    data_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    decision_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty_score: float = Field(default=0.0, ge=0.0, le=1.0)
    observation_count: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    recommendation_count: int = Field(default=0, ge=0)
    assessment_count: int = Field(default=0, ge=0)
    dimension_scores: dict[str, float] = Field(
        default_factory=dict, description="Per-dimension overall scores"
    )
    assessed_confidences: dict[str, float] = Field(
        default_factory=dict, description="Per-dimension assessed confidence"
    )
    rule_hits: dict[str, int] = Field(
        default_factory=dict,
        description="Count of observations per source_rule",
    )

    def to_dict(self) -> dict:
        """Deterministic JSON-friendly serialization."""
        return {
            "version": self.version,
            "overall_score": self.overall_score,
            "overall_confidence": self.overall_confidence,
            "data_completeness": self.data_completeness,
            "decision_confidence": self.decision_confidence,
            "uncertainty_score": self.uncertainty_score,
            "observation_count": self.observation_count,
            "evidence_count": self.evidence_count,
            "recommendation_count": self.recommendation_count,
            "assessment_count": self.assessment_count,
            "dimension_scores": dict(self.dimension_scores),
            "assessed_confidences": dict(self.assessed_confidences),
            "rule_hits": dict(self.rule_hits),
        }


class PerformanceTrend(BaseModel):
    """Aggregate of a collection of :class:`PerformanceSnapshot` records.

    ``records`` are the individual snapshots (ordered by ``version`` for
    deterministic chronological trends).  ``metric_trends`` summarises each
    tracked metric using :class:`TrendDirection`.  ``mean_*`` fields are
    population means for convenient dashboard output.
    """

    records: list[PerformanceSnapshot] = Field(default_factory=list)
    metric_trends: dict[str, TrendDirection] = Field(
        default_factory=dict, description="Per-metric trend direction"
    )
    mean_overall_score: float = Field(default=0.0)
    mean_overall_confidence: float = Field(default=0.0)
    mean_decision_confidence: float = Field(default=0.0)
    mean_data_completeness: float = Field(default=0.0)
    record_count: int = Field(default=0, ge=0)


class RegressionFinding(BaseModel):
    """A single detected regression between two snapshots.

    Mirrors the semantics of the existing benchmark regression diff but is
    computed directly from :class:`PerformanceSnapshot` records for the
    intelligence layer (genuinely additive).
    """

    metric: str = Field(..., description="Name of the regressed metric")
    from_value: float = Field(..., description="Baseline value")
    to_value: float = Field(..., description="Comparison value")
    delta: float = Field(..., description="to_value - from_value")


class EngineQualityMetrics(BaseModel):
    """Consolidated engine-quality analytics for a collection of reports.

    ``calibration_error`` reuses the Sprint 8 Expected Calibration Error
    computation.  ``explainability_coverage``, ``recommendation_quality``
    and ``data_completeness`` are lightweight deterministic summaries
    computed here (they do not re-run :class:`BenchmarkMetrics`).
    ``rule_effectiveness`` aggregates per-rule findings.  ``drift`` holds
    the :class:`ConfidenceDrift` when two collections were provided.
    """

    report_count: int = Field(default=0, ge=0)
    calibration_error: float = Field(default=0.0, ge=0.0, le=1.0)
    calibration_quality: str = Field(
        default="insufficient_data", description="Sprint 8 calibration band"
    )
    overconfidence_detected: bool = Field(default=False)
    explainability_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    recommendation_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    data_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    rule_effectiveness: list[RuleEffectiveness] = Field(default_factory=list)
    drift: ConfidenceDrift | None = Field(default=None)


class IntelligenceDashboard(BaseModel):
    """Single container for every Sprint 9 analytic artifact.

    The code-model "performance dashboard" (no UI).  Combines trend,
    regression, rule, recommendation, drift, quality, and calibration
    analytics into one serializable object.
    """

    version: str = Field(..., description="Engine version label")
    trend: PerformanceTrend = Field(
        default_factory=PerformanceTrend, description="Historical trend"
    )
    regressions: list[RegressionFinding] = Field(
        default_factory=list, description="Detected regressions"
    )
    rule_effectiveness: list[RuleEffectiveness] = Field(default_factory=list)
    recommendation_effectiveness: RecommendationEffectiveness = Field(
        default_factory=RecommendationEffectiveness
    )
    confidence_drift: ConfidenceDrift | None = Field(default=None)
    quality: EngineQualityMetrics = Field(default_factory=EngineQualityMetrics)

    def to_dict(self) -> dict:
        """Deterministic JSON-friendly serialization of the whole dashboard."""
        return {
            "version": self.version,
            "trend": self.trend.model_dump(),
            "regressions": [r.model_dump() for r in self.regressions],
            "rule_effectiveness": [r.model_dump() for r in self.rule_effectiveness],
            "recommendation_effectiveness": (
                self.recommendation_effectiveness.model_dump()
            ),
            "confidence_drift": (
                self.confidence_drift.model_dump()
                if self.confidence_drift
                else None
            ),
            "quality": self.quality.model_dump(),
        }
