"""Phase 7 — Continuous Learning Intelligence: value objects.

Deterministic, pure pydantic models that describe the *learning surface* of
the platform: resolved evaluation samples, per-dimension intelligence,
canonical observations, patterns, and rule-based recommendations.  No
computation happens here — every value is produced by the deterministic
builder in :mod:`predictron_engine.learning.engine`.

Mathematical definitions (documented per requirement — no hidden heuristics)
----------------------------------------------------------------------------
* **Accuracy** ``(tp+tn)/(tp+tn+fp+fn)`` over binary-scoreable samples.
  For a cohort of evaluated predictions this equals the canonical
  ``TN`` / ``FN`` confusion built from the binary labels of
  :func:`predictron_engine.dataset.metrics.binary_label`.
* **Population accuracy** — the same accuracy over the whole evaluated
  population (the learning baseline every dimension observation compares
  against).
* **Confidence bias** — ``mean(predicted confidence) - rate(actual positive
  outcomes)`` for a cohort.  Positive means the model over-predicts
  confidence relative to observed success frequency; negative means it
  under-predicts.
* **False positive / false negative rate** — ``FP/(FP+TN)`` /
  ``FN/(FN+TP)``.  ``None`` when the underlying support is zero.
* **Expected calibration error (ECE)** — equal-width binned absolute
  confidence-vs-accuracy disagreement over samples with a
  binary-outcome label, weighted by bin support (see
  :func:`predictron_engine.learning.engine.compute_calibration`).

Thresholds (all at module level — nothing hidden)
-------------------------------------------------
Every threshold is a named constant in this module.  The engine and the
recommendation rules reference these constants exclusively.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from predictron_engine.dataset.evaluation import EvaluationVerdict

# ---------------------------------------------------------------------------
# Thresholds (named configuration constants)
# ---------------------------------------------------------------------------

#: Dimensions of categorical intelligence the platform can reason about.
LEARNING_DIMENSIONS: tuple[str, ...] = (
    "sector",
    "stage",
    "country",
    "technology",
    "business_model",
    "founder",
)

#: Canonical value used when attribute resolution cannot determine a value.
ATTR_UNKNOWN = "unknown"

#: Version of the on-disk/on-line snapshots the learning layer produces.
LEARNING_SNAPSHOT_SCHEMA_VERSION = "1.0"

#: Minimum number of binary-scoreable samples before a dimension bucket is
#: eligible for an accuracy/confidence observation.
OBSERVATION_MIN_SAMPLES = 5

#: Minimum absolute accuracy difference (on 0..1) before an observation is
#: emitted about a dimension bucket being above/below platform accuracy.
OBSERVATION_ACCURACY_DELTA = 0.10

#: Absolute confidence-bias (on 0..1) above which over/under-confidence is
#: reported for a dimension bucket.
OBSERVATION_BIAS_THRESHOLD = 0.05

#: Minimum sample count before a per-dimension bucket recommendation states
#: something stronger than "await more evidence".
RECOMMENDATION_MIN_SAMPLES = 10

#: Accuracy delta used by the rule engine when it suggests adjusting the
#: confidence weighting of a dimension bucket.
RECOMMENDATION_ACCURACY_DELTA = 0.10

#: Number of equal-width bins used for ECE.
CALIBRATION_BIN_COUNT = 5

#: Upper ECE bound before the rule engine requests a confidence recalibration.
RECOMMENDATION_MAX_ECE = 0.10

#: Bias bound used by the rule engine for a global confidence calibration.
RECOMMENDATION_MAX_CONFIDENCE_BIAS = 0.05

#: Standard deviation of predicted confidence below which the confidence
#: distribution is considered "stable"/"concentrated".
CONFIDENCE_STABLE_STD = 0.15


class LearningDimension(str, Enum):
    """Categorical intelligence dimensions tracked by the learning layer."""

    SECTOR = "sector"
    STAGE = "stage"
    COUNTRY = "country"
    TECHNOLOGY = "technology"
    BUSINESS_MODEL = "business_model"
    FOUNDER = "founder"


class LearningPeriodKind(str, Enum):
    """Period cadence of an append-only learning snapshot."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class LearningObservationCategory(str, Enum):
    """Canonical observation categories the engine can emit."""

    ACCURACY_EXCESS = "accuracy_excess"
    ACCURACY_DEFICIT = "accuracy_deficit"
    OVERCONFIDENCE = "overconfidence"
    UNDERCONFIDENCE = "underconfidence"
    LOW_SUPPORT = "low_support"
    STRONG_ACCURACY = "strong_accuracy"
    HIGH_FALSE_POSITIVE_RATE = "high_false_positive_rate"
    HIGH_FALSE_NEGATIVE_RATE = "high_false_negative_rate"


class TrendDirection(str, Enum):
    """Endpoint-slope direction reused by observations."""

    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class LearningPeriod(BaseModel):
    """A paired value-bucket within one intelligence dimension."""

    dimension: LearningDimension = Field(...)
    value: str = Field(..., description="Resolved dimension value")
    sample_size: int = Field(default=0, ge=0)
    accuracy: float | None = Field(default=None)
    confidence_bias: float | None = Field(default=None)
    false_positive_rate: float | None = Field(default=None)
    false_negative_rate: float | None = Field(default=None)
    recommendation: str = Field(
        default="", description="Deterministic rule-engine guidance"
    )


class LearningSample(BaseModel):
    """One evaluated prediction resolved to all learning dimensions.

    The categorical attributes (:attr:`resolved`) are the only inputs the
    aggregation engine needs; the app layer resolves them deterministically
    from the stored report features and frozen prediction.  Every numeric
    value here is stored as-is from the evaluation row.
    """

    evaluation_id: str = Field(..., description="Deterministic evaluation id")
    confidence: float = Field(..., ge=0.0, le=1.0)
    verdict: EvaluationVerdict = Field(...)
    actual_positive: bool | None = Field(default=None)
    resolved: dict[str, str] = Field(default_factory=dict)


class LearningPattern(BaseModel):
    """Cohort aggregation of samples for one ``(dimension, value)``."""

    dimension: LearningDimension = Field(...)
    value: str = Field(...)
    samples: int = Field(default=0, ge=0)
    scoreable: int = Field(default=0, ge=0)
    true_positive: int = Field(default=0, ge=0)
    true_negative: int = Field(default=0, ge=0)
    false_positive: int = Field(default=0, ge=0)
    false_negative: int = Field(default=0, ge=0)
    accuracy: float | None = Field(default=None)
    precision: float | None = Field(default=None)
    false_positive_rate: float | None = Field(default=None)
    false_negative_rate: float | None = Field(default=None)
    confidence: float = Field(default=0.0, description="Mean predicted confidence")
    confidence_bias: float | None = Field(default=None)
    recommendation: str = Field(
        default="", description="Rule-engine guidance for this bucket"
    )


class LearningObservation(BaseModel):
    """One canonical, deterministic observation about the population."""

    category: LearningObservationCategory = Field(...)
    dimension: LearningDimension = Field(...)
    value: str = Field(...)
    metric: str = Field(..., description="Metric key being observed")
    metric_value: float | None = Field(default=None)
    delta: float | None = Field(default=None)
    direction: TrendDirection = Field(default=TrendDirection.FLAT)
    baseline: float | None = Field(default=None)
    sample_size: int = Field(default=0, ge=0)
    summary: str = Field(default="", description="Canned deterministic statement")


class LearningRecommendation(BaseModel):
    """Canned, rule-based recommendation at the snapshot level."""

    kind: str = Field(..., description="Stable machine-readable rule key")
    message: str = Field(..., description="Deterministic human-readable guidance")
    severity: str = Field(default="info", description="info | warning | critical")
    parameters: dict[str, object] = Field(default_factory=dict)


class LearningConfidence(BaseModel):
    """Population confidence statistics for the learning snapshot."""

    count: int = Field(default=0, ge=0)
    mean: float | None = Field(default=None)
    std_dev: float | None = Field(default=None)
    bias: float = Field(default=0.0)
    calibrated: bool = Field(default=True)


class LearningCalibration(BaseModel):
    """Deterministic calibration digest computed over learned samples."""

    expected_calibration_error: float = Field(default=0.0)
    overconfidence_detected: bool = Field(default=False)
    overconfident_bins: int = Field(default=0, ge=0)
    total_samples: int = Field(default=0, ge=0)
    bins: list[dict[str, object]] = Field(default_factory=list)


class LearningDigest(BaseModel):
    """Flat aggregate metrics and confusion counts for the snapshot."""

    evaluation_count: int = Field(default=0, ge=0)
    sample_count: int = Field(default=0, ge=0)
    scoreable: int = Field(default=0, ge=0)
    accuracy: float | None = Field(default=None)
    precision: float | None = Field(default=None)
    recall: float | None = Field(default=None)
    false_positive_rate: float | None = Field(default=None)
    false_negative_rate: float | None = Field(default=None)
    true_positive: int = Field(default=0, ge=0)
    true_negative: int = Field(default=0, ge=0)
    false_positive: int = Field(default=0, ge=0)
    false_negative: int = Field(default=0, ge=0)
    verdict_counts: dict[str, int] = Field(default_factory=dict)


class LearningSnapshot(BaseModel):
    """One point-in-time, append-only snapshot of the learning layer.

    ``scope`` / ``period_kind`` / ``anchor_date`` / ``snapshot_id`` mirror the
    monitoring snapshot contract so both append-only histories stay
    comparable.  ``content_hash`` covers the analysis surface (explicitly
    excluding ``snapshot_id`` and ``recorded_at``).
    """

    scope: str = Field(default="repository")
    snapshot_id: str = Field(default="")
    period_kind: LearningPeriodKind = Field(...)
    anchor_date: date = Field(...)
    engine_version: str = Field(...)
    recorded_at: datetime = Field(...)
    content_hash: str = Field(default="")
    schema_version: str = Field(default=LEARNING_SNAPSHOT_SCHEMA_VERSION)
    counts: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    digest: LearningDigest = Field(default_factory=LearningDigest)
    calibration: LearningCalibration = Field(default_factory=LearningCalibration)
    confidence: LearningConfidence = Field(default_factory=LearningConfidence)
    distributions: dict[str, dict[str, int]] = Field(default_factory=dict)
    knowledge: dict[str, list[LearningPeriod]] = Field(default_factory=dict)
    patterns: list[LearningPattern] = Field(default_factory=list)
    observations: list[LearningObservation] = Field(default_factory=list)
    recommendations: list[LearningRecommendation] = Field(default_factory=list)
    meta: dict[str, str] = Field(default_factory=dict)

    def analytic_payload(self) -> dict[str, Any]:
        """The canonical, hash-covered analysis surface (no snapshots ids)."""
        return {
            "scope": self.scope,
            "period_kind": self.period_kind.value,
            "anchor_date": self.anchor_date.isoformat(),
            "engine_version": self.engine_version,
            "schema_version": self.schema_version,
            "counts": self.counts,
            "metrics": {k: v for k, v in self.metrics.items()},
            "digest": self.digest.model_dump(mode="json"),
            "calibration": self.calibration.model_dump(mode="json"),
            "confidence": self.confidence.model_dump(mode="json"),
            "distributions": self.distributions,
            "knowledge": {
                dimension: [entry.model_dump(mode="json") for entry in entries]
                for dimension, entries in sorted(self.knowledge.items())
            },
            "patterns": [pattern.model_dump(mode="json") for pattern in self.patterns],
            "observations": [obs.model_dump(mode="json") for obs in self.observations],
            "recommendations": [
                rec.model_dump(mode="json") for rec in self.recommendations
            ],
        }

    def content_fingerprint(self) -> str:
        """Deterministic SHA-256 of the canonical analysis payload."""
        material = json.dumps(
            self.analytic_payload(), sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def verify(self) -> bool:
        """``True`` when the stored hash matches the recomputed payload."""
        return self.content_hash == self.content_fingerprint()


class LearningSnapshotSummary(BaseModel):
    """Lightweight header of a recorded learning snapshot."""

    snapshot_id: str = Field(...)
    scope: str = Field(default="repository")
    period_kind: LearningPeriodKind = Field(...)
    anchor_date: date = Field(...)
    engine_version: str = Field(...)
    content_hash: str = Field(...)
    recorded_at: datetime = Field(...)

    def to_dict(self) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "scope": self.scope,
            "period_kind": self.period_kind.value,
            "anchor_date": self.anchor_date.isoformat(),
            "engine_version": self.engine_version,
            "content_hash": self.content_hash,
            "recorded_at": self.recorded_at.isoformat(),
        }


__all__ = [
    "ATTR_UNKNOWN",
    "CALIBRATION_BIN_COUNT",
    "CONFIDENCE_STABLE_STD",
    "LEARNING_DIMENSIONS",
    "LEARNING_SNAPSHOT_SCHEMA_VERSION",
    "LearningCalibration",
    "LearningConfidence",
    "LearningDigest",
    "LearningDimension",
    "LearningObservation",
    "LearningObservationCategory",
    "LearningPattern",
    "LearningPeriod",
    "LearningPeriodKind",
    "LearningRecommendation",
    "LearningSample",
    "LearningSnapshot",
    "LearningSnapshotSummary",
    "OBSERVATION_ACCURACY_DELTA",
    "OBSERVATION_BIAS_THRESHOLD",
    "OBSERVATION_MIN_SAMPLES",
    "RECOMMENDATION_ACCURACY_DELTA",
    "RECOMMENDATION_MAX_CONFIDENCE_BIAS",
    "RECOMMENDATION_MAX_ECE",
    "RECOMMENDATION_MIN_SAMPLES",
    "TrendDirection",
]
