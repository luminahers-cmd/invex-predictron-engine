"""Phase 7 — Continuous Learning Intelligence.

The deterministic, post-deployment learning layer of the platform.  It mines
the evaluated prediction ledger (forecasts, outcomes, evaluations, and their
frozen report features) into reusable, rule-based intelligence:

* per-dimension knowledge (sector / stage / country / technology / business
  model / founder) with accuracy, confidence bias, and sample support;
* canonical observations (accuracy excess/deficit, over/under-confidence,
  low support, strong accuracy, high false-positive/negative rate);
* deterministic calibration (equal-width ECE) and confidence statistics;
* canned, threshold-driven recommendations — never LLM-generated, never
  fabricated.

The layer is pure and deterministic: identical inputs always produce
identical snapshots.  The engine never touches the database, filesystem, or
any mutable state.

Modules:
    models:  Value objects, thresholds, and snapshot contracts.
    engine:  Deterministic snapshot builder and canonical aggregation rules.
"""

from predictron_engine.learning.engine import build_learning_snapshot
from predictron_engine.learning.models import (
    ATTR_UNKNOWN,
    CALIBRATION_BIN_COUNT,
    CONFIDENCE_STABLE_STD,
    LEARNING_DIMENSIONS,
    LEARNING_SNAPSHOT_SCHEMA_VERSION,
    OBSERVATION_ACCURACY_DELTA,
    OBSERVATION_BIAS_THRESHOLD,
    OBSERVATION_MIN_SAMPLES,
    RECOMMENDATION_ACCURACY_DELTA,
    RECOMMENDATION_MAX_CONFIDENCE_BIAS,
    RECOMMENDATION_MAX_ECE,
    RECOMMENDATION_MIN_SAMPLES,
    LearningCalibration,
    LearningConfidence,
    LearningDigest,
    LearningDimension,
    LearningObservation,
    LearningObservationCategory,
    LearningPattern,
    LearningPeriod,
    LearningPeriodKind,
    LearningRecommendation,
    LearningSample,
    LearningSnapshot,
    LearningSnapshotSummary,
    TrendDirection,
)

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
    "build_learning_snapshot",
]
