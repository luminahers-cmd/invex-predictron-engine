"""Investment Decision Engine — deterministic decision synthesis.

Sprint 6B adds the Decision Calibration layer:

* :mod:`predictron_engine.decision.models` — calibration value objects
* :mod:`predictron_engine.decision.calibration` — pure deterministic
  functions combining existing pipeline outputs into calibrated
  confidence, uncertainty, explanations, and recommendation risk.
"""

from predictron_engine.decision.calibration import (
    action_for_confidence_level,
    apply_recommendation_risk,
    build_calibration_summary,
    classify_confidence_level,
    compute_decision_confidence,
)
from predictron_engine.decision.models import (
    CalibrationSummary,
    ConfidenceBreakdown,
    ConfidenceFactor,
    ConfidenceLevel,
    DecisionConfidence,
    UncertaintyBreakdown,
)

__all__ = [
    "CalibrationSummary",
    "ConfidenceBreakdown",
    "ConfidenceFactor",
    "ConfidenceLevel",
    "DecisionConfidence",
    "UncertaintyBreakdown",
    "action_for_confidence_level",
    "apply_recommendation_risk",
    "build_calibration_summary",
    "classify_confidence_level",
    "compute_decision_confidence",
]
