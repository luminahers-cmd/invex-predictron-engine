"""Investment Decision Intelligence — deterministic, transparent reasoning layer.

Project E7 adds the Decision Intelligence layer on top of the existing
decision engine:

* :mod:`predictron_engine.decision.intelligence_models` — value objects
  for contributions, traces, explanations, and calibration adjustments.
* :mod:`predictron_engine.decision.feature_engine` — consumes Feature
  Store snapshots, normalizes values, preserves evidence provenance.
* :mod:`predictron_engine.decision.contribution` — computes positive /
  negative / neutral / confidence contributions.
* :mod:`predictron_engine.decision.trace` — deterministic reasoning graph.
* :mod:`predictron_engine.decision.explainability` — human-readable
  explanations referencing actual evidence.
* :mod:`predictron_engine.decision.calibration_layer` — confidence
  calibration consuming Benchmark Platform outputs.
* :mod:`predictron_engine.decision.report` — deterministic reports.
* :mod:`predictron_engine.decision.service` — top-level orchestration.
"""

from predictron_engine.decision.calibration import (
    action_for_confidence_level,
    apply_recommendation_risk,
    build_calibration_summary,
    classify_confidence_level,
    compute_decision_confidence,
)
from predictron_engine.decision.calibration_layer import (
    CalibrationAdjustment,
    CalibrationLayer,
    CalibrationPoint,
    calibrate_confidence,
)
from predictron_engine.decision.contribution import ContributionEngine
from predictron_engine.decision.explainability import ExplainabilityEngine
from predictron_engine.decision.feature_engine import DecisionFeatureEngine
from predictron_engine.decision.intelligence_models import (
    Contribution,
    ContributionType,
    DecisionIntelligenceReport,
    DecisionTrace,
    DecisionVerdict,
    Explanation,
    TraceNode,
    TraceNodeType,
)
from predictron_engine.decision.models import (
    CalibrationSummary,
    ConfidenceBreakdown,
    ConfidenceFactor,
    ConfidenceLevel,
    DecisionConfidence,
    UncertaintyBreakdown,
)
from predictron_engine.decision.report import DecisionReportBuilder
from predictron_engine.decision.service import DecisionIntelligenceService
from predictron_engine.decision.trace import DecisionTraceEngine

__all__ = [
    "CalibrationAdjustment",
    "CalibrationLayer",
    "CalibrationPoint",
    "CalibrationSummary",
    "ConfidenceBreakdown",
    "ConfidenceFactor",
    "ConfidenceLevel",
    "Contribution",
    "ContributionEngine",
    "ContributionType",
    "DecisionConfidence",
    "DecisionFeatureEngine",
    "DecisionIntelligenceReport",
    "DecisionIntelligenceService",
    "DecisionReportBuilder",
    "DecisionTrace",
    "DecisionTraceEngine",
    "DecisionVerdict",
    "ExplainabilityEngine",
    "Explanation",
    "TraceNode",
    "TraceNodeType",
    "UncertaintyBreakdown",
    "action_for_confidence_level",
    "apply_recommendation_risk",
    "build_calibration_summary",
    "calibrate_confidence",
    "classify_confidence_level",
    "compute_decision_confidence",
]
