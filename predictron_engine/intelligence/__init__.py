"""Sprint 9 — Adaptive Intelligence & Continuous Improvement.

A self-evaluation / observability layer that sits *around* the
deterministic pipeline.  It consumes already-produced outputs (Reports,
evidence, calibration) and computes deterministic analytics:

* benchmark-driven evaluation (:mod:`benchmarks`)
* regression detection (:mod:`regression`)
* historical performance tracking (:mod:`history`)
* calibration monitoring (:mod:`quality`, reusing Sprint 8)
* rule effectiveness (:mod:`rules`)
* recommendation effectiveness (:mod:`recommendations`)
* confidence drift analysis (:mod:`confidence_drift`)
* engine quality metrics (:mod:`quality`)
* performance dashboards (:mod:`analyzer` — code models only, no UI)

This package is fully additive: it never duplicates, replaces, or
parallel-pipelines existing models.  All functions are pure and
deterministic; collections are supplied by the caller (no global state,
no hidden caches, no randomness).

Entry point:
    from predictron_engine.intelligence import IntelligenceAnalyzer
    dashboard = IntelligenceAnalyzer().dashboard(reports=[...])
"""

from predictron_engine.intelligence.analyzer import IntelligenceAnalyzer
from predictron_engine.intelligence.models import (
    CalibrationDrift,
    ConfidenceDrift,
    DriftDirection,
    EngineQualityMetrics,
    IntelligenceDashboard,
    PerformanceSnapshot,
    PerformanceTrend,
    RecommendationEffectiveness,
    RegressionFinding,
    RuleEffectiveness,
    TrendDirection,
)

__all__ = [
    "CalibrationDrift",
    "ConfidenceDrift",
    "DriftDirection",
    "EngineQualityMetrics",
    "IntelligenceAnalyzer",
    "IntelligenceDashboard",
    "PerformanceSnapshot",
    "PerformanceTrend",
    "RecommendationEffectiveness",
    "RegressionFinding",
    "RuleEffectiveness",
    "TrendDirection",
]
