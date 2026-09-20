"""Decision Intelligence Layer — top-level service.

Orchestrates the Decision Intelligence pipeline:

    Feature Store Snapshot
        ↓
    DecisionFeatureEngine  (normalize + weight + provenance)
        ↓
    ContributionEngine     (positive / negative / neutral / confidence)
        ↓
    DecisionTraceEngine    (reasoning graph)
        ↓
    ExplainabilityEngine   (human-readable explanation)
        ↓
    CalibrationLayer       (benchmark-driven confidence adjustment)
        ↓
    DecisionReportBuilder  (deterministic report)

Consumes existing pipeline artifacts only; never recomputes raw features.
No ML, no LLMs, no probabilistic black boxes — every output is fully
explainable and deterministic.
"""

from __future__ import annotations

from typing import Any

from predictron_engine.decision.calibration_layer import (
    CalibrationAdjustment,
    CalibrationLayer,
    CalibrationPoint,
)
from predictron_engine.decision.contribution import ContributionEngine
from predictron_engine.decision.explainability import ExplainabilityEngine
from predictron_engine.decision.feature_engine import DecisionFeatureEngine
from predictron_engine.decision.intelligence_models import (
    Contribution,
    DecisionIntelligenceReport,
    DecisionTrace,
    DecisionVerdict,
    Explanation,
)
from predictron_engine.decision.report import DecisionReportBuilder
from predictron_engine.decision.trace import DecisionTraceEngine
from predictron_engine.feature_store.models import CompanyFeatureSet, FeatureCategory

__all__ = [
    "CalibrationAdjustment",
    "CalibrationLayer",
    "CalibrationPoint",
    "Contribution",
    "ContributionEngine",
    "DecisionFeatureEngine",
    "DecisionIntelligenceReport",
    "DecisionReportBuilder",
    "DecisionTrace",
    "DecisionTraceEngine",
    "DecisionVerdict",
    "ExplainabilityEngine",
    "Explanation",
]

# Re-export FeatureCategory for convenience in configuration.
_ = FeatureCategory


class DecisionIntelligenceService:
    """High-level service for the Decision Intelligence layer.

    Provides a single entry point for producing decisions, traces,
    explanations, calibrations, and reports from Feature Store snapshots.
    """

    def __init__(
        self,
        feature_engine: DecisionFeatureEngine | None = None,
        contribution_engine: ContributionEngine | None = None,
        trace_engine: DecisionTraceEngine | None = None,
        explainability_engine: ExplainabilityEngine | None = None,
        calibration_layer: CalibrationLayer | None = None,
        report_builder: DecisionReportBuilder | None = None,
    ) -> None:
        self.feature_engine = feature_engine or DecisionFeatureEngine()
        self.contribution_engine = contribution_engine or ContributionEngine(
            self.feature_engine,
        )
        self.trace_engine = trace_engine or DecisionTraceEngine(self.feature_engine)
        self.explainability_engine = (
            explainability_engine
            or ExplainabilityEngine(self.contribution_engine)
        )
        self.calibration_layer = calibration_layer or CalibrationLayer()
        self.report_builder = report_builder or DecisionReportBuilder(
            feature_engine=self.feature_engine,
            contribution_engine=self.contribution_engine,
            trace_engine=self.trace_engine,
            explainability_engine=self.explainability_engine,
            calibration_layer=self.calibration_layer,
        )

    # ------------------------------------------------------------------
    # Primary operations
    # ------------------------------------------------------------------

    def produce_decision(
        self,
        feature_set: CompanyFeatureSet,
        *,
        verdict: DecisionVerdict | None = None,
        confidence: float | None = None,
        recommendation_text: str | None = None,
        calibration: CalibrationAdjustment | None = None,
    ) -> DecisionIntelligenceReport:
        """Produce a full decision intelligence report for a company."""
        return self.report_builder.build_report(
            feature_set,
            verdict=verdict,
            confidence=confidence,
            calibration=calibration,
            recommendation_text=recommendation_text,
        )

    def produce_trace(
        self,
        feature_set: CompanyFeatureSet,
        *,
        confidence: float | None = None,
    ) -> DecisionTrace:
        """Produce a decision reasoning trace for a company."""
        contributions = self.contribution_engine.compute_contributions(feature_set)
        return self.trace_engine.build_trace(
            feature_set, contributions, confidence=confidence,
        )

    def produce_explanation(
        self,
        feature_set: CompanyFeatureSet,
        *,
        verdict: DecisionVerdict | None = None,
        recommendation_text: str | None = None,
    ) -> Explanation:
        """Produce a human-readable explanation for a company."""
        contributions = self.contribution_engine.compute_contributions(feature_set)
        return self.explainability_engine.build_explanation(
            feature_set,
            contributions,
            verdict=verdict,
            recommendation_text=recommendation_text,
        )

    def calibrate(
        self,
        *,
        expected_confidence: float,
        actual_confidence: float,
        benchmark_case_id: str = "default",
    ) -> CalibrationAdjustment:
        """Calibrate confidence using Benchmark Platform outputs."""
        return self.calibration_layer.compute_adjustment(
            expected_confidence=expected_confidence,
            actual_confidence=actual_confidence,
            benchmark_case_id=benchmark_case_id,
        )

    def compare(
        self,
        reports: list[DecisionIntelligenceReport],
    ) -> list[dict[str, Any]]:
        """Compare multiple company reports."""
        return self.report_builder.compare_reports(reports)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def compute_contributions(
        self,
        feature_set: CompanyFeatureSet,
    ) -> list[Contribution]:
        """Compute all contributions for a company."""
        return self.contribution_engine.compute_contributions(feature_set)

    def classify_contributions(
        self,
        feature_set: CompanyFeatureSet,
    ) -> dict[str, list[Contribution]]:
        """Classify contributions into bucket lists."""
        contributions = self.contribution_engine.compute_contributions(feature_set)
        return self.contribution_engine.classify(contributions)

    def add_historical_calibration_point(
        self,
        point: CalibrationPoint,
    ) -> None:
        """Record a historical calibration point."""
        self.calibration_layer.add_historical_point(point)

    def calibration_summary(
        self,
        adjustments: list[CalibrationAdjustment],
    ) -> dict[str, Any]:
        """Summarize a set of calibration adjustments."""
        return self.calibration_layer.build_calibration_summary(adjustments)
