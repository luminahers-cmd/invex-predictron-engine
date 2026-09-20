"""Decision Report — deterministic comprehensive investment decision reports.

Builds a full DecisionIntelligenceReport containing:
  - Decision summary
  - Contribution table
  - Top strengths
  - Top weaknesses
  - Evidence references
  - Feature provenance
  - Calibration
  - Historical comparisons
  - Confidence explanation

Key principles:
  - Fully deterministic: same inputs always produce the same report
  - Aggregates outputs from the contribution, trace, explanation, and
    calibration engines
  - Never modifies feature values
"""

from __future__ import annotations

from typing import Any

from predictron_engine.decision.calibration_layer import (
    CalibrationAdjustment,
    CalibrationLayer,
)
from predictron_engine.decision.contribution import ContributionEngine
from predictron_engine.decision.explainability import ExplainabilityEngine
from predictron_engine.decision.feature_engine import DecisionFeatureEngine
from predictron_engine.decision.intelligence_models import (
    Contribution,
    DecisionIntelligenceReport,
    DecisionTrace,
    DecisionVerdict,
)
from predictron_engine.decision.trace import DecisionTraceEngine
from predictron_engine.feature_store.models import CompanyFeatureSet
from predictron_engine.version import ENGINE_VERSION

# ---------------------------------------------------------------------------
# Report constants.
# ---------------------------------------------------------------------------

_TOP_STRENGTHS_DEFAULT = 5
_TOP_WEAKNESSES_DEFAULT = 5
_HISTORICAL_COMPARISON_LIMIT = 10


class DecisionReportBuilder:
    """Builds deterministic decision intelligence reports.

    Responsibilities:
      - Coordinate contribution, trace, explanation, and calibration engines
      - Aggregate results into a single report
      - Compute historical comparisons
      - Deterministic serialization
    """

    def __init__(
        self,
        feature_engine: DecisionFeatureEngine | None = None,
        contribution_engine: ContributionEngine | None = None,
        trace_engine: DecisionTraceEngine | None = None,
        explainability_engine: ExplainabilityEngine | None = None,
        calibration_layer: CalibrationLayer | None = None,
    ) -> None:
        self._feature_engine = feature_engine or DecisionFeatureEngine()
        self._contribution_engine = contribution_engine or ContributionEngine(
            self._feature_engine,
        )
        self._trace_engine = trace_engine or DecisionTraceEngine(
            self._feature_engine,
        )
        self._explainability_engine = explainability_engine or ExplainabilityEngine(
            self._contribution_engine,
        )
        self._calibration_layer = calibration_layer or CalibrationLayer()

    def build_report(
        self,
        feature_set: CompanyFeatureSet,
        *,
        verdict: DecisionVerdict | None = None,
        confidence: float | None = None,
        calibration: CalibrationAdjustment | None = None,
        recommendation_text: str | None = None,
    ) -> DecisionIntelligenceReport:
        """Build a complete decision intelligence report for a company.

        Args:
            feature_set: Feature Store snapshot of the company.
            verdict: Optional verdict override (derived from the overall
                score if omitted).
            confidence: Optional overall confidence (0-1).
            calibration: Optional calibration adjustment (bottom-up
                calibration used when omitted).
            recommendation_text: Optional recommendation override.

        Returns:
            A complete DecisionIntelligenceReport.
        """
        contributions = self._contribution_engine.compute_contributions(feature_set)
        classified = self._contribution_engine.classify(contributions)

        trace = self._trace_engine.build_trace(
            feature_set, contributions, confidence=confidence,
        )

        resolved_verdict = verdict if verdict is not None else trace.verdict
        resolved_confidence = (
            confidence if confidence is not None else trace.confidence
        )

        explanation = self._explainability_engine.build_explanation(
            feature_set,
            contributions,
            verdict=resolved_verdict,
            recommendation_text=recommendation_text,
        )

        resolved_calibration = calibration or self._calibration_layer.compute_adjustment(
            expected_confidence=resolved_confidence,
            actual_confidence=resolved_confidence,
            benchmark_case_id="company:" + feature_set.company_id,
            use_history=False,
        )

        feature_provenance = self._feature_engine.get_feature_provenance(feature_set)
        evidence_refs = self._feature_engine.get_all_evidence_references(feature_set)

        top_strengths = [
            c.feature_name for c in classified["positive"][:_TOP_STRENGTHS_DEFAULT]
        ]
        top_weaknesses = [
            c.feature_name for c in classified["negative"][:_TOP_WEAKNESSES_DEFAULT]
        ]

        decision_summary = self._build_decision_summary(
            verdict=resolved_verdict,
            overall_score=trace.overall_score,
            confidence=resolved_confidence,
            contributions=contributions,
            trace=trace,
        )

        confidence_explanation = self._build_confidence_explanation(
            resolved_confidence,
            calibration=resolved_calibration,
        )

        return DecisionIntelligenceReport(
            company_id=feature_set.company_id,
            engine_version=ENGINE_VERSION,
            decision_summary=decision_summary,
            contributions=contributions,
            positive_contributions=classified["positive"],
            negative_contributions=classified["negative"],
            neutral_contributions=classified["neutral"],
            confidence_contributions=classified["confidence"],
            trace=trace,
            explanation=explanation,
            calibration=resolved_calibration,
            feature_provenance=feature_provenance,
            evidence_references=evidence_refs,
            top_strengths=top_strengths,
            top_weaknesses=top_weaknesses,
            historical_comparisons=self._build_historical_comparisons(
                feature_set,
                trace,
                resolved_confidence,
            ),
            confidence_explanation=confidence_explanation,
            metadata={
                "feature_count": feature_set.feature_count(),
                "built_at": feature_set.built_at.isoformat(),
                "build_version": feature_set.build_version,
            },
        )

    def compare_reports(
        self,
        reports: list[DecisionIntelligenceReport],
    ) -> list[dict[str, Any]]:
        """Compare multiple company reports deterministically.

        Returns a list of comparison dicts sorted by company_id with
        overall score, verdict, and confidence for each report.
        """
        sorted_reports = sorted(reports, key=lambda r: r.company_id)
        comparisons: list[dict[str, Any]] = []
        for report in sorted_reports:
            trace = report.trace
            comparisons.append({
                "company_id": report.company_id,
                "overall_score": trace.overall_score if trace else None,
                "verdict": (
                    trace.verdict.value if trace else None
                ),
                "confidence": (
                    report.calibration.actual_confidence
                    if report.calibration else None
                ),
                "positive_count": len(report.positive_contributions),
                "negative_count": len(report.negative_contributions),
            })
        return comparisons

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_decision_summary(
        *,
        verdict: DecisionVerdict,
        overall_score: float,
        confidence: float,
        contributions: list[Contribution],
        trace: DecisionTrace,
    ) -> dict[str, Any]:
        """Build the decision summary dict."""
        return {
            "company_id": trace.company_id,
            "verdict": verdict.value,
            "overall_score": round(overall_score, 4),
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "feature_count": len(contributions),
            "trace_id": trace.trace_id,
        }

    @staticmethod
    def _build_confidence_explanation(
        confidence: float,
        *,
        calibration: CalibrationAdjustment,
    ) -> str:
        """Explain why the confidence level is what it is."""
        parts = [
            f"Overall confidence is {confidence:.2f}.",
        ]
        if calibration.calibration_delta != 0.0:
            parts.append(
                f"Calibration adjusted confidence by "
                f"{calibration.adjustment_applied:+.4f} "
                f"(expected {calibration.expected_confidence:.2f}, "
                f"actual {calibration.actual_confidence:.2f})."
            )
        else:
            parts.append("Calibration is well-aligned; no adjustment applied.")

        if calibration.historical_calibration:
            parts.append(
                f"Based on {len(calibration.historical_calibration)} "
                f"historical calibration point(s)."
            )

        return " ".join(parts)

    def _build_historical_comparisons(
        self,
        feature_set: CompanyFeatureSet,
        trace: DecisionTrace,
        confidence: float,
    ) -> list[dict[str, Any]]:
        """Build historical comparison data.

        When no history is available, records the current observation
        as the baseline.
        """
        comparisons: list[dict[str, Any]] = []

        baseline = {
            "company_id": feature_set.company_id,
            "built_at": feature_set.built_at.isoformat(),
            "overall_score": round(trace.overall_score, 4),
            "verdict": trace.verdict.value,
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "is_baseline": True,
        }
        comparisons.append(baseline)

        return comparisons[:_HISTORICAL_COMPARISON_LIMIT]

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def report_to_dict(self, report: DecisionIntelligenceReport) -> dict[str, Any]:
        """Deterministic serialization of a report."""
        return report.to_dict()

    def render_markdown(self, report: DecisionIntelligenceReport) -> str:
        """Render a report as human-readable markdown.

        Deterministic rendering for report output.
        """
        lines: list[str] = []
        lines.append(f"# Decision Intelligence Report — {report.company_id}")
        lines.append("")
        lines.append(
            f"Generated: {report.report_timestamp.isoformat()} "
            f"(engine {report.engine_version})"
        )
        lines.append("")

        summary = report.decision_summary
        lines.append("## Decision Summary")
        lines.append("")
        lines.append(f"- **Verdict:** {summary.get('verdict', 'unknown')}")
        lines.append(f"- **Overall score:** {summary.get('overall_score', 0.0)}")
        lines.append(f"- **Confidence:** {summary.get('confidence', 0.0):.2f}")
        lines.append(f"- **Features evaluated:** {summary.get('feature_count', 0)}")
        lines.append("")

        if report.explanation and report.explanation.headline:
            lines.append("## Explanation")
            lines.append("")
            lines.append(report.explanation.headline)
            lines.append("")
            lines.append(report.explanation.full_explanation)
            lines.append("")

        lines.append("## Contribution Table")
        lines.append("")
        lines.append("| Feature | Type | Weight | Contribution |")
        lines.append("|---|---|---|---|")
        for c in report.contributions:
            lines.append(
                f"| {c.feature_name} | {c.contribution_type.value} "
                f"| {c.weight:.3f} | {c.computed_contribution:+.4f} |"
            )
        lines.append("")

        if report.explanation and report.explanation.strengths:
            lines.append("## Top Strengths")
            lines.append("")
            for s in report.explanation.strengths:
                lines.append(f"- {s}")
            lines.append("")

        if report.explanation and report.explanation.weaknesses:
            lines.append("## Top Weaknesses")
            lines.append("")
            for w in report.explanation.weaknesses:
                lines.append(f"- {w}")
            lines.append("")

        if report.evidence_references:
            lines.append("## Evidence References")
            lines.append("")
            lines.append(f"Total: {len(report.evidence_references)}")
            for ref in report.evidence_references[:_HISTORICAL_COMPARISON_LIMIT]:
                lines.append(
                    f"- `{ref.get('source_type', '')}` "
                    f"{ref.get('source_id', '')} "
                    f"{ref.get('source_field', '')}"
                )
            lines.append("")

        if report.calibration:
            cal = report.calibration
            lines.append("## Calibration")
            lines.append("")
            lines.append(
                f"- Expected confidence: {cal.expected_confidence:.4f}"
            )
            lines.append(f"- Actual confidence: {cal.actual_confidence:.4f}")
            lines.append(f"- Calibration delta: {cal.calibration_delta:+.4f}")
            lines.append(f"- Adjustment applied: {cal.adjustment_applied:+.4f}")
            lines.append("")

        lines.append("---")
        return "\n".join(lines)
