"""DecisionSynthesisEngine — Sprint 6C composition root.

Purely deterministic: consumes only artifacts already produced by the
pipeline (evidence, trust, features, observations, consistency,
evaluations, scores, readiness, recommendations, decision, calibrated
confidence) and aggregates them into a DecisionSynthesis. It never
recomputes scores, confidence, trust, evidence, reasoning, or
evaluations.
"""

from __future__ import annotations

from predictron_engine.decision.models import (
    CalibrationSummary,
    DecisionConfidence,
)
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import (
    DecisionSynthesis,
    DimensionAssessment,
    InvestmentDecision,
    InvestmentReadiness,
    Observation,
    Recommendation,
    ScoreResult,
)
from predictron_engine.synthesis.opportunities import aggregate_opportunities
from predictron_engine.synthesis.priorities import prioritize_recommendations
from predictron_engine.synthesis.risks import aggregate_risks
from predictron_engine.synthesis.scenarios import generate_scenarios
from predictron_engine.synthesis.summary import build_executive_summary
from predictron_engine.synthesis.tradeoffs import analyze_trade_offs


class DecisionSynthesisEngine:
    """Aggregates existing pipeline outputs into one synthesis container."""

    def synthesize(
        self,
        *,
        features: ExtractedFeatures,
        observations: list[Observation],
        assessments: list[DimensionAssessment],
        scores: list[ScoreResult],
        recommendations: list[Recommendation],
        readiness: InvestmentReadiness | None,
        decision: InvestmentDecision | None,
        decision_confidence: DecisionConfidence | None,
        calibration_summary: CalibrationSummary | None = None,
        consistency: object | None = None,
        contradiction_graph: object | None = None,
    ) -> DecisionSynthesis:
        """Build the DecisionSynthesis from already-produced outputs."""
        prioritized = prioritize_recommendations(recommendations)
        trade_offs = analyze_trade_offs(scores, assessments, observations, readiness)
        scenarios = generate_scenarios(decision, decision_confidence, readiness)
        risks = aggregate_risks(
            features,
            observations,
            assessments,
            readiness,
            decision_confidence,
            consistency=consistency,
            contradiction_graph=contradiction_graph,
        )
        opportunities = aggregate_opportunities(
            features, scores, assessments, observations, readiness
        )

        overall_confidence = (
            decision_confidence.confidence if decision_confidence else 0.0
        )
        uncertainty_score = (
            decision_confidence.uncertainty_score if decision_confidence else 0.0
        )
        confidence_level = decision_confidence.level.value if decision_confidence else ""
        recommended_action = (
            calibration_summary.recommended_action
            if calibration_summary is not None and calibration_summary.recommended_action
            else (decision_confidence.recommended_action if decision_confidence else "")
        )

        draft = DecisionSynthesis(
            prioritized_recommendations=prioritized,
            trade_offs=trade_offs,
            scenarios=scenarios,
            risks=risks,
            opportunities=opportunities,
            overall_confidence=overall_confidence,
            uncertainty_score=uncertainty_score,
            confidence_level=confidence_level,
            recommended_action=recommended_action,
        )

        summary_text, key_points = build_executive_summary(
            decision,
            readiness,
            draft,
        )
        draft.executive_summary = summary_text
        draft.executive_summary_key_points = key_points
        return draft
