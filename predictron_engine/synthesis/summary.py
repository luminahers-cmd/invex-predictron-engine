"""Deterministic executive summary generation (Sprint 6C).

Produces one deterministic synthesis paragraph plus bullet points from
already-produced outputs: decision, calibrated confidence, uncertainty,
top opportunities, top risks, prioritized recommendations, and next
action. No generative text — fixed templates only.
"""

from __future__ import annotations

from predictron_engine.models.report import (
    DecisionSynthesis,
    InvestmentDecision,
    InvestmentReadiness,
)

_MAX_LISTED_ITEMS = 3


def _label(value: str) -> str:
    return value.replace("_", " ")


def build_executive_summary(
    decision: InvestmentDecision | None,
    readiness: InvestmentReadiness | None,
    synthesis: DecisionSynthesis,
) -> tuple[str, list[str]]:
    """Return (summary_paragraph, key_points) using fixed templates."""
    if decision is None:
        return (
            "Decision synthesis unavailable: no investment decision was "
            "produced by the pipeline.",
            [],
        )

    category = _label(decision.category.value)
    conviction = _label(decision.conviction.value)
    confidence_pct = synthesis.overall_confidence * 100
    uncertainty_pct = synthesis.uncertainty_score * 100

    summary = (
        f"Deterministic synthesis: {category} with {conviction} conviction "
        f"at a composite score of {decision.composite_score:.1f}/100. "
        f"Calibrated confidence is {confidence_pct:.0f}% "
        f"({synthesis.confidence_level}) with {uncertainty_pct:.0f}% "
        f"uncertainty; recommended action: {synthesis.recommended_action}."
    )

    if synthesis.opportunities:
        top_opportunities = ", ".join(
            item.label for item in synthesis.opportunities[:_MAX_LISTED_ITEMS]
        )
        summary += f" Leading opportunities: {top_opportunities}."
    if synthesis.risks:
        top_risks = ", ".join(
            f"{item.label} ({item.severity.value})"
            for item in synthesis.risks[:_MAX_LISTED_ITEMS]
        )
        summary += f" Principal risks: {top_risks}."
    if synthesis.trade_offs:
        trade_off = synthesis.trade_offs[0]
        summary += (
            f" Key tension in {_label(trade_off.dimension)}: "
            f"{trade_off.strength} versus {trade_off.concern}."
        )
    if readiness is not None and readiness.gaps:
        gaps = "; ".join(readiness.gaps[:2])
        summary += f" Closing the biggest gaps would raise conviction: {gaps}."

    key_points: list[str] = [
        f"Investment decision: {category}",
        f"Conviction: {conviction}",
        f"Calibrated confidence: {confidence_pct:.0f}% ({synthesis.confidence_level})",
        f"Uncertainty: {uncertainty_pct:.0f}%",
    ]
    for opportunity in synthesis.opportunities[:_MAX_LISTED_ITEMS]:
        key_points.append(
            f"Opportunity: {opportunity.label} "
            f"[impact: {opportunity.impact.value}, "
            f"confidence: {opportunity.confidence:.2f}]"
        )
    for risk in synthesis.risks[:_MAX_LISTED_ITEMS]:
        key_points.append(
            f"Risk: {risk.label} [severity: {risk.severity.value}, "
            f"confidence: {risk.confidence:.2f}]"
        )
    for recommendation in synthesis.prioritized_recommendations[:_MAX_LISTED_ITEMS]:
        rank_label = (
            f"#{recommendation.rank}" if recommendation.rank is not None else ""
        )
        key_points.append(
            f"Recommendation {rank_label}: {recommendation.action}".strip()
        )
    if synthesis.recommended_action:
        key_points.append(f"Next action: {synthesis.recommended_action}")

    return summary, key_points
