"""Traction dimension evaluator.

Translates observations and evidence about traction signals into a
structured DimensionAssessment. Augments qualitative observations with
quantitative metric references where available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from predictron_engine.evaluation.evaluation_models import DimensionAssessment
from predictron_engine.evaluation.evaluators.base import (
    build_citations_for_assessment,
    calculate_average_confidence,
    calculate_weighted_importance,
    filter_evidence_by_domain,
    filter_observations,
    generate_cross_signal_context,
    generate_rationale,
    generate_summary,
)

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation


def _build_quantitative_context(features: ExtractedFeatures) -> str:
    """Build a quantitative context string from structured metrics."""
    parts: list[str] = []

    if features.arr_usd is not None:
        parts.append(f"ARR of ${features.arr_usd / 1_000_000:.1f}M")
    if features.mrr_usd is not None:
        parts.append(f"MRR of ${features.mrr_usd / 1_000:.0f}K")
    if features.growth_rate_pct is not None:
        parts.append(f"growth rate of {features.growth_rate_pct:.0f}%")
    if features.nrr_pct is not None:
        parts.append(f"NRR of {features.nrr_pct:.0f}%")
    if features.churn_rate_pct is not None:
        parts.append(f"churn of {features.churn_rate_pct:.1f}%")
    if features.customer_count is not None:
        parts.append(f"{features.customer_count:,} customers")
    if features.runway_months is not None:
        parts.append(f"{features.runway_months} months runway")
    if features.burn_rate_usd is not None:
        parts.append(f"burn rate of ${features.burn_rate_usd / 1_000:.0f}K/mo")
    if features.cac_usd is not None and features.ltv_usd is not None:
        ratio = features.ltv_usd / max(features.cac_usd, 1)
        parts.append(f"LTV/CAC of {ratio:.1f}x")

    if not parts:
        return ""
    return " Quantitative metrics: " + "; ".join(parts) + "."


class TractionEvaluator:
    """Evaluates the traction dimension.

    Translates traction-related observations and evidence into a structured
    assessment explaining market validation and growth signals. Incorporates
    quantitative metric context where available.
    """

    @property
    def dimension(self) -> str:
        return "traction_signals"

    def evaluate(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        evidence: list[EvidenceItem],
    ) -> DimensionAssessment:
        """Evaluate traction signals based on observations and evidence."""
        traction_obs = filter_observations(observations, self.dimension)
        traction_evidence = filter_evidence_by_domain(evidence, "traction")

        summary = generate_summary(self.dimension, traction_obs, traction_evidence)
        rationale = generate_rationale(self.dimension, traction_obs, traction_evidence)
        confidence = calculate_average_confidence(traction_obs)
        weighted_conf = calculate_weighted_importance(traction_obs)

        cross_ctx = generate_cross_signal_context(observations, self.dimension)
        if cross_ctx:
            rationale += cross_ctx

        quant_ctx = _build_quantitative_context(features)
        if quant_ctx:
            rationale += quant_ctx

        metadata: dict[str, object] = {
            "has_revenue": features.has_revenue or False,
            "funding_stage": features.funding_stage or "unknown",
            "weighted_confidence": round(weighted_conf, 4),
            "cross_signal_available": bool(cross_ctx),
        }
        if features.arr_usd is not None:
            metadata["arr_usd"] = features.arr_usd
        if features.growth_rate_pct is not None:
            metadata["growth_rate_pct"] = features.growth_rate_pct
        if features.nrr_pct is not None:
            metadata["nrr_pct"] = features.nrr_pct
        if features.churn_rate_pct is not None:
            metadata["churn_rate_pct"] = features.churn_rate_pct
        if features.customer_count is not None:
            metadata["customer_count"] = features.customer_count
        if features.runway_months is not None:
            metadata["runway_months"] = features.runway_months

        citations = build_citations_for_assessment(traction_obs, traction_evidence)

        return DimensionAssessment(
            dimension=self.dimension,
            summary=summary,
            rationale=rationale,
            confidence=confidence,
            supporting_observations=traction_obs,
            supporting_evidence=traction_evidence,
            metadata=metadata,
            citations=citations,
        )
