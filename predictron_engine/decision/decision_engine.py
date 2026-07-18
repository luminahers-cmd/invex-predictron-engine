"""Investment Decision Engine — deterministic decision synthesis.

Transforms scores, confidence, risk, evidence quality, cross-signal
reasoning, and recommendations into a deterministic investment decision.

The engine produces:
  - A decision category (Strong Invest, Invest, Watch, Investigate Further, Pass)
  - A conviction level (Very High, High, Moderate, Low, Very Low)
  - A structured rationale traceable to evidence
  - Quantitative decision factors for explainability

Key principles:
  - Fully deterministic: same inputs always produce the same output
  - No arbitrary magic numbers — all thresholds are named constants
  - Conviction is independent of confidence (confidence = evidence reliability;
    conviction = investment attractiveness)
  - Every rationale statement traces to pipeline evidence
  - Decision categories use score ranges derived from the composite score

Extensibility:
  - Thresholds can be adjusted per investment context
  - Factor weights can be overridden
  - Rationale generators can be replaced
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from predictron_engine.knowledge.concepts import DIMENSION_LABELS, AnalysisDimension
from predictron_engine.models.report import (
    ConfidenceAssessment,
    ConvictionLevel,
    DecisionCategory,
    DecisionRationale,
    DimensionAssessment,
    InvestmentDecision,
    Observation,
    ScoreResult,
    SignalRelationship,
)

if TYPE_CHECKING:
    from predictron_engine.models.extracted_features import ExtractedFeatures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Decision thresholds — named constants, not magic numbers.
#
# The composite score maps to a decision category via these thresholds.
# Ranges are inclusive at the lower bound: score >= threshold => category.
# ---------------------------------------------------------------------------

_DECISION_THRESHOLDS: list[tuple[float, DecisionCategory]] = [
    (75.0, DecisionCategory.STRONG_INVEST),
    (60.0, DecisionCategory.INVEST),
    (45.0, DecisionCategory.WATCH),
    (30.0, DecisionCategory.INVESTIGATE_FURTHER),
    (0.0, DecisionCategory.PASS),
]

# ---------------------------------------------------------------------------
# Conviction thresholds — independent of decision thresholds.
#
# Conviction measures investment attractiveness, not evidence reliability.
# It is derived from a conviction composite that blends confidence,
# evidence quality, data completeness, and cross-signal coherence.
# ---------------------------------------------------------------------------

_CONVICTION_THRESHOLDS: list[tuple[float, ConvictionLevel]] = [
    (0.80, ConvictionLevel.VERY_HIGH),
    (0.65, ConvictionLevel.HIGH),
    (0.45, ConvictionLevel.MODERATE),
    (0.25, ConvictionLevel.LOW),
    (0.0, ConvictionLevel.VERY_LOW),
]

# ---------------------------------------------------------------------------
# Factor weights for composite score computation.
#
# These weights determine how much each signal category contributes to
# the final composite score. They sum to 1.0.
# ---------------------------------------------------------------------------

_SCORE_WEIGHT: float = 0.40
_READINESS_WEIGHT: float = 0.25
_CONFIDENCE_WEIGHT: float = 0.15
_EVIDENCE_QUALITY_WEIGHT: float = 0.10
_CROSS_SIGNAL_WEIGHT: float = 0.10

# ---------------------------------------------------------------------------
# Conviction factor weights.
#
# Conviction blends evidence reliability with opportunity attractiveness.
# ---------------------------------------------------------------------------

_CONV_CONFIDENCE_WEIGHT: float = 0.30
_CONV_EVIDENCE_QUALITY_WEIGHT: float = 0.20
_CONV_DATA_COMPLETENESS_WEIGHT: float = 0.20
_CONV_CROSS_SIGNAL_WEIGHT: float = 0.15
_CONV_READINESS_WEIGHT: float = 0.15

# ---------------------------------------------------------------------------
# Risk modifier constants.
# ---------------------------------------------------------------------------

_RISK_PENALTY_PER_SIGNAL: float = 0.02
_RISK_MODIFIER_FLOOR: float = 0.50
_MAX_RISK_SIGNALS_FOR_FULL_PENALTY: int = 15

# ---------------------------------------------------------------------------
# Data quality modifier constants.
# ---------------------------------------------------------------------------

_DATA_QUALITY_FULL_THRESHOLD: float = 0.7
_DATA_QUALITY_MODIFIER_FLOOR: float = 0.60


class DefaultDecisionEngine:
    """Deterministic investment decision engine.

    Synthesizes all pipeline signals into a single investment decision
    with category, conviction, rationale, and quantitative factors.
    """

    def decide(
        self,
        features: ExtractedFeatures,
        observations: list[Observation],
        scores: list[ScoreResult],
        confidence: list[ConfidenceAssessment],
        assessments: list[DimensionAssessment] | None = None,
        signal_relationships: list[SignalRelationship] | None = None,
    ) -> InvestmentDecision:
        """Produce a deterministic investment decision.

        Args:
            features: Extracted features from the startup.
            observations: Reasoning layer observations.
            scores: Dimension scores from the scoring engine.
            confidence: Per-dimension confidence assessments.
            assessments: Per-dimension evaluations (optional).
            signal_relationships: Cross-signal relationships (optional).

        Returns:
            A fully populated InvestmentDecision.
        """
        logger.info("Computing investment decision")

        assessments = assessments or []
        signal_relationships = signal_relationships or []

        # Compute quantitative factors
        score_factor = self._compute_score_factor(scores)
        readiness_factor = self._compute_readiness_factor(assessments)
        confidence_factor = self._compute_confidence_factor(confidence)
        evidence_quality_factor = self._compute_evidence_quality_factor(
            observations, evidence_items=None,
        )
        cross_signal_factor = self._compute_cross_signal_factor(
            signal_relationships,
        )
        risk_modifier = self._compute_risk_modifier(features)
        data_quality_modifier = self._compute_data_quality_modifier(features)

        # Composite score: weighted blend, clamped to [0, 100]
        raw_composite = (
            score_factor * _SCORE_WEIGHT
            + readiness_factor * _READINESS_WEIGHT
            + confidence_factor * 100.0 * _CONFIDENCE_WEIGHT
            + evidence_quality_factor * 100.0 * _EVIDENCE_QUALITY_WEIGHT
            + cross_signal_factor * 100.0 * _CROSS_SIGNAL_WEIGHT
        )

        # Apply modifiers
        composite_score = raw_composite * data_quality_modifier * risk_modifier
        composite_score = max(0.0, min(100.0, composite_score))

        # Determine decision category
        category = self._classify_decision(composite_score)

        # Determine conviction
        conviction_composite = self._compute_conviction_composite(
            confidence_factor,
            evidence_quality_factor,
            features.data_completeness,
            cross_signal_factor,
            readiness_factor / 100.0,
        )
        conviction = self._classify_conviction(conviction_composite)

        # Compute margins to next category
        next_threshold, margin = self._compute_margin(composite_score)

        # Build decision factors
        decision_factors = {
            "score_contribution": round(score_factor * _SCORE_WEIGHT, 4),
            "readiness_contribution": round(
                readiness_factor * _READINESS_WEIGHT, 4
            ),
            "confidence_contribution": round(
                confidence_factor * 100.0 * _CONFIDENCE_WEIGHT, 4
            ),
            "evidence_quality_contribution": round(
                evidence_quality_factor * 100.0 * _EVIDENCE_QUALITY_WEIGHT, 4
            ),
            "cross_signal_contribution": round(
                cross_signal_factor * 100.0 * _CROSS_SIGNAL_WEIGHT, 4
            ),
            "data_quality_modifier": round(data_quality_modifier, 4),
            "risk_modifier": round(risk_modifier, 4),
            "conviction_composite": round(conviction_composite, 4),
        }

        # Build rationale
        rationale = self._build_rationale(
            category,
            conviction,
            composite_score,
            scores,
            confidence,
            assessments,
            observations,
            signal_relationships,
            features,
            decision_factors,
            next_threshold,
        )

        logger.info(
            "Decision: %s (conviction: %s, composite: %.1f)",
            category.value,
            conviction.value,
            composite_score,
        )

        return InvestmentDecision(
            category=category,
            conviction=conviction,
            composite_score=round(composite_score, 2),
            rationale=rationale,
            decision_factors=decision_factors,
            next_category_threshold=next_threshold,
            margin_to_next_category=round(margin, 2),
            data_quality_modifier=round(data_quality_modifier, 4),
            risk_modifier=round(risk_modifier, 4),
        )

    # ------------------------------------------------------------------
    # Factor computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_score_factor(scores: list[ScoreResult]) -> float:
        """Average of all dimension scores (0-100)."""
        if not scores:
            return 50.0
        return sum(s.score for s in scores) / len(scores)

    @staticmethod
    def _compute_readiness_factor(
        assessments: list[DimensionAssessment],
    ) -> float:
        """Average dimension contribution from assessments (0-100)."""
        if not assessments:
            return 50.0
        scores = [a.score for a in assessments if a.score is not None]
        if not scores:
            return 50.0
        return sum(scores) / len(scores)

    @staticmethod
    def _compute_confidence_factor(
        confidence: list[ConfidenceAssessment],
    ) -> float:
        """Average confidence across dimensions (0-1)."""
        if not confidence:
            return 0.0
        return sum(c.confidence for c in confidence) / len(confidence)

    @staticmethod
    def _compute_evidence_quality_factor(
        observations: list[Observation],
        evidence_items: list | None = None,
    ) -> float:
        """Evidence quality factor (0-1) based on observation confidence
        and diversity."""
        if not observations:
            return 0.0

        avg_confidence = sum(o.confidence for o in observations) / len(
            observations
        )

        categories = {o.category for o in observations}
        diversity = min(len(categories) / 5.0, 1.0)

        high_importance_count = sum(
            1 for o in observations if o.importance >= 0.7
        )
        importance_ratio = min(high_importance_count / max(len(observations), 1), 1.0)

        quality = (
            avg_confidence * 0.5
            + diversity * 0.25
            + importance_ratio * 0.25
        )
        return max(0.0, min(1.0, quality))

    @staticmethod
    def _compute_cross_signal_factor(
        relationships: list[SignalRelationship],
    ) -> float:
        """Cross-signal coherence factor (0-1).

        Higher when signals reinforce each other.
        Lower when signals conflict.
        """
        if not relationships:
            return 0.5  # Neutral when no cross-signals detected

        reinforcing = sum(
            1 for r in relationships if r.relationship_type == "reinforcing"
        )
        conflicting = sum(
            1 for r in relationships if r.relationship_type == "conflicting"
        )
        total = reinforcing + conflicting

        if total == 0:
            return 0.5

        return reinforcing / total

    @staticmethod
    def _compute_risk_modifier(features: ExtractedFeatures) -> float:
        """Risk modifier (0.5-1.0) that penalizes high risk signal counts."""
        risk_fields = [
            features.market_risk,
            features.founder_risk,
            features.execution_risk,
            features.product_risk,
            features.technology_risk,
            features.business_model_risk,
            features.traction_risk,
            features.competitive_risk,
            features.regulatory_risk,
            features.operational_risk,
            features.platform_dependency_risk,
            features.customer_concentration_risk,
            features.hiring_risk,
            features.funding_risk,
            features.scaling_risk,
            features.security_risk,
            features.compliance_risk,
        ]

        total_risk_signals = sum(len(r) for r in risk_fields if r)
        capped = min(total_risk_signals, _MAX_RISK_SIGNALS_FOR_FULL_PENALTY)
        penalty = capped * _RISK_PENALTY_PER_SIGNAL

        return max(_RISK_MODIFIER_FLOOR, 1.0 - penalty)

    @staticmethod
    def _compute_data_quality_modifier(features: ExtractedFeatures) -> float:
        """Data quality modifier (0.6-1.0) that dampens decisions
        based on data completeness."""
        dc = features.data_completeness
        if dc >= _DATA_QUALITY_FULL_THRESHOLD:
            return 1.0
        modifier = _DATA_QUALITY_MODIFIER_FLOOR + (
            dc / _DATA_QUALITY_FULL_THRESHOLD
        ) * (1.0 - _DATA_QUALITY_MODIFIER_FLOOR)
        return max(_DATA_QUALITY_MODIFIER_FLOOR, min(1.0, modifier))

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_decision(composite_score: float) -> DecisionCategory:
        """Map composite score to a decision category via thresholds."""
        for threshold, category in _DECISION_THRESHOLDS:
            if composite_score >= threshold:
                return category
        return DecisionCategory.PASS

    @staticmethod
    def _classify_conviction(conviction_composite: float) -> ConvictionLevel:
        """Map conviction composite to a conviction level via thresholds."""
        for threshold, level in _CONVICTION_THRESHOLDS:
            if conviction_composite >= threshold:
                return level
        return ConvictionLevel.VERY_LOW

    @staticmethod
    def _compute_margin(
        composite_score: float,
    ) -> tuple[float, float]:
        """Compute the next category threshold and margin to it."""
        for threshold, _category in _DECISION_THRESHOLDS:
            if composite_score >= threshold:
                # Find the next higher threshold
                for next_threshold, _next_cat in _DECISION_THRESHOLDS:
                    if next_threshold > composite_score:
                        return (
                            next_threshold,
                            next_threshold - composite_score,
                        )
                # Already at highest category
                return (100.0, 100.0 - composite_score)
        # Below all thresholds
        return (_DECISION_THRESHOLDS[-1][0], composite_score)

    def _compute_conviction_composite(
        self,
        confidence_factor: float,
        evidence_quality: float,
        data_completeness: float,
        cross_signal_factor: float,
        readiness_factor_normalized: float,
    ) -> float:
        """Compute conviction composite (0-1) from blended factors."""
        return (
            confidence_factor * _CONV_CONFIDENCE_WEIGHT
            + evidence_quality * _CONV_EVIDENCE_QUALITY_WEIGHT
            + data_completeness * _CONV_DATA_COMPLETENESS_WEIGHT
            + cross_signal_factor * _CONV_CROSS_SIGNAL_WEIGHT
            + readiness_factor_normalized * _CONV_READINESS_WEIGHT
        )

    # ------------------------------------------------------------------
    # Rationale generation
    # ------------------------------------------------------------------

    def _build_rationale(
        self,
        category: DecisionCategory,
        conviction: ConvictionLevel,
        composite_score: float,
        scores: list[ScoreResult],
        confidence: list[ConfidenceAssessment],
        assessments: list[DimensionAssessment],
        observations: list[Observation],
        signal_relationships: list[SignalRelationship],
        features: ExtractedFeatures,
        decision_factors: dict[str, float],
        next_threshold: float,
    ) -> DecisionRationale:
        """Build structured rationale for the decision."""
        primary_for = self._collect_reasons_for(
            scores, assessments, observations, features,
        )
        primary_against = self._collect_reasons_against(
            scores, assessments, observations, features,
        )
        positive_signals = self._collect_top_positive_signals(
            scores, assessments, observations,
        )
        negative_signals = self._collect_top_negative_signals(
            scores, assessments, observations,
        )
        missing_info = self._collect_missing_information(features)
        why_not_higher = self._build_why_not_higher(
            category, composite_score, next_threshold, scores, confidence,
        )
        key_evidence = self._build_key_evidence_summary(
            positive_signals, negative_signals,
        )
        info_to_change = self._collect_information_that_could_change(
            category, features, observations,
        )
        confidence_explanation = self._build_confidence_explanation(
            confidence, features,
        )

        return DecisionRationale(
            primary_reasons_for=primary_for,
            primary_reasons_against=primary_against,
            highest_impact_positive=positive_signals,
            highest_impact_negative=negative_signals,
            missing_information=missing_info,
            confidence_explanation=confidence_explanation,
            why_not_higher=why_not_higher,
            key_evidence_summary=key_evidence,
            information_that_could_change_decision=info_to_change,
        )

    @staticmethod
    def _collect_reasons_for(
        scores: list[ScoreResult],
        assessments: list[DimensionAssessment],
        observations: list[Observation],
        features: ExtractedFeatures,
    ) -> list[str]:
        """Collect primary reasons supporting the decision."""
        reasons: list[str] = []

        high_scores = [s for s in scores if s.score >= 65.0]
        for s in sorted(high_scores, key=lambda x: x.score, reverse=True)[:3]:
            label = _get_dimension_label(s.dimension)
            reasons.append(f"{label} scores {s.score:.0f}/100 — strong signal")

        high_conf_assessments = [
            a for a in assessments if a.confidence >= 0.6 and a.score and a.score >= 60.0
        ]
        for a in sorted(
            high_conf_assessments, key=lambda x: x.confidence, reverse=True
        )[:2]:
            label = _get_dimension_label(a.dimension)
            reasons.append(
                f"{label} assessment has high confidence ({a.confidence:.2f})"
            )

        if features.data_completeness >= 0.7:
            reasons.append(
                f"Data completeness is strong ({features.data_completeness:.0%})"
            )

        if features.has_revenue:
            reasons.append("Revenue generation is evident")

        reinforcing = [
            o for o in observations
            if o.category not in ("signal_conflict",) and o.confidence >= 0.6
        ]
        if len(reinforcing) >= 3:
            reasons.append(
                f"{len(reinforcing)} reinforcing observations with confidence >= 0.6"
            )

        return reasons[:5]

    @staticmethod
    def _collect_reasons_against(
        scores: list[ScoreResult],
        assessments: list[DimensionAssessment],
        observations: list[Observation],
        features: ExtractedFeatures,
    ) -> list[str]:
        """Collect primary reasons reducing conviction."""
        reasons: list[str] = []

        low_scores = [s for s in scores if s.score < 45.0]
        for s in sorted(low_scores, key=lambda x: x.score)[:3]:
            label = _get_dimension_label(s.dimension)
            reasons.append(f"{label} scores {s.score:.0f}/100 — below midpoint")

        low_conf_assessments = [
            a for a in assessments if a.confidence < 0.3
        ]
        for a in low_conf_assessments[:2]:
            label = _get_dimension_label(a.dimension)
            reasons.append(
                f"Low confidence ({a.confidence:.2f}) in {label} assessment"
            )

        conflicts = [o for o in observations if o.category == "signal_conflict"]
        if conflicts:
            reasons.append(
                f"{len(conflicts)} conflicting signal(s) detected"
            )

        if features.data_completeness < 0.3:
            reasons.append(
                f"Very low data completeness ({features.data_completeness:.0%})"
            )

        risk_fields = [
            features.market_risk,
            features.founder_risk,
            features.execution_risk,
            features.product_risk,
            features.technology_risk,
            features.business_model_risk,
            features.traction_risk,
        ]
        total_risk = sum(len(r) for r in risk_fields if r)
        if total_risk >= 5:
            reasons.append(f"High risk signal count ({total_risk} signals)")

        return reasons[:5]

    @staticmethod
    def _collect_top_positive_signals(
        scores: list[ScoreResult],
        assessments: list[DimensionAssessment],
        observations: list[Observation],
    ) -> list[str]:
        """Collect the highest-impact positive signals."""
        signals: list[str] = []

        for s in sorted(scores, key=lambda x: x.score, reverse=True)[:3]:
            if s.score >= 55.0:
                label = _get_dimension_label(s.dimension)
                signals.append(f"{label} score: {s.score:.1f}/100")

        high_conf = [
            a for a in assessments
            if a.confidence >= 0.5 and a.supporting_observations
        ]
        for a in sorted(
            high_conf, key=lambda x: x.confidence, reverse=True
        )[:2]:
            label = _get_dimension_label(a.dimension)
            signals.append(
                f"{label} assessment confidence: {a.confidence:.2f}"
            )

        return signals[:5]

    @staticmethod
    def _collect_top_negative_signals(
        scores: list[ScoreResult],
        assessments: list[DimensionAssessment],
        observations: list[Observation],
    ) -> list[str]:
        """Collect the highest-impact negative signals."""
        signals: list[str] = []

        for s in sorted(scores, key=lambda x: x.score)[:3]:
            if s.score < 50.0:
                label = _get_dimension_label(s.dimension)
                signals.append(f"{label} score: {s.score:.1f}/100")

        conflicts = [o for o in observations if o.category == "signal_conflict"]
        for o in conflicts[:2]:
            signals.append(f"Signal conflict: {o.statement}")

        low_conf = [
            a for a in assessments if a.confidence < 0.35
        ]
        for a in low_conf[:2]:
            label = _get_dimension_label(a.dimension)
            signals.append(
                f"Low {label} confidence: {a.confidence:.2f}"
            )

        return signals[:5]

    @staticmethod
    def _collect_missing_information(
        features: ExtractedFeatures,
    ) -> list[str]:
        """Identify information gaps that would improve conviction."""
        gaps: list[str] = []

        if not features.industry:
            gaps.append("Industry classification")
        if not features.business_model:
            gaps.append("Business model details")
        if not features.funding_stage:
            gaps.append("Funding stage")
        if not features.geography:
            gaps.append("Geographic market")
        if not features.technology_stack:
            gaps.append("Technology stack")
        if features.founder_profile_count == 0:
            gaps.append("Founder profiles")
        if not features.has_pitch_deck:
            gaps.append("Pitch deck")
        if features.has_revenue is None:
            gaps.append("Revenue status")

        return gaps[:5]

    @staticmethod
    def _build_why_not_higher(
        category: DecisionCategory,
        composite_score: float,
        next_threshold: float,
        scores: list[ScoreResult],
        confidence: list[ConfidenceAssessment],
    ) -> str:
        """Explain why the decision is not the next higher category."""
        if category == DecisionCategory.STRONG_INVEST:
            return "Already at the highest decision category."

        margin = next_threshold - composite_score
        reasons: list[str] = []

        reasons.append(
            f"Composite score ({composite_score:.1f}) is {margin:.1f} points "
            f"below the {category.value} threshold ({next_threshold:.0f})."
        )

        low_scores = [s for s in scores if s.score < 50.0]
        if low_scores:
            dims = [_get_dimension_label(s.dimension) for s in low_scores[:2]]
            reasons.append(
                f"Below-midpoint scores in: {', '.join(dims)}."
            )

        avg_conf = (
            sum(c.confidence for c in confidence) / len(confidence)
            if confidence
            else 0.0
        )
        if avg_conf < 0.5:
            reasons.append(
                f"Average confidence ({avg_conf:.2f}) is below the 0.50 threshold."
            )

        return " ".join(reasons)

    @staticmethod
    def _build_key_evidence_summary(
        positive: list[str],
        negative: list[str],
    ) -> str:
        """Summarize the most influential evidence."""
        parts: list[str] = []
        if positive:
            parts.append(f"Strongest positive: {positive[0]}")
        if negative:
            parts.append(f"Strongest negative: {negative[0]}")
        if not parts:
            return "Insufficient evidence to identify key signals."
        return " ".join(parts)

    @staticmethod
    def _collect_information_that_could_change(
        category: DecisionCategory,
        features: ExtractedFeatures,
        observations: list[Observation],
    ) -> list[str]:
        """Identify specific information that could change the decision."""
        info: list[str] = []

        if category in (DecisionCategory.PASS, DecisionCategory.INVESTIGATE_FURTHER):
            if features.data_completeness < 0.5:
                info.append(
                    "More complete data (pitch deck, founder profiles, metrics)"
                )
            if not features.has_revenue:
                info.append("Revenue or traction evidence")
            if features.founder_profile_count == 0:
                info.append("Founder background and track record")
        elif category == DecisionCategory.WATCH:
            if features.data_completeness < 0.7:
                info.append("Additional data to increase confidence")
            low_dims = [
                s.dimension for s in []
                if s.score < 50
            ]
            if low_dims:
                info.append(
                    "Improved performance in underperforming dimensions"
                )
        elif category == DecisionCategory.INVEST:
            info.append("Stronger traction metrics")
            info.append("Additional customer validation")

        conflicts = [o for o in observations if o.category == "signal_conflict"]
        if conflicts:
            info.append("Resolution of conflicting signals")

        return info[:4]

    @staticmethod
    def _build_confidence_explanation(
        confidence: list[ConfidenceAssessment],
        features: ExtractedFeatures,
    ) -> str:
        """Explain overall confidence in the decision."""
        if not confidence:
            return (
                "No confidence assessments available. "
                "Decision is based on limited signal data."
            )

        avg_conf = sum(c.confidence for c in confidence) / len(confidence)
        dc = features.data_completeness

        if avg_conf >= 0.7 and dc >= 0.7:
            return (
                f"High confidence ({avg_conf:.2f}) supported by strong "
                f"data completeness ({dc:.0%})."
            )
        elif avg_conf >= 0.5:
            return (
                f"Moderate confidence ({avg_conf:.2f}). "
                f"Data completeness is {dc:.0%}."
            )
        elif avg_conf >= 0.3:
            return (
                f"Low-moderate confidence ({avg_conf:.2f}). "
                f"Some dimensions have sparse evidence."
            )
        else:
            return (
                f"Low confidence ({avg_conf:.2f}). "
                f"Decision should be treated as preliminary. "
                f"Data completeness is {dc:.0%}."
            )


def _get_dimension_label(dimension_value: str) -> str:
    """Map a dimension value string to its human-readable label."""
    for dim in AnalysisDimension:
        if dim.value == dimension_value:
            return DIMENSION_LABELS.get(dim, dimension_value)
    return dimension_value
