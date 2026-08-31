"""Investment Readiness computation.

Deterministic synthesis of all pipeline signals into a structured
investment readiness assessment. Combines scores, observations,
assessments, and cross-signal relationships into a single model
with per-dimension contributions, strengths, concerns, and gaps.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from predictron_engine.evaluation.aggregation import (
    DEFAULT_DIMENSION_WEIGHT,
    DIMENSION_WEIGHTS,
    MAX_DIMENSION_WEIGHT,
    weighted_composite,
)
from predictron_engine.knowledge.concepts import DIMENSION_LABELS, AnalysisDimension
from predictron_engine.models.report import (
    InvestmentReadiness,
    Observation,
    ScoreResult,
    SignalRelationship,
)

if TYPE_CHECKING:
    from predictron_engine.evaluation.evaluation_models import DimensionAssessment
    from predictron_engine.models.extracted_features import ExtractedFeatures

logger = logging.getLogger(__name__)

_LEVEL_THRESHOLDS: list[tuple[float, str]] = [
    (80.0, "investment_ready"),
    (65.0, "strong"),
    (50.0, "moderate"),
    (35.0, "developing"),
    (15.0, "early"),
    (0.0, "needs_data"),
]

_GAP_FIELDS: list[tuple[str, str]] = [
    ("industry", "Industry classification"),
    ("business_model", "Business model"),
    ("funding_stage", "Funding stage"),
    ("geography", "Geographic market"),
    ("technology_stack", "Technology stack"),
]


def compute_investment_readiness(
    features: ExtractedFeatures,
    observations: list[Observation],
    scores: list[ScoreResult],
    assessments: list[DimensionAssessment],
) -> InvestmentReadiness:
    """Compute investment readiness from all pipeline signals.

    Pure deterministic computation — no randomness, no external calls.
    """
    score_map = {s.dimension: s.score for s in scores}
    obs_by_dim: dict[str, list[Observation]] = {}
    for obs in observations:
        obs_by_dim.setdefault(obs.dimension, []).append(obs)

    dimension_contributions = _compute_dimension_contributions(
        score_map, obs_by_dim, assessments
    )

    # H3 — missing-data calibration: a dimension that carries no evidence
    # (no score, no observations, no assessment) is *missing*, not *average*.
    # It is excluded from the weighted readiness composite so a data-poor
    # startup no longer looks "middling" at ~50 through fabricated neutral
    # contributions.
    assessed_dims = _assessed_dimensions(score_map, obs_by_dim, assessments)

    readiness_score = weighted_composite(
        dimension_contributions,
        DIMENSION_WEIGHTS,
        present_keys=assessed_dims,
    )

    cross_obs = obs_by_dim.get("investment_thesis", [])
    reinforcing = [o for o in cross_obs if o.category not in ("signal_conflict",)]
    conflicts = [o for o in cross_obs if o.category == "signal_conflict"]

    if reinforcing:
        avg_reinforcing_conf = (
            sum(o.confidence for o in reinforcing) / len(reinforcing)
        )
        reinforcement_bonus = min(
            len(reinforcing) * 1.5 * avg_reinforcing_conf, 6.0
        )
        readiness_score += reinforcement_bonus

    if conflicts:
        avg_conflict_importance = (
            sum(o.importance for o in conflicts) / len(conflicts)
        )
        conflict_penalty = min(
            len(conflicts) * 2.0 * avg_conflict_importance, 8.0
        )
        readiness_score -= conflict_penalty

    readiness_score = max(0.0, min(100.0, readiness_score))

    readiness_level = _score_to_level(readiness_score)

    key_strengths = _collect_strengths(observations, assessments, score_map, features)
    key_concerns = _collect_concerns(observations, assessments, score_map, features)
    gaps = _identify_gaps(features)

    signal_relationships = _build_signal_relationships(cross_obs)
    reinforcing_count = sum(
        1 for r in signal_relationships if r.relationship_type == "reinforcing"
    )
    conflicting_count = sum(
        1 for r in signal_relationships if r.relationship_type == "conflicting"
    )

    summary = _build_summary(
        readiness_level, readiness_score, len(key_strengths),
        len(key_concerns), reinforcing_count, conflicting_count,
    )

    return InvestmentReadiness(
        readiness_score=round(readiness_score, 2),
        readiness_level=readiness_level,
        key_strengths=key_strengths,
        key_concerns=key_concerns,
        dimension_contributions=dimension_contributions,
        signal_relationships=signal_relationships,
        reinforcing_count=reinforcing_count,
        conflicting_count=conflicting_count,
        gaps=gaps,
        summary=summary,
    )


def _compute_dimension_contributions(
    score_map: dict[str, float],
    obs_by_dim: dict[str, list[Observation]],
    assessments: list[DimensionAssessment],
) -> dict[str, float]:
    """Compute a 0-100 contribution per dimension."""
    contributions: dict[str, float] = {}
    assessment_conf = {a.dimension: a.confidence for a in assessments}

    for dim in AnalysisDimension:
        dim_val = dim.value
        score = score_map.get(dim_val, 50.0)
        obs_list = obs_by_dim.get(dim_val, [])

        avg_obs_conf = 0.0
        if obs_list:
            avg_obs_conf = sum(o.confidence for o in obs_list) / len(obs_list)

        assess_c = assessment_conf.get(dim_val, 0.0)

        contrib = score * 0.6 + avg_obs_conf * 100.0 * 0.25 + assess_c * 100.0 * 0.15
        contributions[dim_val] = round(max(0.0, min(100.0, contrib)), 2)

    return contributions


def _assessed_dimensions(
    score_map: dict[str, float],
    obs_by_dim: dict[str, list[Observation]],
    assessments: list[DimensionAssessment],
) -> set[str]:
    """Return the set of dimensions that actually carry evidence.

    H3 — a dimension is considered *assessed* when it has a score, at least
    one observation, or a dimension assessment.  Dimensions absent from all
    three are *missing* and must not be averaged in as if they were neutral;
    excluding them is what makes "unknown" mean unknown rather than average.
    """
    assessed: set[str] = set(score_map.keys())
    assessed |= set(obs_by_dim.keys())
    assessed |= {a.dimension for a in assessments}
    return assessed


def _score_to_level(score: float) -> str:
    for threshold, level in _LEVEL_THRESHOLDS:
        if score >= threshold:
            return level
    return "needs_data"


def _collect_strengths(
    observations: list[Observation],
    assessments: list[DimensionAssessment],
    score_map: dict[str, float],
    features: ExtractedFeatures | None = None,
) -> list[str]:
    strengths: list[str] = []

    for dim in AnalysisDimension:
        dim_val = dim.value
        score = score_map.get(dim_val, 50.0)
        if score >= 60.0:
            label = DIMENSION_LABELS.get(dim, dim_val)
            strengths.append(f"{label} scores {score:.0f}/100")

    if features is not None:
        if features.arr_usd is not None and features.arr_usd >= 10_000_000:
            strengths.append(
                f"ARR of ${features.arr_usd / 1_000_000:.1f}M"
            )
        if features.nrr_pct is not None and features.nrr_pct >= 110:
            strengths.append(f"NRR of {features.nrr_pct:.0f}%")
        if features.growth_rate_pct is not None and features.growth_rate_pct >= 50:
            strengths.append(f"Growth rate of {features.growth_rate_pct:.0f}%")
        if (
            features.cac_usd is not None
            and features.ltv_usd is not None
            and features.ltv_usd / max(features.cac_usd, 1) >= 5.0
        ):
            strengths.append(
                f"LTV/CAC of {features.ltv_usd / features.cac_usd:.1f}x"
            )

    high_conf = [
        a for a in assessments
        if a.confidence >= 0.6 and a.supporting_observations
    ]
    for a in sorted(high_conf, key=lambda x: x.confidence, reverse=True)[:2]:
        if a.summary and a.summary not in strengths:
            strengths.append(a.summary)

    return strengths[:5]


def _collect_concerns(
    observations: list[Observation],
    assessments: list[DimensionAssessment],
    score_map: dict[str, float],
    features: ExtractedFeatures | None = None,
) -> list[str]:
    concerns: list[str] = []

    for dim in AnalysisDimension:
        dim_val = dim.value
        score = score_map.get(dim_val, 50.0)
        if score < 50.0:
            label = DIMENSION_LABELS.get(dim, dim_val)
            concerns.append(f"{label} scores {score:.0f}/100 (below midpoint)")

    if features is not None:
        if features.runway_months is not None and features.runway_months < 6:
            concerns.append(
                f"Runway of {features.runway_months} months — critical funding risk"
            )
        if features.nrr_pct is not None and features.nrr_pct < 90:
            concerns.append(
                f"NRR of {features.nrr_pct:.0f}% — net revenue contraction"
            )
        if features.churn_rate_pct is not None and features.churn_rate_pct > 10:
            concerns.append(
                f"Churn rate of {features.churn_rate_pct:.1f}% — severe retention issues"
            )
        if (
            features.cac_usd is not None
            and features.ltv_usd is not None
            and features.ltv_usd / max(features.cac_usd, 1) < 1.0
        ):
            concerns.append("LTV/CAC below 1.0x — unsustainable unit economics")

    conflict_obs = [
        o for o in observations
        if o.category == "signal_conflict"
    ]
    for o in conflict_obs[:2]:
        concerns.append(o.statement)

    low_conf = [
        a for a in assessments
        if a.confidence < 0.3
    ]
    for a in low_conf[:1]:
        concerns.append(f"Low confidence ({a.confidence:.2f}) in {a.dimension} assessment")

    return concerns[:5]


def _identify_gaps(features: ExtractedFeatures) -> list[str]:
    gaps: list[str] = []
    for field_name, label in _GAP_FIELDS:
        value = getattr(features, field_name, None)
        if value is None or value == [] or value == "":
            gaps.append(f"Missing {label}")
    return gaps


def _build_signal_relationships(
    cross_obs: list[Observation],
) -> list[SignalRelationship]:
    """Convert cross-signal observations into structured relationships."""
    relationships: list[SignalRelationship] = []
    dimension_pairs: dict[str, tuple[str, str]] = {
        "market_product_fit": ("market_opportunity", "product_strength"),
        "team_market_alignment": ("founder_quality", "market_opportunity"),
        "business_model_market_fit": ("business_model_viability", "market_opportunity"),
        "traction_consistency": ("traction_signals", "business_model_viability"),
        "competition_differentiation": ("competitive_position", "product_strength"),
        "technology_market_alignment": ("product_strength", "market_opportunity"),
        "founder_domain_alignment": ("founder_quality", "market_opportunity"),
        "scale_evidence": ("traction_signals", "product_strength"),
        "signal_conflict": ("unknown", "unknown"),
        # Quantitative cross-signal categories
        "rev_per_customer_enterprise": ("traction_signals", "business_model_viability"),
        "rev_per_customer_mid_market": ("traction_signals", "business_model_viability"),
        "rev_per_customer_smb": ("traction_signals", "business_model_viability"),
        "rev_per_employee_exceptional": ("traction_signals", "team_execution"),
        "rev_per_employee_strong": ("traction_signals", "team_execution"),
        "rev_per_employee_acceptable": ("traction_signals", "team_execution"),
        "rev_per_employee_low": ("traction_signals", "team_execution"),
        "growth_burn_excellent": ("traction_signals", "business_model_viability"),
        "growth_burn_good": ("traction_signals", "business_model_viability"),
        "growth_burn_moderate": ("traction_signals", "business_model_viability"),
        "growth_burn_concerning": ("traction_signals", "business_model_viability"),
        "burn_runway_critical": ("traction_signals", "market_opportunity"),
        "burn_runway_warning": ("traction_signals", "market_opportunity"),
        "burn_runway_sustainable": ("traction_signals", "market_opportunity"),
        "nrr_churn_reinforcing": ("traction_signals", "product_strength"),
        "nrr_churn_conflict": ("traction_signals", "product_strength"),
        "nrr_churn_masking": ("traction_signals", "product_strength"),
        "nrr_churn_downgrade": ("traction_signals", "product_strength"),
        "funding_efficiency_strong": ("traction_signals", "business_model_viability"),
        "funding_efficiency_good": ("traction_signals", "business_model_viability"),
        "funding_efficiency_moderate": ("traction_signals", "business_model_viability"),
        "funding_efficiency_weak": ("traction_signals", "business_model_viability"),
        "valuation_arr_high": ("traction_signals", "market_opportunity"),
        "valuation_arr_moderate": ("traction_signals", "market_opportunity"),
        "valuation_arr_low": ("traction_signals", "market_opportunity"),
        "valuation_arr_very_low": ("traction_signals", "market_opportunity"),
        "funding_valuation_inconsistent": ("traction_signals", "market_opportunity"),
        "funding_valuation_high_creation": ("traction_signals", "market_opportunity"),
        "customer_count_stage_exceeds": ("traction_signals", "founder_quality"),
        "customer_count_stage_below": ("traction_signals", "founder_quality"),
        "team_revenue_inconsistent": ("team_execution", "traction_signals"),
        "team_revenue_efficient": ("team_execution", "traction_signals"),
        "valuation_weak_traction": ("market_opportunity", "traction_signals"),
        "high_arr_concentrated_customers": ("traction_signals", "business_model_viability"),
        "high_burn_low_growth": ("traction_signals", "business_model_viability"),
        "funding_without_execution": ("traction_signals", "founder_quality"),
        "arr_exceeds_burn": ("traction_signals", "business_model_viability"),
        "arr_partial_burn_coverage": ("traction_signals", "business_model_viability"),
        "arr_burn_gap": ("traction_signals", "business_model_viability"),
    }

    _conflict_categories: set[str] = {
        "signal_conflict",
        "nrr_churn_conflict",
        "burn_runway_critical",
        "growth_burn_concerning",
        "funding_efficiency_weak",
        "funding_valuation_inconsistent",
        "customer_count_stage_below",
        "team_revenue_inconsistent",
        "valuation_weak_traction",
        "high_arr_concentrated_customers",
        "high_burn_low_growth",
        "funding_without_execution",
        "arr_burn_gap",
        "rev_per_employee_low",
        "nrr_churn_masking",
    }
    for obs in cross_obs:
        pair = dimension_pairs.get(obs.category, ("unknown", "unknown"))
        rel_type = "conflicting" if obs.category in _conflict_categories else "reinforcing"

        relationships.append(
            SignalRelationship(
                source_dimension=pair[0],
                target_dimension=pair[1],
                relationship_type=rel_type,
                description=obs.statement,
                confidence=obs.confidence,
                source_observations=[obs.category],
            )
        )

    return relationships


def _build_summary(
    level: str,
    score: float,
    strength_count: int,
    concern_count: int,
    reinforcing: int,
    conflicting: int,
) -> str:
    parts = [
        f"Investment readiness assessed as {level} (score: {score:.1f}/100)."
    ]
    parts.append(
        f"{strength_count} strength(s) and {concern_count} concern(s) identified."
    )
    if reinforcing or conflicting:
        parts.append(
            f"{reinforcing} reinforcing and {conflicting} conflicting "
            f"cross-signal relationship(s) detected."
        )
    return " ".join(parts)


def dimension_contribution(
    dimension: str,
    scores: list[ScoreResult],
    readiness: InvestmentReadiness | None = None,
) -> float:
    """Return the 0-100 readiness contribution for a dimension.

    Reuses existing weighted readiness outputs: the readiness
    ``dimension_contributions`` mapping is preferred, falling back to the
    scored dimension output and finally a neutral 50.0 baseline.
    """
    if readiness is not None and readiness.dimension_contributions:
        contribution = readiness.dimension_contributions.get(dimension)
        if contribution is not None:
            return contribution
    for result in scores:
        if result.dimension == dimension:
            return result.score
    return 50.0


def weighted_gap_magnitude(
    dimension: str,
    scores: list[ScoreResult],
    readiness: InvestmentReadiness | None = None,
) -> float:
    """Normalized [0,1] weighted readiness gap for a dimension.

    Derives the gap from existing weighted readiness outputs: the
    canonical dimension weight scaled by how far the dimension's
    contribution sits below full readiness. Larger gaps come from weaker
    contributions on higher-weighted dimensions. Returns 0.0 when the
    dimension is at full readiness.
    """
    weight = DIMENSION_WEIGHTS.get(dimension, DEFAULT_DIMENSION_WEIGHT)
    contribution = dimension_contribution(dimension, scores, readiness)
    raw_gap = max(0.0, 100.0 - contribution)
    normalized = (raw_gap / 100.0) * (weight / MAX_DIMENSION_WEIGHT)
    return round(normalized, 4)
