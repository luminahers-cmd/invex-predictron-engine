"""Investment Readiness computation.

Deterministic synthesis of all pipeline signals into a structured
investment readiness assessment. Combines scores, observations,
assessments, and cross-signal relationships into a single model
with per-dimension contributions, strengths, concerns, and gaps.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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

_DIMENSION_WEIGHTS: dict[str, float] = {
    AnalysisDimension.MARKET_OPPORTUNITY.value: 0.20,
    AnalysisDimension.FOUNDER_QUALITY.value: 0.18,
    AnalysisDimension.PRODUCT_STRENGTH.value: 0.15,
    AnalysisDimension.BUSINESS_MODEL_VIABILITY.value: 0.15,
    AnalysisDimension.TRACTION_SIGNALS.value: 0.15,
    AnalysisDimension.COMPETITIVE_POSITION.value: 0.10,
    AnalysisDimension.TEAM_EXECUTION.value: 0.07,
}

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

    total_weight = sum(
        _DIMENSION_WEIGHTS.get(d, 0.05)
        for d in dimension_contributions
        if d != "investment_thesis"
    )
    if total_weight == 0.0:
        total_weight = 1.0

    readiness_score = 0.0
    for dim, contrib in dimension_contributions.items():
        if dim == "investment_thesis":
            continue
        w = _DIMENSION_WEIGHTS.get(dim, 0.05)
        readiness_score += contrib * (w / total_weight)

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

    key_strengths = _collect_strengths(observations, assessments, score_map)
    key_concerns = _collect_concerns(observations, assessments, score_map)
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


def _score_to_level(score: float) -> str:
    for threshold, level in _LEVEL_THRESHOLDS:
        if score >= threshold:
            return level
    return "needs_data"


def _collect_strengths(
    observations: list[Observation],
    assessments: list[DimensionAssessment],
    score_map: dict[str, float],
) -> list[str]:
    strengths: list[str] = []

    for dim in AnalysisDimension:
        dim_val = dim.value
        score = score_map.get(dim_val, 50.0)
        if score >= 60.0:
            label = DIMENSION_LABELS.get(dim, dim_val)
            strengths.append(f"{label} scores {score:.0f}/100")

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
) -> list[str]:
    concerns: list[str] = []

    for dim in AnalysisDimension:
        dim_val = dim.value
        score = score_map.get(dim_val, 50.0)
        if score < 50.0:
            label = DIMENSION_LABELS.get(dim, dim_val)
            concerns.append(f"{label} scores {score:.0f}/100 (below midpoint)")

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
    }

    for obs in cross_obs:
        pair = dimension_pairs.get(obs.category, ("unknown", "unknown"))
        rel_type = "conflicting" if obs.category == "signal_conflict" else "reinforcing"

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
