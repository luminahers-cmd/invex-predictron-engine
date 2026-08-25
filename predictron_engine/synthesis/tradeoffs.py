"""Deterministic trade-off analysis (Sprint 6C).

Connects existing strengths and concerns per dimension into explicit
trade-offs. Every side of a trade-off comes from already-produced
evaluator outputs (ScoreResult, DimensionAssessment, Observation,
InvestmentReadiness). No information is invented; confidence is the
minimum of the contributing observation confidences.
"""

from __future__ import annotations

from predictron_engine.models.report import (
    DimensionAssessment,
    EvidenceCitation,
    InvestmentReadiness,
    Observation,
    ScoreResult,
    TradeOff,
)

# Score bands mirror the readiness/scoring conventions:
#   >= STRONG_SCORE  -> strength side
#   <= CONCERN_SCORE -> concern side
STRONG_SCORE = 60.0
CONCERN_SCORE = 50.0

# Minimum observation confidence for an observation to be cited.
_MIN_OBSERVATION_CONFIDENCE = 0.5

_MAX_TRADE_OFFS = 5

_STRENGTH_DOMINANT = "strength_dominant"
_CONCERN_DOMINANT = "concern_dominant"
_BALANCED = "balanced"


def _label(dimension: str) -> str:
    return dimension.replace("_", " ").title()


def _pick_observations(
    observations: list[Observation],
    dimension: str,
) -> list[Observation]:
    """Observations for a dimension, best first, above confidence floor."""
    matching = [
        obs
        for obs in observations
        if obs.dimension == dimension
        and obs.confidence >= _MIN_OBSERVATION_CONFIDENCE
        and bool(obs.statement)
    ]
    return sorted(
        matching,
        key=lambda obs: (-(obs.importance * obs.confidence),),
    )


def _net_assessment(score: float) -> str:
    if score >= STRONG_SCORE:
        return _STRENGTH_DOMINANT
    if score < CONCERN_SCORE:
        return _CONCERN_DOMINANT
    return _BALANCED


def analyze_trade_offs(
    scores: list[ScoreResult],
    assessments: list[DimensionAssessment],
    observations: list[Observation],
    readiness: InvestmentReadiness | None = None,
    *,
    max_trade_offs: int = _MAX_TRADE_OFFS,
) -> list[TradeOff]:
    """Build deterministic trade-offs from evaluator outputs.

    A trade-off is emitted for a dimension when both sides exist:
      - strength: score >= 60 or the readiness key_strengths mention it
      - concern: score < 50, a low-confidence assessment (< 0.35), a
        conflicting signal relationship on the dimension, or the
        readiness key_concerns mention it

    Sorted by polarization (|score - 50| descending) then dimension name;
    capped at ``max_trade_offs``.
    """
    score_by_dimension = {score.dimension: score for score in scores}
    assessment_by_dimension = {
        assessment.dimension: assessment for assessment in assessments
    }

    concerns_by_dimension: dict[str, list[str]] = {}
    strengths_by_dimension: dict[str, list[str]] = {}
    if readiness is not None:
        for statement in readiness.key_concerns:
            lowered = statement.lower()
            for dimension in score_by_dimension:
                keyword = dimension.split("_")[0]
                if keyword and keyword in lowered:
                    concerns_by_dimension.setdefault(dimension, []).append(statement)
                    break
        for statement in readiness.key_strengths:
            lowered = statement.lower()
            for dimension in score_by_dimension:
                keyword = dimension.split("_")[0]
                if keyword and keyword in lowered:
                    strengths_by_dimension.setdefault(dimension, []).append(statement)
                    break

    conflict_dimensions = set()
    if readiness is not None:
        for relationship in readiness.signal_relationships:
            if relationship.relationship_type == "conflicting":
                conflict_dimensions.add(relationship.source_dimension)
                conflict_dimensions.add(relationship.target_dimension)

    trade_offs: list[TradeOff] = []
    for dimension in sorted(score_by_dimension):
        score_result = score_by_dimension[dimension]
        observations_for_dim = _pick_observations(observations, dimension)

        strength_parts: list[str] = []
        concern_parts: list[str] = []
        cited: list[Observation] = []

        # Readiness-derived statements apply on their side regardless of
        # which band the raw score falls into.
        if score_result.score >= STRONG_SCORE:
            strength_parts.append(
                f"{_label(dimension)} scores {score_result.score:.1f}/100"
            )
        elif score_result.score < CONCERN_SCORE:
            concern_parts.append(
                f"{_label(dimension)} scores {score_result.score:.1f}/100"
            )
        strength_parts.extend(strengths_by_dimension.get(dimension, []))
        concern_parts.extend(concerns_by_dimension.get(dimension, []))

        if observations_for_dim and (strength_parts or concern_parts):
            cited.append(observations_for_dim[0])

        assessment = assessment_by_dimension.get(dimension)
        if assessment is not None and assessment.confidence < 0.35:
            concern_parts.append(
                f"{_label(dimension)} assessment confidence is only "
                f"{assessment.confidence:.2f}"
            )
            if assessment.supporting_observations:
                candidate = assessment.supporting_observations[0]
                if candidate not in cited:
                    cited.append(candidate)

        if dimension in conflict_dimensions:
            concern_parts.append(
                f"{_label(dimension)} has conflicting cross-signal relationships"
            )

        if not strength_parts or not concern_parts:
            continue

        statements = [obs.statement for obs in cited]
        citations: list[EvidenceCitation] = []
        seen_claims: set[tuple[str, str, str]] = set()
        for obs in cited:
            for citation in obs.citations:
                key = (citation.claim, citation.domain, citation.category)
                if key not in seen_claims:
                    seen_claims.add(key)
                    citations.append(citation)

        confidences = [obs.confidence for obs in cited]
        if assessment is not None:
            confidences.append(assessment.confidence)

        trade_offs.append(
            TradeOff(
                dimension=dimension,
                strength="; ".join(strength_parts),
                concern="; ".join(concern_parts),
                strength_score=score_result.score
                if score_result.score >= STRONG_SCORE
                else None,
                concern_score=score_result.score
                if score_result.score < CONCERN_SCORE
                else None,
                supporting_evidence=statements,
                supporting_citations=citations,
                net_assessment=_net_assessment(score_result.score),
                confidence=round(min(confidences), 4) if confidences else 0.0,
            )
        )

    def _polarization(trade_off: TradeOff) -> float:
        deltas = [
            abs(value - CONCERN_SCORE)
            for value in (trade_off.strength_score, trade_off.concern_score)
            if value is not None
        ]
        return max(deltas, default=0.0)

    trade_offs.sort(key=lambda t: (-_polarization(t), t.dimension))
    return trade_offs[:max_trade_offs]
