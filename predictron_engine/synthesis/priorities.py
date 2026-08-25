"""Deterministic recommendation prioritization (Sprint 6C).

Takes the concatenated recommendations produced by the recommendation
strategies and produces a single deduplicated, globally ordered,
capped list.

Rules:
  - Merge duplicates by (category, normalized action).
  - Merging never recomputes confidence: it keeps the maximum of the
    merged confidences and unions citations / supporting observations /
    provenance in first-seen order.
  - Global ordering is a pure arithmetic function of the existing
    priority label, the existing confidence, and the existing
    expected_confidence assigned during Sprint 6B calibration.
  - Sorting is stable; equal scores keep insertion order.
"""

from __future__ import annotations

import re

from predictron_engine.models.report import Recommendation

_PRIORITY_WEIGHTS: dict[str, float] = {
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
}

_CATEGORY_TIE_BREAK: dict[str, float] = {
    "due_diligence": 3.0,
    "risk": 2.0,
    "opportunity": 1.0,
}

_DEFAULT_CATEGORY_TIE_BREAK = 0.0

# Score composition weights (documented, deterministic, no hidden state).
_WEIGHT_PRIORITY = 100.0
_WEIGHT_CONFIDENCE = 20.0
_WEIGHT_EXPECTED_CONFIDENCE = 10.0

_MAX_PRIORITIZED = 8

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_action(action: str) -> str:
    """Normalize an action string into a dedup key component."""
    normalized = _WHITESPACE_RE.sub(" ", action.strip().lower())
    return normalized.rstrip(".")


def _dedup_key(rec: Recommendation) -> tuple[str, str]:
    return (
        rec.category.strip().lower(),
        _normalize_action(rec.action),
    )


def _priority_weight(priority: str) -> float:
    return _PRIORITY_WEIGHTS.get(priority.strip().lower(), _PRIORITY_WEIGHTS["medium"])


def _merge(first: Recommendation, second: Recommendation) -> Recommendation:
    """Merge a duplicate into the kept recommendation.

    Deterministic merge policy:
      - priority: highest wins
      - confidence: max (no recomputation)
      - rationale: kept rationale, second appended when different
      - citations / observations / assessments: union in first-seen order
      - action_items: union in first-seen order
      - expected_confidence / expected_uncertainty: max / max
    """
    priority = first.priority
    if _priority_weight(second.priority) > _priority_weight(first.priority):
        priority = second.priority

    rationale = first.rationale
    if second.rationale and second.rationale != first.rationale:
        rationale = f"{first.rationale} {second.rationale}" if first.rationale else second.rationale

    citations: list = []
    seen_claims: set[tuple[str, str, str]] = set()
    for citation in [*first.citations, *second.citations]:
        key = (citation.claim, citation.domain, citation.category)
        if key not in seen_claims:
            seen_claims.add(key)
            citations.append(citation)

    observations: list = []
    seen_obs: set[int] = set()
    for observation in [*first.supporting_observations, *second.supporting_observations]:
        obs_id = id(observation)
        if obs_id not in seen_obs:
            seen_obs.add(obs_id)
            observations.append(observation)

    assessments: list = []
    seen_asmt: set[int] = set()
    for assessment in [*first.supporting_assessments, *second.supporting_assessments]:
        asmt_id = id(assessment)
        if asmt_id not in seen_asmt:
            seen_asmt.add(asmt_id)
            assessments.append(assessment)

    action_items = list(dict.fromkeys([*first.action_items, *second.action_items]))

    expected_confidence = first.expected_confidence
    if second.expected_confidence is not None:
        expected_confidence = (
            second.expected_confidence
            if expected_confidence is None
            else max(expected_confidence, second.expected_confidence)
        )
    expected_uncertainty = first.expected_uncertainty
    if second.expected_uncertainty is not None:
        expected_uncertainty = (
            second.expected_uncertainty
            if expected_uncertainty is None
            else max(expected_uncertainty, second.expected_uncertainty)
        )

    updates: dict = {
        "priority": priority,
        "confidence": max(first.confidence, second.confidence),
        "rationale": rationale,
        "supporting_observations": observations,
        "supporting_assessments": assessments,
        "citations": citations,
        "action_items": action_items,
        "expected_confidence": expected_confidence,
        "expected_uncertainty": expected_uncertainty,
    }
    if second.recommended_action and not first.recommended_action:
        updates["recommended_action"] = second.recommended_action
    return first.model_copy(update=updates)


def _rank_score(rec: Recommendation) -> tuple[float, int]:
    """Deterministic descending sort key from existing fields only."""
    score = (
        _priority_weight(rec.priority) * _WEIGHT_PRIORITY
        + rec.confidence * _WEIGHT_CONFIDENCE
        + (rec.expected_confidence or 0.0) * _WEIGHT_EXPECTED_CONFIDENCE
        + _CATEGORY_TIE_BREAK.get(
            rec.category.strip().lower(), _DEFAULT_CATEGORY_TIE_BREAK
        )
    )
    # Round to kill float noise so ordering is fully deterministic.
    return (-round(score, 6),)


def prioritize_recommendations(
    recommendations: list[Recommendation],
    *,
    max_count: int = _MAX_PRIORITIZED,
) -> list[Recommendation]:
    """Deduplicate, globally prioritize, cap, and rank recommendations.

    Returns new Recommendation instances (originals untouched) with the
    ``rank`` field set to a 1-based position on the final capped list.
    Citations, provenance (via supporting observations), confidence,
    and Sprint 6B uncertainty fields are preserved.
    """
    merged: dict[tuple[str, str], Recommendation] = {}
    order: list[tuple[str, str]] = []
    for rec in recommendations:
        key = _dedup_key(rec)
        if key in merged:
            merged[key] = _merge(merged[key], rec)
        else:
            merged[key] = rec
            order.append(key)

    ranked = sorted((merged[key] for key in order), key=_rank_score)
    capped = ranked[:max_count]

    prioritized: list[Recommendation] = []
    for index, rec in enumerate(capped, start=1):
        prioritized.append(rec.model_copy(update={"rank": index}))
    return prioritized
