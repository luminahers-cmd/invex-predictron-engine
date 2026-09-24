"""Deterministic source ranker — Phase 8 Sprint 2.

Ranks candidate research sources for a research topic using fixed,
module-level weights.  The rank is a pure function of source metadata
and the topic:

* **Trust** — how authoritative the source is.
* **Topic coverage** — how broad the source is, boosted when the source
  explicitly prefers the topic.
* **Freshness** — how current the source's data is, scaled by the task's
  freshness requirement.
* **Cost** — lower relative cost scores higher.
* **Structured data** — structured sources score higher.

Ties break deterministically on a stable order (the canonical registry
index when supplied, otherwise the input order).  The same input always
produces the same ordered list — no randomness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from predictron_engine.research.models import (
    ResearchSource,
    SourceRank,
    SourceScore,
)

TRUST_WEIGHT = 0.30
COVERAGE_WEIGHT = 0.25
FRESHNESS_WEIGHT = 0.15
COST_WEIGHT = 0.20
STRUCTURED_WEIGHT = 0.10

RANKING_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "trust": TRUST_WEIGHT,
        "coverage": COVERAGE_WEIGHT,
        "freshness": FRESHNESS_WEIGHT,
        "cost": COST_WEIGHT,
        "structured_data": STRUCTURED_WEIGHT,
    }
)

TOPIC_FIT_COVERAGE_FACTOR = 1.0
NO_TOPIC_FIT_COVERAGE_FACTOR = 0.5

SCORE_PRECISION = 6

FRESHNESS_SHORT_DAYS = 60
FRESHNESS_NOMINAL_DAYS = 180
FRESHNESS_LONG_DAYS = 365
FRESHNESS_SHORT_SCALE = 1.0
FRESHNESS_NOMINAL_SCALE = 0.9
FRESHNESS_LONG_SCALE = 0.8
FRESHNESS_EXTENDED_SCALE = 0.7

_DEFAULT_FRESHNESS_REQUIREMENT_DAYS = 180

# (score, source, input position) for one candidate.
_ScoredCandidate = tuple[SourceScore, ResearchSource, int]


def freshness_priority_scale(freshness_requirement_days: int) -> float:
    """Scale how much a topic's freshness requirement weights freshness.

    Shorter freshness requirements (fast-moving topics) place more
    emphasis on a source's freshness; longer requirements down-weight it
    so the total score stays within ``[0, 1]``.
    """
    if freshness_requirement_days < 1:
        raise ValueError(
            "freshness_requirement_days must be greater than zero"
        )
    if freshness_requirement_days <= FRESHNESS_SHORT_DAYS:
        return FRESHNESS_SHORT_SCALE
    if freshness_requirement_days <= FRESHNESS_NOMINAL_DAYS:
        return FRESHNESS_NOMINAL_SCALE
    if freshness_requirement_days <= FRESHNESS_LONG_DAYS:
        return FRESHNESS_LONG_SCALE
    return FRESHNESS_EXTENDED_SCALE


def topic_fit_factor(source: ResearchSource, topic_id: str) -> float:
    """Return ``1.0`` when ``source`` prefers ``topic_id``, else ``0.0``."""
    return 1.0 if topic_id in source.preferred_topics else 0.0


def compute_source_score(
    source: ResearchSource,
    topic_id: str,
    *,
    freshness_requirement_days: int = _DEFAULT_FRESHNESS_REQUIREMENT_DAYS,
) -> SourceScore:
    """Compute the deterministic score for ``source`` against ``topic_id``.

    The score is a weighted sum of normalized components in ``[0, 1]``.
    Topic fit doubles the effective coverage contribution so preferred
    sources outrank equally-scoring non-preferred ones.
    """
    fit = topic_fit_factor(source, topic_id)
    coverage = source.coverage_score * (
        TOPIC_FIT_COVERAGE_FACTOR if fit else NO_TOPIC_FIT_COVERAGE_FACTOR
    )
    freshness = source.freshness_score * freshness_priority_scale(
        freshness_requirement_days
    )
    cost = 1.0 - source.relative_cost
    structured = 1.0 if source.supports_structured_data else 0.0
    total = (
        source.trust_score * TRUST_WEIGHT
        + coverage * COVERAGE_WEIGHT
        + freshness * FRESHNESS_WEIGHT
        + cost * COST_WEIGHT
        + structured * STRUCTURED_WEIGHT
    )
    return SourceScore(
        trust=round(source.trust_score, SCORE_PRECISION),
        coverage=round(coverage, SCORE_PRECISION),
        freshness=round(freshness, SCORE_PRECISION),
        cost=round(cost, SCORE_PRECISION),
        structured_data=structured,
        topic_fit=fit,
        total=round(total, SCORE_PRECISION),
    )


def rank_sources(
    sources: Sequence[ResearchSource],
    topic_id: str,
    *,
    freshness_requirement_days: int = _DEFAULT_FRESHNESS_REQUIREMENT_DAYS,
    registry_order: Mapping[str, int] | None = None,
) -> tuple[SourceRank, ...]:
    """Rank ``sources`` for ``topic_id`` deterministically.

    Parameters
    ----------
    sources:
        Candidate sources (any order).
    topic_id:
        Topic the sources are ranked against.
    freshness_requirement_days:
        The task's freshness requirement; scales the freshness component.
    registry_order:
        Optional mapping of ``identifier -> canonical position`` used as
        the tie-breaker.  When ``None`` the input position is used.
    """
    if freshness_requirement_days < 1:
        raise ValueError("freshness_requirement_days must be greater than zero")

    def order_index(position: int, source: ResearchSource) -> int:
        if registry_order is None:
            return position
        return registry_order.get(source.identifier, position)

    scored: list[_ScoredCandidate] = []
    for position, source in enumerate(sources):
        score = compute_source_score(
            source,
            topic_id,
            freshness_requirement_days=freshness_requirement_days,
        )
        scored.append((score, source, position))
    scored.sort(
        key=lambda item: (
            -item[0].total,
            order_index(item[2], item[1]),
        )
    )
    return tuple(
        SourceRank(rank=position + 1, source=source, score=score)
        for position, (score, source, _) in enumerate(scored)
    )
