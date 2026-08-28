"""Reasoning Trace — explainable reasoning path for auditability (Sprint 8).

Produces a structured reasoning trace that shows:
  - Strongest supporting evidence
  - Strongest opposing evidence
  - Confidence evolution across reasoning stages
  - Final rationale

Key principles:
  - Every trace entry is deterministic
  - Strongest evidence is ranked by importance × confidence
  - Confidence evolution is a time-series of confidence snapshots
  - Final rationale is built from template rules (no generative text)
  - No external dependencies

Public API:
  - :func:`build_reasoning_trace` — build trace from pipeline outputs
  - :class:`ReasoningTrace` — complete reasoning trace container
  - :class:`TraceEntry` — single step in the reasoning trace
  - :class:`ConfidenceEvolution` — confidence snapshot at a reasoning stage
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.report import Observation, ScoreResult

from predictron_engine.reasoning.contradiction_graph import (
    ContradictionGraph,
    DominantConflict,
    build_contradiction_graph,
)


@dataclass(frozen=True)
class TraceEntry:
    """One step in the reasoning trace.

    Each entry represents a reasoning observation with its supporting
    evidence ranked by relevance.
    """

    dimension: str
    category: str
    statement: str
    confidence: float
    importance: float
    source_rule: str
    supporting_evidence: tuple[str, ...] = ()
    rank: int = 0

    @property
    def strength_score(self) -> float:
        """Deterministic strength = importance × confidence."""
        return round(self.importance * self.confidence, 4)


@dataclass(frozen=True)
class ConfidenceEvolution:
    """Confidence snapshot at a reasoning stage.

    Tracks how confidence evolves as evidence is processed.
    """

    stage: str
    confidence: float
    observation_count: int
    evidence_count: int
    contradictions: int


@dataclass
class ReasoningTrace:
    """Complete reasoning trace container.

    Captures the full explainable reasoning path for one analysis run.
    """

    strongest_supporting: tuple[TraceEntry, ...] = ()
    strongest_opposing: tuple[TraceEntry, ...] = ()
    confidence_evolution: tuple[ConfidenceEvolution, ...] = ()
    final_rationale: str = ""
    contradiction_graph: ContradictionGraph | None = None
    dominant_conflict: DominantConflict | None = None
    total_observations: int = 0
    total_evidence: int = 0
    overall_confidence: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Serialize for reporting and API responses."""
        return {
            "strongest_supporting": [
                {
                    "dimension": e.dimension,
                    "statement": e.statement,
                    "confidence": e.confidence,
                    "importance": e.importance,
                    "strength_score": e.strength_score,
                    "source_rule": e.source_rule,
                }
                for e in self.strongest_supporting
            ],
            "strongest_opposing": [
                {
                    "dimension": e.dimension,
                    "statement": e.statement,
                    "confidence": e.confidence,
                    "importance": e.importance,
                    "strength_score": e.strength_score,
                    "source_rule": e.source_rule,
                }
                for e in self.strongest_opposing
            ],
            "confidence_evolution": [
                {
                    "stage": c.stage,
                    "confidence": c.confidence,
                    "observation_count": c.observation_count,
                    "evidence_count": c.evidence_count,
                    "contradictions": c.contradictions,
                }
                for c in self.confidence_evolution
            ],
            "final_rationale": self.final_rationale,
            "dominant_conflict": (
                {
                    "statement_a": self.dominant_conflict.observation_a_statement,
                    "statement_b": self.dominant_conflict.observation_b_statement,
                    "dimension": self.dominant_conflict.dimension,
                    "intensity": self.dominant_conflict.intensity,
                    "explanation": self.dominant_conflict.explanation,
                }
                if self.dominant_conflict
                else None
            ),
            "total_observations": self.total_observations,
            "total_evidence": self.total_evidence,
            "overall_confidence": self.overall_confidence,
        }


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------

_MAX_TRACE_ENTRIES: int = 5


def _rank_observations(
    observations: list[Observation],
    max_entries: int = _MAX_TRACE_ENTRIES,
) -> tuple[TraceEntry, ...]:
    """Rank observations by strength score (importance × confidence)."""
    entries: list[TraceEntry] = []
    for obs in observations:
        entries.append(
            TraceEntry(
                dimension=obs.dimension,
                category=obs.category,
                statement=obs.statement,
                confidence=obs.confidence,
                importance=obs.importance,
                source_rule=obs.source_rule,
                supporting_evidence=tuple(obs.evidence[:3]),
            )
        )

    ranked = sorted(entries, key=lambda e: e.strength_score, reverse=True)
    ranked_with_idx = tuple(
        TraceEntry(
            dimension=e.dimension,
            category=e.category,
            statement=e.statement,
            confidence=e.confidence,
            importance=e.importance,
            source_rule=e.source_rule,
            supporting_evidence=e.supporting_evidence,
            rank=i + 1,
        )
        for i, e in enumerate(ranked[:max_entries])
    )
    return ranked_with_idx


def _rank_by_supporting(
    observations: list[Observation],
) -> tuple[TraceEntry, ...]:
    """Get strongest supporting observations (high confidence + positive category)."""
    positive_categories = {
        "market_context", "team_assessment", "product_assessment",
        "business_model_assessment", "technology_assessment",
        "traction", "competitive_advantage", "strength",
        "opportunity", "reinforcing",
    }
    supporting = [
        o for o in observations
        if o.category.lower() in positive_categories or o.confidence >= 0.6
    ]
    return _rank_observations(supporting)


def _rank_by_opposing(
    observations: list[Observation],
) -> tuple[TraceEntry, ...]:
    """Get strongest opposing observations (risk, conflict, low confidence)."""
    opposing_categories = {
        "risk", "signal_conflict", "decline", "weakness",
        "disadvantage", "threat",
    }
    opposing = [
        o for o in observations
        if o.category.lower() in opposing_categories
        or o.confidence < 0.4
        or o.evidence_conflict_count > 0
    ]
    return _rank_observations(opposing)


# ---------------------------------------------------------------------------
# Confidence evolution
# ---------------------------------------------------------------------------


def _build_confidence_evolution(
    observations: list[Observation],
    evidence: list[EvidenceItem],
    scores: list[ScoreResult],
) -> tuple[ConfidenceEvolution, ...]:
    """Build deterministic confidence evolution timeline.

    Confidence evolves through four stages:
      1. Post-extraction (data completeness proxy)
      2. Post-reasoning (observation confidence)
      3. Post-evaluation (assessment confidence proxy)
      4. Post-scoring (final score-based confidence proxy)
    """
    stages: list[ConfidenceEvolution] = []

    # Stage 1: Post-extraction (no observations yet)
    base_confidence = 0.0
    if observations:
        base_confidence = sum(o.confidence for o in observations) / len(observations) * 0.5
    stages.append(
        ConfidenceEvolution(
            stage="post_extraction",
            confidence=round(min(base_confidence, 1.0), 4),
            observation_count=0,
            evidence_count=len(evidence),
            contradictions=0,
        )
    )

    # Stage 2: Post-reasoning
    obs_conf = (
        sum(o.confidence for o in observations) / len(observations)
        if observations
        else 0.0
    )
    contradictions = sum(o.evidence_conflict_count for o in observations)
    stages.append(
        ConfidenceEvolution(
            stage="post_reasoning",
            confidence=round(obs_conf, 4),
            observation_count=len(observations),
            evidence_count=len(evidence),
            contradictions=contradictions,
        )
    )

    # Stage 3: Post-evaluation (assessments use observations)
    eval_conf = obs_conf * 0.9 + (0.1 if observations else 0.0)
    stages.append(
        ConfidenceEvolution(
            stage="post_evaluation",
            confidence=round(min(eval_conf, 1.0), 4),
            observation_count=len(observations),
            evidence_count=len(evidence),
            contradictions=contradictions,
        )
    )

    # Stage 4: Post-scoring (scores incorporate everything)
    if scores:
        score_spread = max(s.score for s in scores) - min(s.score for s in scores)
        score_factor = 1.0 - (score_spread / 100.0)
        final_conf = eval_conf * 0.7 + score_factor * 0.3
    else:
        final_conf = eval_conf
    stages.append(
        ConfidenceEvolution(
            stage="post_scoring",
            confidence=round(min(final_conf, 1.0), 4),
            observation_count=len(observations),
            evidence_count=len(evidence),
            contradictions=contradictions,
        )
    )

    return tuple(stages)


# ---------------------------------------------------------------------------
# Rationale generation
# ---------------------------------------------------------------------------


def _build_final_rationale(
    supporting: tuple[TraceEntry, ...],
    opposing: tuple[TraceEntry, ...],
    dominant_conflict: DominantConflict | None,
    overall_confidence: float,
) -> str:
    """Build deterministic final rationale from trace components."""
    parts: list[str] = []

    if supporting:
        top = supporting[0]
        parts.append(
            f"Strongest positive signal: {top.statement} "
            f"(importance={top.importance:.2f}, confidence={top.confidence:.2f})"
        )

    if opposing:
        top = opposing[0]
        parts.append(
            f"Strongest concern: {top.statement} "
            f"(importance={top.importance:.2f}, confidence={top.confidence:.2f})"
        )

    if dominant_conflict and dominant_conflict.intensity > 0.3:
        parts.append(
            f"Primary uncertainty driver: {dominant_conflict.explanation} "
            f"(intensity={dominant_conflict.intensity:.2f})"
        )

    if overall_confidence >= 0.75:
        parts.append(f"High overall confidence ({overall_confidence:.2f})")
    elif overall_confidence <= 0.35:
        parts.append(
            f"Low overall confidence ({overall_confidence:.2f}): "
            "recommend additional due diligence"
        )
    else:
        parts.append(f"Moderate overall confidence ({overall_confidence:.2f})")

    return ". ".join(parts) + "."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_reasoning_trace(
    observations: list[Observation],
    evidence: list[EvidenceItem],
    scores: list[ScoreResult] | None = None,
) -> ReasoningTrace:
    """Build the complete reasoning trace from pipeline outputs.

    Parameters
    ----------
    observations:
        Reasoning layer observations (already enriched).
    evidence:
        Evidence items used in the analysis.
    scores:
        Optional scoring results for confidence evolution.
    """
    score_list = scores or []

    supporting = _rank_by_supporting(observations)
    opposing = _rank_by_opposing(observations)
    evolution = _build_confidence_evolution(observations, evidence, score_list)
    graph = build_contradiction_graph(observations)
    dominant = graph.dominant_conflict

    overall_conf = (
        sum(o.confidence for o in observations) / len(observations)
        if observations
        else 0.0
    )

    rationale = _build_final_rationale(
        supporting, opposing, dominant, overall_conf,
    )

    return ReasoningTrace(
        strongest_supporting=supporting,
        strongest_opposing=opposing,
        confidence_evolution=evolution,
        final_rationale=rationale,
        contradiction_graph=graph,
        dominant_conflict=dominant,
        total_observations=len(observations),
        total_evidence=len(evidence),
        overall_confidence=round(overall_conf, 4),
    )
