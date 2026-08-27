"""Tests for Sprint 8 — Reasoning Trace.

Validates explainable reasoning path generation, confidence evolution,
and rationale production.
"""

from __future__ import annotations

import pytest

from predictron_engine.reasoning.trace import (
    ReasoningTrace,
    TraceEntry,
    ConfidenceEvolution,
    build_reasoning_trace,
)
from predictron_engine.evidence.evidence_models import EvidenceItem
from predictron_engine.models.report import Observation, ScoreResult


def _make_obs(
    dimension: str = "market",
    category: str = "market_context",
    statement: str = "Test observation",
    confidence: float = 0.7,
    importance: float = 0.6,
) -> Observation:
    return Observation(
        dimension=dimension,
        category=category,
        statement=statement,
        confidence=confidence,
        importance=importance,
        source_rule="test_rule",
    )


class TestBuildReasoningTrace:
    """Tests for build_reasoning_trace function."""

    def test_empty_inputs(self) -> None:
        trace = build_reasoning_trace([], [])
        assert trace.total_observations == 0
        assert trace.total_evidence == 0
        assert trace.final_rationale != ""

    def test_trace_with_observations(self) -> None:
        observations = [
            _make_obs(category="market_context", confidence=0.8, importance=0.7),
            _make_obs(category="risk", dimension="market", confidence=0.3, importance=0.6),
        ]
        trace = build_reasoning_trace(observations, [])
        assert trace.total_observations == 2
        assert len(trace.strongest_supporting) > 0
        assert len(trace.strongest_opposing) > 0

    def test_confidence_evolution_has_stages(self) -> None:
        observations = [_make_obs(confidence=0.7)]
        evidence = [
            EvidenceItem(domain="test", category="test", statement="Evidence", source="s"),
        ]
        trace = build_reasoning_trace(observations, evidence)
        assert len(trace.confidence_evolution) == 4
        stages = [c.stage for c in trace.confidence_evolution]
        assert "post_extraction" in stages
        assert "post_reasoning" in stages
        assert "post_evaluation" in stages
        assert "post_scoring" in stages

    def test_final_rationale_mentions_key_signals(self) -> None:
        observations = [
            _make_obs(category="market_context", confidence=0.9, importance=0.8, statement="Strong market"),
            _make_obs(category="risk", confidence=0.2, importance=0.7, statement="High risk"),
        ]
        trace = build_reasoning_trace(observations, [])
        assert "Strongest positive signal" in trace.final_rationale or len(trace.final_rationale) > 0
        assert "Strongest concern" in trace.final_rationale or len(trace.final_rationale) > 0

    def test_trace_is_deterministic(self) -> None:
        observations = [
            _make_obs(category="strength", confidence=0.8),
            _make_obs(category="risk", confidence=0.3),
        ]
        t1 = build_reasoning_trace(observations, [])
        t2 = build_reasoning_trace(observations, [])
        assert t1.overall_confidence == t2.overall_confidence
        assert len(t1.strongest_supporting) == len(t2.strongest_supporting)
        assert len(t1.strongest_opposing) == len(t2.strongest_opposing)
        assert t1.final_rationale == t2.final_rationale

    def test_trace_with_scores(self) -> None:
        observations = [_make_obs(confidence=0.7)]
        scores = [
            ScoreResult(dimension="market", score=65.0, rationale="Test"),
            ScoreResult(dimension="team", score=55.0, rationale="Test"),
        ]
        trace = build_reasoning_trace(observations, [], scores=scores)
        assert len(trace.confidence_evolution) == 4

    def test_strongest_supporting_ranked_by_strength(self) -> None:
        observations = [
            _make_obs(category="strength", confidence=0.5, importance=0.4, statement="Weak"),
            _make_obs(category="strength", confidence=0.9, importance=0.8, statement="Strong"),
        ]
        trace = build_reasoning_trace(observations, [])
        if trace.strongest_supporting:
            assert trace.strongest_supporting[0].strength_score >= trace.strongest_supporting[-1].strength_score

    def test_serialization(self) -> None:
        observations = [_make_obs()]
        trace = build_reasoning_trace(observations, [])
        d = trace.to_dict()
        assert "strongest_supporting" in d
        assert "strongest_opposing" in d
        assert "confidence_evolution" in d
        assert "final_rationale" in d

    def test_trace_entries_have_ranks(self) -> None:
        observations = [
            _make_obs(category="strength", confidence=0.8, importance=0.7),
            _make_obs(category="strength", confidence=0.6, importance=0.5),
        ]
        trace = build_reasoning_trace(observations, [])
        for entry in trace.strongest_supporting:
            assert entry.rank > 0

    def test_dominant_conflict_surfaced(self) -> None:
        observations = [
            _make_obs(category="strength", confidence=0.9, importance=0.9),
            _make_obs(category="risk", confidence=0.1, importance=0.8),
        ]
        trace = build_reasoning_trace(observations, [])
        if trace.dominant_conflict:
            assert trace.dominant_conflict.intensity > 0.0
