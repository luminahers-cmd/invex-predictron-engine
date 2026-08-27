"""Tests for Sprint 8 — Explanation Consistency.

Validates that reasoning traces are consistent across runs,
that confidence evolution is monotonically bounded, and that
rationales are deterministic and meaningful.
"""

from __future__ import annotations

import pytest

from predictron_engine.reasoning.trace import (
    build_reasoning_trace,
    ReasoningTrace,
)
from predictron_engine.reasoning.contradiction_graph import build_contradiction_graph
from predictron_engine.reasoning.adaptive_budget import compute_reasoning_budget
from predictron_engine.evidence.evidence_models import EvidenceItem
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.report import Observation, ScoreResult


def _make_obs(
    dimension: str = "market",
    category: str = "market_context",
    statement: str = "Test",
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


class TestExplanationConsistency:
    """Validate that explanations are deterministic and consistent."""

    def test_trace_determinism_across_runs(self) -> None:
        """Same observations produce identical traces every time."""
        observations = [
            _make_obs(category="market_context", confidence=0.8, importance=0.7),
            _make_obs(category="risk", confidence=0.3, importance=0.6),
            _make_obs(category="strength", confidence=0.9, importance=0.8),
        ]
        evidence = [
            EvidenceItem(domain="industry", category="m", statement="E", source="s"),
        ]
        t1 = build_reasoning_trace(observations, evidence)
        t2 = build_reasoning_trace(observations, evidence)
        t3 = build_reasoning_trace(observations, evidence)

        assert t1.final_rationale == t2.final_rationale == t3.final_rationale
        assert t1.overall_confidence == t2.overall_confidence == t3.overall_confidence
        assert (
            len(t1.strongest_supporting)
            == len(t2.strongest_supporting)
            == len(t3.strongest_supporting)
        )

    def test_confidence_evolution_bounded(self) -> None:
        """All confidence evolution values must be in [0, 1]."""
        observations = [_make_obs(confidence=0.7)]
        trace = build_reasoning_trace(observations, [])
        for stage in trace.confidence_evolution:
            assert 0.0 <= stage.confidence <= 1.0

    def test_rationale_always_nonempty(self) -> None:
        """Rationale must always be generated, even for empty inputs."""
        trace = build_reasoning_trace([], [])
        assert trace.final_rationale != ""
        assert len(trace.final_rationale) > 10

    def test_rationale_references_evidence_when_present(self) -> None:
        """Rationale should reference evidence when available."""
        observations = [
            _make_obs(category="market_context", confidence=0.9, importance=0.8, statement="Strong market position"),
        ]
        trace = build_reasoning_trace(observations, [])
        assert "Strongest positive signal" in trace.final_rationale

    def test_strongest_evidence_ranked_deterministically(self) -> None:
        """Same observations always produce the same ranking order."""
        observations = [
            _make_obs(category="strength", confidence=0.5, importance=0.3, statement="Weak"),
            _make_obs(category="strength", confidence=0.95, importance=0.9, statement="Strong"),
            _make_obs(category="strength", confidence=0.7, importance=0.6, statement="Medium"),
        ]
        t1 = build_reasoning_trace(observations, [])
        t2 = build_reasoning_trace(observations, [])

        if t1.strongest_supporting:
            for e1, e2 in zip(t1.strongest_supporting, t2.strongest_supporting):
                assert e1.statement == e2.statement
                assert e1.rank == e2.rank

    def test_contradiction_graph_consistency(self) -> None:
        """Same observations always produce the same contradiction graph."""
        observations = [
            _make_obs(category="strength", confidence=0.9),
            _make_obs(category="risk", confidence=0.2),
        ]
        g1 = build_contradiction_graph(observations)
        g2 = build_contradiction_graph(observations)
        assert g1.contradiction_intensity == g2.contradiction_intensity
        assert g1.conflicting_count == g2.conflicting_count

    def test_budget_determinism(self) -> None:
        """Same features produce the same budget allocation."""
        features = ExtractedFeatures(
            industry="SaaS",
            data_completeness=0.75,
            founder_profile_count=2,
        )
        evidence = [
            EvidenceItem(domain="industry", category="m", statement="E", source="s"),
        ]
        b1 = compute_reasoning_budget(features, evidence)
        b2 = compute_reasoning_budget(features, evidence)
        assert b1.budget_fraction == b2.budget_fraction
        assert b1.skipped_rules == b2.skipped_rules

    def test_empty_trace_has_contradiction_graph(self) -> None:
        """Even empty observations produce a valid (empty) graph."""
        trace = build_reasoning_trace([], [])
        assert trace.contradiction_graph is not None
        assert trace.contradiction_graph.edge_count == 0

    def test_trace_with_many_observations_scales(self) -> None:
        """Trace handles large observation sets without error."""
        observations = [
            _make_obs(
                dimension=f"dim_{i % 5}",
                category="strength" if i % 3 == 0 else "risk",
                confidence=0.3 + (i * 0.05) % 0.6,
                importance=0.4 + (i * 0.03) % 0.5,
                statement=f"Observation {i}",
            )
            for i in range(20)
        ]
        trace = build_reasoning_trace(observations, [])
        assert trace.total_observations == 20
        assert len(trace.strongest_supporting) <= 5
        assert len(trace.strongest_opposing) <= 5

    def test_confidence_evolution_monotonic_bounds(self) -> None:
        """Confidence at each stage must stay within [0, 1]."""
        observations = [
            _make_obs(confidence=0.1),
            _make_obs(confidence=0.9),
        ]
        scores = [
            ScoreResult(dimension="market", score=40.0, rationale="Low"),
            ScoreResult(dimension="team", score=80.0, rationale="High"),
        ]
        trace = build_reasoning_trace(observations, [], scores=scores)
        for stage in trace.confidence_evolution:
            assert 0.0 <= stage.confidence <= 1.0, (
                f"Stage {stage.stage} has out-of-range confidence: {stage.confidence}"
            )
