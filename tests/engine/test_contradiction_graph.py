"""Tests for Sprint 8 — Contradiction Graph.

Validates deterministic contradiction detection, intensity scoring,
and dominant conflict identification.
"""

from __future__ import annotations

from predictron_engine.models.report import Observation
from predictron_engine.reasoning.contradiction_graph import (
    ContradictionEdge,
    build_contradiction_graph,
)


def _make_obs(
    dimension: str = "market",
    category: str = "strength",
    statement: str = "Test statement",
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


class TestContradictionGraph:
    """Tests for build_contradiction_graph function."""

    def test_empty_observations(self) -> None:
        graph = build_contradiction_graph([])
        assert graph.edge_count == 0
        assert graph.conflicting_count == 0
        assert graph.dominant_conflict is None

    def test_single_observation(self) -> None:
        graph = build_contradiction_graph([_make_obs()])
        assert graph.edge_count == 0

    def test_two_supporting_observations(self) -> None:
        obs_a = _make_obs(category="strength", confidence=0.8, importance=0.7)
        obs_b = _make_obs(category="strength", confidence=0.75, importance=0.65)
        graph = build_contradiction_graph([obs_a, obs_b])
        assert graph.edge_count == 1
        assert graph.supporting_count >= 1 or graph.neutral_count >= 1

    def test_two_conflicting_observations(self) -> None:
        obs_a = _make_obs(category="strength", confidence=0.9, importance=0.8)
        obs_b = _make_obs(category="risk", confidence=0.2, importance=0.7)
        graph = build_contradiction_graph([obs_a, obs_b])
        assert graph.edge_count == 1
        assert graph.conflicting_count >= 1

    def test_graph_is_deterministic(self) -> None:
        obs_list = [
            _make_obs(category="strength", confidence=0.8, importance=0.7),
            _make_obs(category="risk", confidence=0.3, importance=0.6),
            _make_obs(category="strength", confidence=0.7, importance=0.5),
        ]
        g1 = build_contradiction_graph(obs_list)
        g2 = build_contradiction_graph(obs_list)
        assert g1.edge_count == g2.edge_count
        assert g1.conflicting_count == g2.conflicting_count
        assert g1.supporting_count == g2.supporting_count
        assert g1.contradiction_intensity == g2.contradiction_intensity

    def test_per_dimension_summaries(self) -> None:
        obs_list = [
            _make_obs(dimension="market", category="strength", confidence=0.8),
            _make_obs(dimension="market", category="risk", confidence=0.3),
            _make_obs(dimension="team", category="strength", confidence=0.7),
        ]
        graph = build_contradiction_graph(obs_list)
        assert len(graph.per_dimension) >= 1

    def test_dominant_conflict_identified(self) -> None:
        obs_list = [
            _make_obs(category="strength", confidence=0.9, importance=0.9),
            _make_obs(category="risk", confidence=0.1, importance=0.8),
        ]
        graph = build_contradiction_graph(obs_list)
        if graph.conflicting_count > 0:
            assert graph.dominant_conflict is not None
            assert graph.dominant_conflict.intensity > 0.0

    def test_graph_serialization(self) -> None:
        obs_list = [
            _make_obs(category="strength", confidence=0.8),
            _make_obs(category="risk", confidence=0.3),
        ]
        graph = build_contradiction_graph(obs_list)
        d = graph.to_dict()
        assert "total_edges" in d
        assert "contradiction_intensity" in d
        assert "per_dimension" in d

    def test_same_dimension_different_categories_conflict(self) -> None:
        obs_a = _make_obs(dimension="market", category="growth", confidence=0.9)
        obs_b = _make_obs(dimension="market", category="decline", confidence=0.2)
        graph = build_contradiction_graph([obs_a, obs_b])
        assert graph.conflicting_count >= 1

    def test_same_category_same_dimension_support(self) -> None:
        obs_a = _make_obs(dimension="market", category="strength", confidence=0.8)
        obs_b = _make_obs(dimension="market", category="strength", confidence=0.75)
        graph = build_contradiction_graph([obs_a, obs_b])
        assert graph.supporting_count >= 1

    def test_different_dimensions_are_neutral(self) -> None:
        obs_a = _make_obs(dimension="market", category="strength")
        obs_b = _make_obs(dimension="team", category="strength")
        graph = build_contradiction_graph([obs_a, obs_b])
        assert graph.neutral_count >= 1

    def test_many_observations_scale(self) -> None:
        obs_list = [
            _make_obs(
                dimension=f"dim_{i % 3}",
                category="strength" if i % 2 == 0 else "risk",
                confidence=0.3 + (i * 0.05) % 0.7,
                importance=0.4 + (i * 0.03) % 0.5,
            )
            for i in range(10)
        ]
        graph = build_contradiction_graph(obs_list)
        assert graph.edge_count == 45  # C(10,2) = 45


class TestContradictionEdge:
    """Tests for ContradictionEdge properties."""

    def test_conflict_detection(self) -> None:
        edge = ContradictionEdge(
            observation_a_idx=0, observation_b_idx=1,
            dimension="test", relationship="conflicting",
            intensity=0.8, reason="test",
        )
        assert edge.is_conflict is True
        assert edge.is_support is False

    def test_support_detection(self) -> None:
        edge = ContradictionEdge(
            observation_a_idx=0, observation_b_idx=1,
            dimension="test", relationship="supporting",
            intensity=0.6, reason="test",
        )
        assert edge.is_support is True
        assert edge.is_conflict is False
