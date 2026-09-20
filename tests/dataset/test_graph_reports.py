"""Tests for knowledge-graph reports."""

from __future__ import annotations

from predictron_engine.dataset.graph import (
    CompanyKnowledgeGraphBuilder,
    KnowledgeGraph,
    build_graph_report,
    build_graph_statistics,
    build_relationship_summary,
    graph_key,
)
from tests.dataset.graph_helpers import small_dataset


def _graph() -> KnowledgeGraph:
    return CompanyKnowledgeGraphBuilder().build(small_dataset()).graph


class TestGraphReport:
    def test_report_covers_all_components(self) -> None:
        graph = _graph()
        report = build_graph_report(graph)
        assert report["overview"]["graph_key"] == graph_key(graph)
        assert report["overview"]["node_count"] == 32
        assert report["overview"]["edge_count"] == 36

    def test_statistics_subset(self) -> None:
        stats = build_graph_statistics(_graph())
        assert stats["connected_components"]["count"] == 2

    def test_relationship_summary_rows_are_sorted(self) -> None:
        summary = build_relationship_summary(_graph())
        rows = summary["relationships"]
        names = [r["edge_type"] for r in rows]
        assert names == sorted(names)
        counts = {r["edge_type"]: r["count"] for r in rows}
        assert counts["LOCATED_IN"] == 11


class TestGraphStatistics:
    def test_empty_graph_statistics(self) -> None:
        empty = KnowledgeGraph()
        stats = build_graph_statistics(empty)
        assert stats["node_counts"] == {}
        assert stats["edge_counts"] == {}
        assert stats["connected_components"]["count"] == 0


class TestRelationshipSummary:
    def test_summary_includes_all_edge_types_with_zero(self) -> None:
        summary = build_relationship_summary(_graph())
        by_type = {r["edge_type"] for r in summary["relationships"]}
        assert len(by_type) == 12
        zero_rows = [r for r in summary["relationships"] if r["count"] == 0]
        assert len(zero_rows) == 1  # RELATED_TO is the only zero edge type
        assert zero_rows[0]["edge_type"] == "RELATED_TO"
