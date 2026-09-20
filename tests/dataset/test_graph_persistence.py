"""Tests for knowledge-graph persistence and canonical serialization."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from predictron_engine.dataset.graph import (
    CompanyKnowledgeGraphBuilder,
    KnowledgeGraph,
    NodeType,
    graph_from_dict,
    graph_key,
    graph_to_dict,
    read_graph,
    write_graph,
)
from predictron_engine.dataset.graph.persistence import (
    GRAPH_SCHEMA_VERSION,
    canonical_graph_json,
)
from tests.dataset.graph_helpers import small_dataset


def _graph() -> KnowledgeGraph:
    return CompanyKnowledgeGraphBuilder().build(small_dataset()).graph


def _edge_tuples(graph: KnowledgeGraph) -> set[tuple[str, str, str]]:
    return {
        (edge.edge_type.value, edge.source_id, edge.target_id)
        for edge in graph.edges()
    }


class TestSerialization:
    def test_graph_to_dict_round_trips(self) -> None:
        graph = _graph()
        restored = graph_from_dict(graph_to_dict(graph))
        assert canonical_graph_json(restored) == canonical_graph_json(graph)
        assert [n.node_id for n in restored.nodes()] == [
            n.node_id for n in graph.nodes()
        ]
        assert _edge_tuples(restored) == _edge_tuples(graph)

    def test_schema_version_recorded(self) -> None:
        payload = graph_to_dict(_graph())
        assert payload["schema_version"] == GRAPH_SCHEMA_VERSION

    def test_canonical_json_is_ordered_and_compact(self) -> None:
        graph = _graph()
        blob = canonical_graph_json(graph)
        parsed = json.loads(blob)
        assert parsed["node_count"] == 32
        assert parsed["edge_count"] == 36
        assert isinstance(parsed["nodes"], list)
        assert isinstance(parsed["edges"], list)

    def test_canonical_json_of_identical_graphs_equal(self) -> None:
        g1 = _graph()
        g2 = CompanyKnowledgeGraphBuilder().build(small_dataset()).graph
        assert canonical_graph_json(g1) == canonical_graph_json(g2)


class TestFilePersistence:
    def test_write_and_read_round_trip(self) -> None:
        graph = _graph()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"
            write_graph(graph, path)
            restored = read_graph(path)
            assert canonical_graph_json(restored) == canonical_graph_json(
                graph
            )
            assert restored is not None
            assert graph_key(restored) == graph_key(graph)

    def test_read_missing_file_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            assert read_graph(Path(tmp) / "missing.json") is None


class TestRestoredGraphUsable:
    def test_restored_graph_supports_traversal(self) -> None:
        from predictron_engine.dataset.graph import GraphQueries

        graph = _graph()
        restored = graph_from_dict(graph_to_dict(graph))
        queries = GraphQueries(restored)
        neighbors = queries.neighbors("company:a+d")
        assert neighbors
        assert restored.node("company:a+d") is not None
        assert restored.node_count == 32
        companies = [n.node_id for n in restored.nodes_of_type(NodeType.COMPANY)]
        assert len(companies) == 5
        assert restored.validate() == []
        assert all(
            node.sources for node in restored.nodes()
        )
        assert all(edge.sources for edge in restored.edges())
