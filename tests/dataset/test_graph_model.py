"""Tests for the knowledge graph model vocabulary and store merging."""

from __future__ import annotations

from predictron_engine.dataset.graph import (
    EdgeType,
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    NodeType,
    canonical_edge_key,
)


class TestEdgeTypes:
    def test_all_required_relationships_defined(self) -> None:
        required = {
            "FOUNDED_BY",
            "INVESTED_BY",
            "LOCATED_IN",
            "OPERATES_IN",
            "USES_TECHNOLOGY",
            "BUILDS_PRODUCT",
            "HAS_DOMAIN",
            "HAS_IDENTIFIER",
            "ALIAS_OF",
            "ACQUIRED_BY",
            "SUBSIDIARY_OF",
            "RELATED_TO",
        }
        assert required == {edge.value for edge in EdgeType}

    def test_all_required_node_types_defined(self) -> None:
        required = {
            "company",
            "founder",
            "investor",
            "organization",
            "industry",
            "technology",
            "product",
            "country",
            "state",
            "city",
            "domain",
            "identifier",
        }
        assert required == {node.value for node in NodeType}


class TestCanonicalEdgeKey:
    def test_directed_order_preserved(self) -> None:
        assert canonical_edge_key("a", EdgeType.FOUNDED_BY, "b") == (
            "a",
            "FOUNDED_BY",
            "b",
        )
        assert canonical_edge_key("b", EdgeType.FOUNDED_BY, "a") == (
            "b",
            "FOUNDED_BY",
            "a",
        )

    def test_symmetric_order_normalized(self) -> None:
        assert canonical_edge_key("b", EdgeType.ALIAS_OF, "a") == (
            "a",
            "ALIAS_OF",
            "b",
        )
        assert canonical_edge_key("a", EdgeType.ALIAS_OF, "b") == (
            "a",
            "ALIAS_OF",
            "b",
        )
        assert canonical_edge_key("y", EdgeType.RELATED_TO, "x") == (
            "x",
            "RELATED_TO",
            "y",
        )


class TestStoreMerging:
    def test_add_node_dedupes_and_unions_provenance(self) -> None:
        graph = KnowledgeGraph()
        first = GraphNode(
            node_id="industry:fintech",
            node_type=NodeType.INDUSTRY,
            label="FinTech",
            sources=["a"],
        )
        second = GraphNode(
            node_id="industry:fintech",
            node_type=NodeType.INDUSTRY,
            label="FinTech",
            sources=["b", "a"],
        )
        graph.add_node(first)
        stored = graph.add_node(second)
        assert stored.sources == ["a", "b"]
        assert graph.node_count == 1

    def test_add_edge_canonicalizes_symmetric_type(self) -> None:
        graph = KnowledgeGraph()
        graph.add_node(
            GraphNode("company:a", NodeType.COMPANY, "A", sources=["a"])
        )
        graph.add_node(
            GraphNode("company:b", NodeType.COMPANY, "B", sources=["b"])
        )
        graph.add_edge(
            GraphEdge(
                edge_type=EdgeType.ALIAS_OF,
                source_id="company:b",
                target_id="company:a",
                sources=["b"],
            )
        )
        assert graph.has_edge("company:a", EdgeType.ALIAS_OF, "company:b")
        assert graph.has_edge("company:b", EdgeType.ALIAS_OF, "company:a")
        assert graph.edge_count == 1

    def test_edges_between_returns_all_types(self) -> None:
        graph = KnowledgeGraph()
        for node_id, label in (("company:a", "A"), ("company:b", "B")):
            graph.add_node(
                GraphNode(node_id, NodeType.COMPANY, label, sources=["x"])
            )
        graph.add_edge(
            GraphEdge(
                EdgeType.ALIAS_OF, "company:a", "company:b", sources=["x"]
            )
        )
        graph.add_edge(
            GraphEdge(
                EdgeType.RELATED_TO, "company:a", "company:b", sources=["x"]
            )
        )
        kinds = {
            edge.edge_type.value
            for edge in graph.edges_between("company:a", "company:b")
        }
        assert kinds == {"ALIAS_OF", "RELATED_TO"}

    def test_validate_flags_missing_provenance(self) -> None:
        graph = KnowledgeGraph()
        graph.add_node(
            GraphNode("company:a", NodeType.COMPANY, "A", sources=[])
        )
        graph.add_node(
            GraphNode("industry:x", NodeType.INDUSTRY, "x", sources=["a"])
        )
        graph.add_edge(
            GraphEdge(
                EdgeType.OPERATES_IN, "company:a", "industry:x", sources=[]
            )
        )
        graph.add_edge(
            GraphEdge(
                EdgeType.OPERATES_IN, "company:a", "industry:y", sources=["a"]
            )
        )
        issues = graph.validate()
        assert any("without provenance" in issue for issue in issues)
        assert any("dangling edge target" in issue for issue in issues)


class TestNodeDictionaries:
    def test_node_to_dict_is_deterministic(self) -> None:
        node = GraphNode(
            node_id="founder:ada lovelace",
            node_type=NodeType.FOUNDER,
            label="Ada Lovelace",
            properties={"variants": ["Ada", "Ada Lovelace"]},
            sources=["b", "a"],
        )
        payload = node.to_dict()
        assert payload["sources"] == ["a", "b"]
        assert payload["properties"] == {
            "variants": ["Ada", "Ada Lovelace"]
        }
