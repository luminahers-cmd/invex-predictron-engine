"""Tests for deterministic knowledge graph construction and provenance."""

from __future__ import annotations

import random

from predictron_engine.dataset.graph import (
    CompanyKnowledgeGraphBuilder,
    EdgeType,
    NodeType,
    graph_key,
)
from predictron_engine.dataset.graph.builder import (
    GraphBuildReport,
    GraphBuildResult,
)
from predictron_engine.dataset.graph.persistence import canonical_graph_json
from predictron_engine.dataset.outcomes import OutcomeRecord, StartupOutcome
from tests.dataset.graph_helpers import COMPANY_IDS, small_dataset


def _build() -> GraphBuildResult:
    return CompanyKnowledgeGraphBuilder().build(small_dataset())


class TestConstruction:
    def test_build_returns_report_and_graph(self) -> None:
        result = _build()
        assert isinstance(result.report, GraphBuildReport)
        assert result.graph.node_count == 32
        assert result.graph.edge_count == 36

    def test_resolved_company_count(self) -> None:
        assert _build().report.identity_count == 5

    def test_company_node_ids_are_record_id_based(self) -> None:
        graph = _build().graph
        expected = {
            COMPANY_IDS["alpha"],
            COMPANY_IDS["beta"],
            COMPANY_IDS["gamma"],
            COMPANY_IDS["us_bank"],
            COMPANY_IDS["uk_bank"],
        }
        companies = {
            node.node_id
            for node in graph.nodes_of_type(NodeType.COMPANY)
        }
        assert companies == expected

    def test_attribute_nodes_cover_all_types(self) -> None:
        graph = _build().graph
        present = {
            node_type
            for node_type in NodeType
            if graph.nodes_of_type(node_type)
        }
        assert set(NodeType) == present

    def test_node_type_counts(self) -> None:
        graph = _build().graph
        counts = {
            node_type: len(graph.nodes_of_type(node_type))
            for node_type in NodeType
        }
        assert counts[NodeType.COMPANY] == 5
        assert counts[NodeType.FOUNDER] == 2
        assert counts[NodeType.INVESTOR] == 1
        assert counts[NodeType.ORGANIZATION] == 1
        assert counts[NodeType.INDUSTRY] == 3
        assert counts[NodeType.TECHNOLOGY] == 2
        assert counts[NodeType.PRODUCT] == 3
        assert counts[NodeType.COUNTRY] == 3
        assert counts[NodeType.STATE] == 3
        assert counts[NodeType.CITY] == 3
        assert counts[NodeType.DOMAIN] == 5
        assert counts[NodeType.IDENTIFIER] == 1

    def test_edge_type_counts(self) -> None:
        graph = _build().graph
        counts = {edge_type: 0 for edge_type in EdgeType}
        for edge in graph.edges():
            counts[edge.edge_type] += 1
        assert counts[EdgeType.FOUNDED_BY] == 2
        assert counts[EdgeType.INVESTED_BY] == 2
        assert counts[EdgeType.LOCATED_IN] == 11
        assert counts[EdgeType.OPERATES_IN] == 5
        assert counts[EdgeType.USES_TECHNOLOGY] == 4
        assert counts[EdgeType.BUILDS_PRODUCT] == 3
        assert counts[EdgeType.HAS_DOMAIN] == 5
        assert counts[EdgeType.HAS_IDENTIFIER] == 1
        assert counts[EdgeType.ALIAS_OF] == 1
        assert counts[EdgeType.ACQUIRED_BY] == 1
        assert counts[EdgeType.SUBSIDIARY_OF] == 1
        assert counts[EdgeType.RELATED_TO] == 0


class TestDeterminism:
    def test_shuffled_input_produces_identical_graph(self) -> None:
        records = small_dataset()
        base = CompanyKnowledgeGraphBuilder().build(records)
        for _ in range(5):
            shuffled = records[:]
            random.Random(7).shuffle(shuffled)
            other = CompanyKnowledgeGraphBuilder().build(shuffled)
            assert other.report.graph_key == base.report.graph_key
            assert canonical_graph_json(other.graph) == canonical_graph_json(
                base.graph
            )

    def test_graph_key_is_hash_of_canonical_json(self) -> None:
        import hashlib

        result = _build()
        expected = hashlib.sha256(
            canonical_graph_json(result.graph).encode("utf-8")
        ).hexdigest()
        assert graph_key(result.graph) == expected


class TestProvenance:
    def test_every_node_and_edge_has_sources(self) -> None:
        result = _build()
        graph = result.graph
        assert graph.validate() == []
        allowed = {record.record_id for record in small_dataset()}
        for node in graph.nodes():
            assert node.sources
            assert set(node.sources) <= allowed
        for edge in graph.edges():
            assert edge.sources
            assert set(edge.sources) <= allowed

    def test_shared_attribute_carries_combined_provenance(self) -> None:
        graph = _build().graph
        industry = graph.node("industry:fintech")
        assert industry is not None
        assert set(industry.sources) == {"a", "c", "d"}

    def test_merged_company_carries_all_record_sources(self) -> None:
        graph = _build().graph
        alpha = graph.node(COMPANY_IDS["alpha"])
        assert alpha is not None
        assert set(alpha.sources) == {"a", "d"}

    def test_acquisition_edge_pinned_to_resolved_company(self) -> None:
        graph = _build().graph
        edges = graph.out_edges(COMPANY_IDS["gamma"], EdgeType.ACQUIRED_BY)
        assert [edge.target_id for edge in edges] == [COMPANY_IDS["alpha"]]

    def test_unresolved_parent_becomes_organization(self) -> None:
        graph = _build().graph
        edges = graph.out_edges(
            COMPANY_IDS["gamma"], EdgeType.SUBSIDIARY_OF
        )
        assert edges
        org = graph.node(edges[0].target_id)
        assert org is not None
        assert org.node_type == NodeType.ORGANIZATION

    def test_alias_edge_links_conflicting_global_banks(self) -> None:
        graph = _build().graph
        assert graph.has_edge(
            COMPANY_IDS["us_bank"], EdgeType.ALIAS_OF, COMPANY_IDS["uk_bank"]
        )


class TestBuildSelection:
    def test_single_record_build_is_self_contained(self) -> None:
        records = small_dataset()
        graph = CompanyKnowledgeGraphBuilder().build([records[0]]).graph
        assert graph.node("company:a") is not None
        assert graph.node_count > 0

    def test_build_honors_include_outcomes_flag(self) -> None:
        records = small_dataset()
        outcome = OutcomeRecord(
            record_id="a",
            outcome=StartupOutcome(investors=["Horizon Ventures", "Accel"]),
        )
        outcomes = {"a": outcome}
        full = CompanyKnowledgeGraphBuilder().build(records, outcomes=outcomes)
        full_labels = {
            "Horizon Ventures"
            for edge in full.graph.edges()
            if edge.edge_type == EdgeType.INVESTED_BY
            and (node := full.graph.node(edge.target_id)) is not None
            and node.label == "Horizon Ventures"
        }
        assert full_labels == {"Horizon Ventures"}
        without = CompanyKnowledgeGraphBuilder(
            include_outcomes=False
        ).build(records, outcomes=outcomes)
        skipped_labels = {
            without.graph.node(edge.target_id).label
            for edge in without.graph.edges()
            if edge.edge_type == EdgeType.INVESTED_BY
            and without.graph.node(edge.target_id) is not None
        }
        assert "Horizon Ventures" not in skipped_labels
