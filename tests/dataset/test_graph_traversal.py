"""Tests for graph traversal: neighbors, shortest path, components."""

from __future__ import annotations

from predictron_engine.dataset.graph import (
    CompanyKnowledgeGraphBuilder,
    EdgeType,
    GraphQueries,
)
from tests.dataset.graph_helpers import COMPANY_IDS, small_dataset


def _queries() -> GraphQueries:
    graph = CompanyKnowledgeGraphBuilder().build(small_dataset()).graph
    return GraphQueries(graph)


class TestNeighbors:
    def test_neighbors_include_all_node_kinds(self) -> None:
        queries = _queries()
        neighbors = queries.neighbors(COMPANY_IDS["alpha"])
        labels = {entry["node"]["label"] for entry in neighbors}
        assert {"fintech"} <= labels  # industry
        assert {"pytorch", "python"} <= labels  # technology
        assert {"Accel"} <= labels  # investor
        assert {"Ada Lovelace"} <= labels  # founder
        assert {"Alpha Platform"} <= labels  # product
        assert {"US"} <= labels  # country
        assert {"Austin"} <= labels  # city
        assert {"alpha.ai"} <= labels  # domain
        assert {"000123"} <= labels  # identifier

    def test_neighbors_filtered_by_edge_type(self) -> None:
        queries = _queries()
        investors = {
            entry["node_id"]
            for entry in queries.neighbors(
                COMPANY_IDS["alpha"], edge_type=EdgeType.INVESTED_BY
            )
        }
        assert investors == {"investor:accel"}

    def test_neighbors_bidirectional_includes_symlink(self) -> None:
        queries = _queries()
        related = queries.neighbors(
            COMPANY_IDS["gamma"],
            edge_type=EdgeType.ACQUIRED_BY,
            bidirectional=True,
        )
        ids = {entry["node_id"] for entry in related}
        assert COMPANY_IDS["alpha"] in ids

    def test_neighbors_unknown_node_is_empty(self) -> None:
        queries = _queries()
        assert queries.neighbors("company:nope") == []


class TestShortestPath:
    def test_path_through_shared_hub_node(self) -> None:
        queries = _queries()
        path = queries.shortest_path(
            COMPANY_IDS["alpha"], COMPANY_IDS["beta"]
        )
        assert path is not None
        assert path[0] == COMPANY_IDS["alpha"]
        assert path[-1] == COMPANY_IDS["beta"]
        assert len(path) == 3
        assert path[1] in {"country:us", "technology:python"}

    def test_direct_company_edge_via_acquisition(self) -> None:
        queries = _queries()
        path = queries.shortest_path(
            COMPANY_IDS["alpha"], COMPANY_IDS["gamma"]
        )
        assert path is not None
        assert set(path) == {COMPANY_IDS["alpha"], COMPANY_IDS["gamma"]}

    def test_beta_to_gamma_only_share_country(self) -> None:
        queries = _queries()
        path = queries.shortest_path(
            COMPANY_IDS["beta"], COMPANY_IDS["gamma"]
        )
        assert path is not None
        assert len(path) == 3
        assert path[1] == "country:us"

    def test_path_between_banks_via_alias(self) -> None:
        queries = _queries()
        path = queries.shortest_path(
            COMPANY_IDS["us_bank"], COMPANY_IDS["uk_bank"]
        )
        assert path is not None
        assert set(path) == {COMPANY_IDS["us_bank"], COMPANY_IDS["uk_bank"]}

    def test_no_path_returns_none(self) -> None:
        queries = _queries()
        assert (
            queries.shortest_path("company:nope", COMPANY_IDS["alpha"])
            is None
        )
        assert queries.shortest_path(COMPANY_IDS["alpha"], "company:nope") is None

    def test_same_node_trivial_path(self) -> None:
        queries = _queries()
        assert (
            queries.shortest_path(
                COMPANY_IDS["alpha"], COMPANY_IDS["alpha"]
            )
            == [COMPANY_IDS["alpha"]]
        )


class TestConnectedComponents:
    def test_two_components_with_documented_sizes(self) -> None:
        queries = _queries()
        components = queries.connected_components()
        sizes = sorted(len(c) for c in components)
        assert sizes == [7, 25]

    def test_every_company_has_a_component(self) -> None:
        queries = _queries()
        components = queries.connected_components()
        flat = {node for component in components for node in component}
        for company_id in COMPANY_IDS.values():
            assert company_id in flat
