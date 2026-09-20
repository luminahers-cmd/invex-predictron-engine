"""Tests for graph metrics."""

from __future__ import annotations

from predictron_engine.dataset.graph import (
    CompanyKnowledgeGraphBuilder,
    GraphMetrics,
    compute_graph_metrics,
)
from tests.dataset.graph_helpers import small_dataset


def _metrics() -> GraphMetrics:
    graph = CompanyKnowledgeGraphBuilder().build(small_dataset()).graph
    return compute_graph_metrics(graph)


class TestMetrics:
    def test_totals_match_graph(self) -> None:
        metrics = _metrics()
        assert metrics.node_count == 32
        assert metrics.edge_count == 36

    def test_connected_components(self) -> None:
        assert _metrics().component_count == 2

    def test_component_sizes(self) -> None:
        metrics = _metrics()
        assert sorted(metrics.component_sizes) == [7, 25]
        assert metrics.largest_component_size == 25

    def test_density_is_bounded(self) -> None:
        m = _metrics()
        assert 0.0 < m.density_directed < 1.0
        assert m.density_directed < m.density_undirected

    def test_degree_distribution_summary(self) -> None:
        dd = _metrics().degree
        assert dd.max_degree >= dd.average_degree >= 1
        assert dd.min_degree == 1
        assert dd.isolated_nodes == 0
        assert dd.histogram
        assert dd.histogram[dd.max_degree] >= 1

    def test_to_dict_is_deterministic(self) -> None:
        d1 = _metrics().to_dict()
        d2 = _metrics().to_dict()
        assert d1 == d2
        assert d1["node_count"] == 32
        assert d1["component_count"] == 2
        assert isinstance(d1["degree_distribution"]["histogram"], dict)
