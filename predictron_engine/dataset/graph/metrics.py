"""Knowledge graph metrics.

Computes the structural metrics required for the milestone:

* node counts (total and by type)
* edge counts (total and by type)
* connected components (count, sizes, largest)
* degree distribution (histogram + summary statistics)
* graph density (directed and undirected)

All metrics are deterministic and read-only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from predictron_engine.dataset.graph.queries import GraphQueries
from predictron_engine.dataset.graph.store import KnowledgeGraph


@dataclass
class DegreeDistribution:
    """Histogram of total (undirected) degree across nodes."""

    histogram: dict[int, int] = field(default_factory=dict)
    min_degree: int = 0
    max_degree: int = 0
    average_degree: float = 0.0
    isolated_nodes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "histogram": {str(k): v for k, v in sorted(self.histogram.items())},
            "min_degree": self.min_degree,
            "max_degree": self.max_degree,
            "average_degree": self.average_degree,
            "isolated_nodes": self.isolated_nodes,
        }


@dataclass
class GraphMetrics:
    """Structural metrics for a knowledge graph."""

    node_count: int = 0
    edge_count: int = 0
    node_counts: dict[str, int] = field(default_factory=dict)
    edge_counts: dict[str, int] = field(default_factory=dict)
    component_count: int = 0
    component_sizes: list[int] = field(default_factory=list)
    largest_component_size: int = 0
    average_component_size: float = 0.0
    degree: DegreeDistribution = field(default_factory=DegreeDistribution)
    density_directed: float = 0.0
    density_undirected: float = 0.0
    is_connected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "node_counts": dict(sorted(self.node_counts.items())),
            "edge_counts": dict(sorted(self.edge_counts.items())),
            "component_count": self.component_count,
            "component_sizes": list(self.component_sizes),
            "largest_component_size": self.largest_component_size,
            "average_component_size": self.average_component_size,
            "degree_distribution": self.degree.to_dict(),
            "density_directed": self.density_directed,
            "density_undirected": self.density_undirected,
            "is_connected": self.is_connected,
        }


def compute_graph_metrics(graph: KnowledgeGraph) -> GraphMetrics:
    """Compute structural metrics for the supplied graph."""
    node_counter: Counter[str] = Counter()
    for node in graph.nodes():
        node_counter[node.node_type.value] += 1
    edge_counter: Counter[str] = Counter()
    for edge in graph.edges():
        edge_counter[edge.edge_type.value] += 1

    components = GraphQueries(graph).connected_components()
    component_sizes = [len(component) for component in components]

    degree = _degree_distribution(graph)
    node_count = graph.node_count
    edge_count = graph.edge_count
    possible = node_count * (node_count - 1)
    density_directed = round(edge_count / possible, 8) if possible else 0.0
    density_undirected = (
        round((2 * edge_count) / possible, 8) if possible else 0.0
    )

    return GraphMetrics(
        node_count=node_count,
        edge_count=edge_count,
        node_counts=dict(node_counter),
        edge_counts=dict(edge_counter),
        component_count=len(components),
        component_sizes=component_sizes,
        largest_component_size=component_sizes[0] if component_sizes else 0,
        average_component_size=(
            round(node_count / len(components), 6) if components else 0.0
        ),
        degree=degree,
        density_directed=density_directed,
        density_undirected=density_undirected,
        is_connected=len(components) <= 1,
    )


def _degree_distribution(graph: KnowledgeGraph) -> DegreeDistribution:
    histogram: Counter[int] = Counter()
    total = 0
    minimum: int | None = None
    maximum = 0
    isolated = 0
    for node in graph.nodes():
        degree = graph.degree(node.node_id)
        histogram[degree] += 1
        total += degree
        minimum = degree if minimum is None else min(minimum, degree)
        maximum = max(maximum, degree)
        if degree == 0:
            isolated += 1
    node_count = graph.node_count
    return DegreeDistribution(
        histogram=dict(histogram),
        min_degree=minimum if minimum is not None else 0,
        max_degree=maximum,
        average_degree=round(total / node_count, 6) if node_count else 0.0,
        isolated_nodes=isolated,
    )
