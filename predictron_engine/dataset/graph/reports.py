"""Knowledge graph reports.

Produces the three deterministic report documents:

* ``graph_report``          — high-level overview + metrics + summary
* ``graph_statistics``      — detailed structural statistics
* ``relationship_summary``  — per-relationship breakdown with provenance

Every value is derived from the graph itself; only a hash of the
canonical serialization is added so reports are reproducible.
"""

from __future__ import annotations

import hashlib
from typing import Any

from predictron_engine.dataset.graph.metrics import compute_graph_metrics
from predictron_engine.dataset.graph.persistence import canonical_graph_json
from predictron_engine.dataset.graph.store import KnowledgeGraph


def graph_key(graph: KnowledgeGraph) -> str:
    """Return the reproducible SHA-256 key of a graph's canonical form."""
    return hashlib.sha256(
        canonical_graph_json(graph).encode("utf-8")
    ).hexdigest()


def build_graph_report(graph: KnowledgeGraph) -> dict[str, Any]:
    """Build the high-level ``graph_report`` document."""
    metrics = compute_graph_metrics(graph)
    return {
        "report_type": "graph_report",
        "overview": {
            "node_count": metrics.node_count,
            "edge_count": metrics.edge_count,
            "company_count": metrics.node_counts.get("company", 0),
            "connected_components": metrics.component_count,
            "is_connected": metrics.is_connected,
            "graph_key": graph_key(graph),
        },
        "metrics": metrics.to_dict(),
        "relationship_summary": build_relationship_summary(graph),
        "top_connected_nodes": _top_connected(graph, limit=10),
        "validation": {
            "valid": not graph.validate(),
            "issues": graph.validate(),
        },
    }


def build_graph_statistics(graph: KnowledgeGraph) -> dict[str, Any]:
    """Build the detailed ``graph_statistics`` document."""
    metrics = compute_graph_metrics(graph)
    return {
        "report_type": "graph_statistics",
        "node_counts": metrics.node_counts,
        "edge_counts": metrics.edge_counts,
        "degree_distribution": metrics.degree.to_dict(),
        "connected_components": {
            "count": metrics.component_count,
            "sizes": metrics.component_sizes,
            "largest_component_size": metrics.largest_component_size,
            "average_component_size": metrics.average_component_size,
        },
        "density": {
            "directed": metrics.density_directed,
            "undirected": metrics.density_undirected,
        },
        "relationships": _relationship_rows(graph),
        "validation": {"valid": not graph.validate()},
    }


def build_relationship_summary(graph: KnowledgeGraph) -> dict[str, Any]:
    """Build the ``relationship_summary`` document."""
    rows = _relationship_rows(graph)
    involved_company_ids: set[str] = set()
    for edge in graph.edges():
        for endpoint in (edge.source_id, edge.target_id):
            node = graph.node(endpoint)
            if node is not None and node.node_type.value == "company":
                involved_company_ids.add(endpoint)
    return {
        "report_type": "relationship_summary",
        "total_edges": graph.edge_count,
        "relationship_count": len(rows),
        "relationships": rows,
        "company_nodes_with_relations": len(involved_company_ids),
        "company_nodes_total": _company_count(graph),
        "provenance": {
            "all_edges_have_provenance": not any(
                not edge.sources for edge in graph.edges()
            ),
            "all_nodes_have_provenance": not any(
                not node.sources for node in graph.nodes()
            ),
        },
    }


def _company_count(graph: KnowledgeGraph) -> int:
    from predictron_engine.dataset.graph.model import NodeType

    return len(graph.node_ids_of_type(NodeType.COMPANY))


def _relationship_rows(graph: KnowledgeGraph) -> list[dict[str, Any]]:
    from predictron_engine.dataset.graph.model import EdgeType

    by_type: dict[str, list[Any]] = {}
    for edge in graph.edges():
        by_type.setdefault(edge.edge_type.value, []).append(edge)

    rows: list[dict[str, Any]] = []
    for edge_type in sorted({edge.value for edge in EdgeType}):
        edges = by_type.get(edge_type, [])
        sources = sorted({edge.source_id for edge in edges})
        targets = sorted({edge.target_id for edge in edges})
        sample = []
        for edge in sorted(
            edges, key=lambda edge: (edge.source_id, edge.target_id)
        )[:3]:
            sample.append(
                {
                    "source": _label(graph, edge.source_id),
                    "target": _label(graph, edge.target_id),
                }
            )
        rows.append(
            {
                "edge_type": edge_type,
                "count": len(edges),
                "distinct_sources": len(sources),
                "distinct_targets": len(targets),
                "sample": sample,
            }
        )
    return rows


def _top_connected(
    graph: KnowledgeGraph, *, limit: int
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = [
        {
            "node_id": node.node_id,
            "label": node.label,
            "node_type": node.node_type.value,
            "degree": graph.degree(node.node_id),
        }
        for node in graph.nodes()
    ]
    entries.sort(
        key=lambda entry: (
            -entry["degree"],
            entry["node_type"],
            entry["label"],
            entry["node_id"],
        )
    )
    return entries[:limit]


def _label(graph: KnowledgeGraph, node_id: str) -> str:
    node = graph.node(node_id)
    return node.label if node else node_id
