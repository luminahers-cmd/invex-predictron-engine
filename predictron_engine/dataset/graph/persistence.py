"""Graph persistence contract (no external services required).

Defines a stable, versioned JSON snapshot of a :class:`KnowledgeGraph`.
The snapshot is the persistence boundary: the in-memory graph is the
working store, and any future backend (a graph database such as Neo4j,
an object store, or a columnar warehouse) only needs to round-trip this
document.  Nothing here depends on a database driver.

The serialization is canonical: nodes are ordered by ``node_id`` and
edges by canonical key, so identical graphs always produce byte-identical
JSON.  This is what makes the graph reproducible and hashable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from predictron_engine.dataset.graph.model import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
)
from predictron_engine.dataset.graph.store import KnowledgeGraph

GRAPH_SCHEMA_VERSION = 1


def graph_to_dict(graph: KnowledgeGraph) -> dict[str, Any]:
    """Return a deterministic, JSON-serializable snapshot of a graph."""
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "nodes": [node.to_dict() for node in graph.nodes()],
        "edges": [edge.to_dict() for edge in graph.edges()],
    }


def canonical_graph_json(graph: KnowledgeGraph) -> str:
    """Return canonical JSON (sorted keys, no insignificant whitespace)."""
    return json.dumps(
        graph_to_dict(graph),
        sort_keys=True,
        separators=(",", ":"),
    )


def graph_from_dict(data: dict[str, Any]) -> KnowledgeGraph:
    """Rebuild a :class:`KnowledgeGraph` from a snapshot document."""
    graph = KnowledgeGraph()
    raw_nodes = data.get("nodes", [])
    raw_edges = data.get("edges", [])
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        msg = "invalid graph snapshot: nodes/edges must be lists"
        raise ValueError(msg)
    for raw in raw_nodes:
        graph.add_node(_node_from_dict(raw))
    for raw in raw_edges:
        graph.add_edge(_edge_from_dict(raw))
    return graph


def write_graph(graph: KnowledgeGraph, path: str | Path) -> Path:
    """Write a graph snapshot to ``path`` and return the resolved path."""
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(graph_to_dict(graph), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def read_graph(path: str | Path) -> KnowledgeGraph | None:
    """Read a graph snapshot from ``path``.

    Returns ``None`` when the file does not exist.
    """
    target = Path(path)
    if not target.exists():
        return None
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = "invalid graph snapshot: root must be an object"
        raise ValueError(msg)
    return graph_from_dict(data)


def _node_from_dict(raw: Any) -> GraphNode:
    if not isinstance(raw, dict):
        msg = "invalid node entry"
        raise ValueError(msg)
    return GraphNode(
        node_id=str(raw["node_id"]),
        node_type=NodeType(str(raw["node_type"])),
        label=str(raw.get("label", "")),
        properties=dict(raw.get("properties") or {}),
        sources=[str(source) for source in raw.get("sources", [])],
    )


def _edge_from_dict(raw: Any) -> GraphEdge:
    if not isinstance(raw, dict):
        msg = "invalid edge entry"
        raise ValueError(msg)
    return GraphEdge(
        edge_type=EdgeType(str(raw["edge_type"])),
        source_id=str(raw["source_id"]),
        target_id=str(raw["target_id"]),
        properties=dict(raw.get("properties") or {}),
        sources=[str(source) for source in raw.get("sources", [])],
    )
