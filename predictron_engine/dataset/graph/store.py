"""In-memory knowledge graph storage optimized for traversal.

The store is a plain adjacency-list graph backed by dictionaries and
sets:

* ``_nodes``   — ``node_id -> GraphNode``
* ``_edges``   — canonical edge key -> :class:`GraphEdge`
* ``_out_adj`` — ``node_id -> {(neighbor_id, edge_type)}`` (outgoing)
* ``_in_adj``  — ``node_id -> {(neighbor_id, edge_type)}`` (incoming)

Lookups are O(1), neighbor iteration is O(degree), and edge attributes
are reachable directly by canonical key.  The representation scales to
millions of edges (an edge is stored once; adjacency holds references).

Persistence is intentionally decoupled: :mod:`persistence` defines a
stable JSON snapshot that future backends (e.g. Neo4j) can implement
without changing this class.  No external service is required.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from predictron_engine.dataset.graph.model import (
    SYMMETRIC_EDGE_TYPES,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    canonical_edge_key,
)


class KnowledgeGraph:
    """Mutable in-memory graph with deterministic ordering guarantees."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[tuple[str, str, str], GraphEdge] = {}
        self._out_adj: dict[str, set[tuple[str, EdgeType]]] = {}
        self._in_adj: dict[str, set[tuple[str, EdgeType]]] = {}

    # ---- Construction ----

    def add_node(self, node: GraphNode) -> GraphNode:
        """Add a node, merging provenance with any existing node.

        Returns the stored node.  Re-adding the same node is idempotent;
        sources union and list-valued properties union deterministically.
        """
        existing = self._nodes.get(node.node_id)
        if existing is None:
            stored = GraphNode(
                node_id=node.node_id,
                node_type=node.node_type,
                label=node.label,
                properties=dict(node.properties),
                sources=sorted(set(node.sources)),
            )
            self._nodes[stored.node_id] = stored
            self._out_adj.setdefault(stored.node_id, set())
            self._in_adj.setdefault(stored.node_id, set())
            return stored

        merged_properties = _merge_properties(existing.properties, node.properties)
        label = existing.label or node.label
        stored = GraphNode(
            node_id=existing.node_id,
            node_type=existing.node_type,
            label=label,
            properties=merged_properties,
            sources=sorted(set([*existing.sources, *node.sources])),
        )
        self._nodes[stored.node_id] = stored
        return stored

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        """Add an edge, canonicalizing symmetric types and merging provenance.

        Re-adding the same edge is idempotent.  Existing scalar
        properties win; list properties union.
        """
        source_id = edge.source_id
        target_id = edge.target_id
        if edge.edge_type in SYMMETRIC_EDGE_TYPES and target_id < source_id:
            source_id, target_id = target_id, source_id

        key = canonical_edge_key(source_id, edge.edge_type, target_id)
        existing = self._edges.get(key)
        if existing is None:
            stored = GraphEdge(
                edge_type=edge.edge_type,
                source_id=source_id,
                target_id=target_id,
                properties=dict(edge.properties),
                sources=sorted(set(edge.sources)),
            )
            self._edges[key] = stored
            self._out_adj.setdefault(source_id, set()).add(
                (target_id, edge.edge_type)
            )
            self._in_adj.setdefault(target_id, set()).add(
                (source_id, edge.edge_type)
            )
            self._out_adj.setdefault(target_id, set())
            self._in_adj.setdefault(source_id, set())
            return stored

        merged_properties = _merge_properties(existing.properties, edge.properties)
        stored = GraphEdge(
            edge_type=existing.edge_type,
            source_id=existing.source_id,
            target_id=existing.target_id,
            properties=merged_properties,
            sources=sorted(set([*existing.sources, *edge.sources])),
        )
        self._edges[key] = stored
        return stored

    # ---- Queries ----

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def has_edge(self, source_id: str, edge_type: EdgeType, target_id: str) -> bool:
        return canonical_edge_key(source_id, edge_type, target_id) in self._edges

    def node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def edge(
        self, source_id: str, edge_type: EdgeType, target_id: str
    ) -> GraphEdge | None:
        return self._edges.get(canonical_edge_key(source_id, edge_type, target_id))

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def nodes(self) -> Iterator[GraphNode]:
        """Iterate nodes in deterministic ``node_id`` order."""
        for node_id in sorted(self._nodes):
            yield self._nodes[node_id]

    def edges(self) -> Iterator[GraphEdge]:
        """Iterate edges in deterministic canonical-key order."""
        for key in sorted(self._edges):
            yield self._edges[key]

    def nodes_of_type(self, node_type: NodeType) -> list[GraphNode]:
        """Return all nodes of a given type, sorted by ``node_id``."""
        return [
            node
            for node in self.nodes()
            if node.node_type == node_type
        ]

    def node_ids_of_type(self, node_type: NodeType) -> list[str]:
        return [
            node.node_id
            for node in self.nodes_of_type(node_type)
        ]

    def out_neighbors(
        self, node_id: str, edge_type: EdgeType | None = None
    ) -> list[GraphNode]:
        """Return distinct outgoing neighbors, deterministically ordered."""
        return self._neighbors(self._out_adj, node_id, edge_type)

    def in_neighbors(
        self, node_id: str, edge_type: EdgeType | None = None
    ) -> list[GraphNode]:
        """Return distinct incoming neighbors, deterministically ordered."""
        return self._neighbors(self._in_adj, node_id, edge_type)

    def out_edges(
        self, node_id: str, edge_type: EdgeType | None = None
    ) -> list[GraphEdge]:
        """Return outgoing edges, sorted by ``(edge_type, target_id)``."""
        return self._incident(self._out_adj, node_id, edge_type, outgoing=True)

    def in_edges(
        self, node_id: str, edge_type: EdgeType | None = None
    ) -> list[GraphEdge]:
        """Return incoming edges, sorted by ``(edge_type, source_id)``."""
        return self._incident(self._in_adj, node_id, edge_type, outgoing=False)

    def incident_edges(self, node_id: str) -> list[GraphEdge]:
        """Return every edge touching ``node_id`` (both directions)."""
        by_key: dict[tuple[str, str, str], GraphEdge] = {}
        for edge in self.out_edges(node_id):
            by_key[edge.key] = edge
        for edge in self.in_edges(node_id):
            by_key[edge.key] = edge
        return [by_key[key] for key in sorted(by_key)]

    def edges_between(self, node_a: str, node_b: str) -> list[GraphEdge]:
        """Return all edge types connecting two nodes (either direction)."""
        keys: set[tuple[str, str, str]] = set()
        for neighbor_id, edge_type in self._out_adj.get(node_a, set()):
            if neighbor_id == node_b:
                keys.add(canonical_edge_key(node_a, edge_type, node_b))
        for neighbor_id, edge_type in self._in_adj.get(node_a, set()):
            if neighbor_id == node_b:
                keys.add(canonical_edge_key(node_b, edge_type, node_a))
        return [self._edges[key] for key in sorted(keys)]

    def degree(self, node_id: str) -> int:
        """Number of distinct incident edges (out + in, de-duplicated)."""
        return len(self.incident_edges(node_id))

    def adjacent(
        self, node_id: str, *, bidirectional: bool = True
    ) -> list[tuple[str, EdgeType]]:
        """Return distinct ``(neighbor_id, edge_type)`` pairs.

        Deterministically sorted by ``(neighbor_id, edge_type)``.  Used by
        traversal queries so they never reach into adjacency internals.
        """
        items: set[tuple[str, EdgeType]] = set(
            self._out_adj.get(node_id, set())
        )
        if bidirectional:
            items.update(self._in_adj.get(node_id, set()))
        return sorted(items, key=lambda item: (item[0], item[1].value))

    def out_degree(self, node_id: str) -> int:
        return len(self._out_adj.get(node_id, set()))

    def in_degree(self, node_id: str) -> int:
        return len(self._in_adj.get(node_id, set()))

    def validate(self) -> list[str]:
        """Return provenance/structure problems, if any.

        A valid knowledge graph has no dangling edges, every node and
        edge carries at least one source record ID, and every edge
        property list is deterministic.
        """
        issues: list[str] = []
        for node in self._nodes.values():
            if not node.sources:
                issues.append(f"node without provenance: {node.node_id}")
        for edge in self._edges.values():
            if not edge.sources:
                issues.append(f"edge without provenance: {edge.source_id}"
                              f" -{edge.edge_type.value}-> {edge.target_id}")
            if edge.source_id not in self._nodes:
                issues.append(f"dangling edge source: {edge.source_id}")
            if edge.target_id not in self._nodes:
                issues.append(f"dangling edge target: {edge.target_id}")
        return issues

    # ---- Internal helpers ----

    def _neighbors(
        self,
        adjacency: dict[str, set[tuple[str, EdgeType]]],
        node_id: str,
        edge_type: EdgeType | None,
    ) -> list[GraphNode]:
        neighbor_ids: set[str] = set()
        for neighbor_id, kind in adjacency.get(node_id, set()):
            if edge_type is not None and kind != edge_type:
                continue
            neighbor_ids.add(neighbor_id)
        found: list[GraphNode] = []
        for neighbor_id in neighbor_ids:
            node = self._nodes.get(neighbor_id)
            if node is not None:
                found.append(node)
        found.sort(key=lambda node: (node.node_type.value, node.node_id))
        return found

    def _incident(
        self,
        adjacency: dict[str, set[tuple[str, EdgeType]]],
        node_id: str,
        edge_type: EdgeType | None,
        *,
        outgoing: bool,
    ) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        for other_id, kind in adjacency.get(node_id, set()):
            if edge_type is not None and kind != edge_type:
                continue
            source_id, target_id = (
                (node_id, other_id) if outgoing else (other_id, node_id)
            )
            edge = self._edges.get(canonical_edge_key(source_id, kind, target_id))
            if edge is not None:
                edges.append(edge)
        edges.sort(
            key=lambda edge: (
                edge.edge_type.value,
                edge.target_id if outgoing else edge.source_id,
            )
        )
        return edges


def _merge_properties(
    existing: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, Any]:
    """Merge property mappings deterministically.

    List/set values union; existing scalars win over incoming scalars so
    the first-seen grounded value is preserved.
    """
    merged: dict[str, Any] = dict(existing)
    for key, value in incoming.items():
        if key not in merged or merged[key] is None:
            merged[key] = value
            continue
        current = merged[key]
        if isinstance(current, list | tuple | set) or isinstance(
            value, list | tuple | set
        ):
            merged[key] = sorted(
                {str(item) for item in _as_iterable(current)}
                | {str(item) for item in _as_iterable(value)}
            )
        else:
            merged[key] = current
    return merged


def _as_iterable(value: object) -> Iterable[object]:
    if isinstance(value, list | tuple | set):
        return value
    return (value,)
