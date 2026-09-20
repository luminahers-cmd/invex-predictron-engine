"""Knowledge graph query API.

Read-only traversal and lookup helpers over a :class:`KnowledgeGraph`.
All methods are deterministic: results are sorted by stable keys and
every tie is broken by ``(node_type, node_id)``.

The API is intentionally small and composable:

* :meth:`GraphQueries.neighbors` — adjacent nodes and edge kinds
* :meth:`GraphQueries.shortest_path` — shortest connecting path (BFS)
* :meth:`GraphQueries.connected_components` — weakly connected components
* :meth:`GraphQueries.similar_companies` — neighbor-overlap similarity
* :meth:`GraphQueries.companies_by_industry` / ``_by_country`` /
  ``companies_using_technology`` — reverse attribute lookups
"""

from __future__ import annotations

from collections import deque
from typing import Any

from predictron_engine.dataset.company_name import core_name
from predictron_engine.dataset.graph.builder import attribute_node_id
from predictron_engine.dataset.graph.extract import entity_key
from predictron_engine.dataset.graph.model import (
    EdgeType,
    GraphNode,
    NodeType,
)
from predictron_engine.dataset.graph.store import KnowledgeGraph


class GraphQueries:
    """Traversal and lookup facade over a knowledge graph."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graph = graph

    @property
    def graph(self) -> KnowledgeGraph:
        return self._graph

    # ---- Lookup ----

    def lookup(self, token: str) -> str | None:
        """Resolve a node ID or human label to a node ID."""
        if self._graph.has_node(token):
            return token
        lowered = token.strip().casefold()
        if not lowered:
            return None
        exact: list[GraphNode] = []
        fuzzy: list[GraphNode] = []
        target_core = core_name(token)
        for node in self._graph.nodes():
            if node.label.casefold() == lowered:
                exact.append(node)
            elif (
                node.node_type == NodeType.COMPANY
                and target_core
                and core_name(node.label) == target_core
            ):
                fuzzy.append(node)
        matches = exact or fuzzy
        if not matches:
            return None
        matches.sort(key=lambda node: (node.node_type.value, node.node_id))
        return matches[0].node_id

    # ---- Neighbors ----

    def neighbors(
        self,
        node_id: str,
        *,
        edge_type: EdgeType | None = None,
        bidirectional: bool = True,
    ) -> list[dict[str, Any]]:
        """Return adjacent nodes with the edge kinds connecting them."""
        resolved = self.lookup(node_id)
        if resolved is None:
            return []
        edge_types: dict[str, set[EdgeType]] = {}
        for adjacent_id, kind in self._graph.adjacent(
            resolved, bidirectional=bidirectional
        ):
            if edge_type is not None and kind != edge_type:
                continue
            edge_types.setdefault(adjacent_id, set()).add(kind)
        return [
            {
                "node": self._node_dict(adjacent_id),
                "node_id": adjacent_id,
                "edge_types": sorted(
                    kind.value for kind in edge_types[adjacent_id]
                ),
            }
            for adjacent_id in sorted(edge_types)
        ]

    # ---- Paths and components ----

    def shortest_path(
        self,
        start: str,
        end: str,
        *,
        bidirectional: bool = True,
    ) -> list[str] | None:
        """Return the shortest node-ID path from ``start`` to ``end``.

        Uses breadth-first search.  Returns ``None`` when no path exists.
        """
        start_id = self.lookup(start)
        end_id = self.lookup(end)
        if start_id is None or end_id is None:
            return None
        if start_id == end_id:
            return [start_id]

        visited = {start_id}
        parent: dict[str, str] = {}
        queue: deque[str] = deque([start_id])
        while queue:
            current = queue.popleft()
            for neighbor, _kind in self._adjacent(current, bidirectional):
                if neighbor.node_id in visited:
                    continue
                visited.add(neighbor.node_id)
                parent[neighbor.node_id] = current
                if neighbor.node_id == end_id:
                    return _reconstruct(parent, start_id, end_id)
                queue.append(neighbor.node_id)
        return None

    def connected_components(self) -> list[list[str]]:
        """Return weakly connected components (direction ignored)."""
        parent: dict[str, str] = {
            node.node_id: node.node_id for node in self._graph.nodes()
        }

        def find(node: str) -> str:
            root = node
            while parent[root] != root:
                root = parent[root]
            while parent[node] != root:
                parent[node], node = root, parent[node]
            return root

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                if left_root < right_root:
                    parent[right_root] = left_root
                else:
                    parent[left_root] = right_root

        for edge in self._graph.edges():
            union(edge.source_id, edge.target_id)

        groups: dict[str, list[str]] = {}
        for node_id in parent:
            groups.setdefault(find(node_id), []).append(node_id)
        components = [sorted(members) for members in groups.values()]
        components.sort(key=lambda members: (-len(members), members[0]))
        return components

    # ---- Similarity ----

    def similar_companies(
        self, company: str, *, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Rank companies by neighbor-set overlap with a company."""
        target_id = self.lookup(company)
        if target_id is None:
            return []
        target_node = self._graph.node(target_id)
        if target_node is None or target_node.node_type != NodeType.COMPANY:
            return []
        target_neighbors = self._neighbor_set(target_id)

        results: list[dict[str, Any]] = []
        for candidate in self._graph.nodes_of_type(NodeType.COMPANY):
            if candidate.node_id == target_id:
                continue
            candidate_neighbors = self._neighbor_set(candidate.node_id)
            shared = target_neighbors & candidate_neighbors
            union = target_neighbors | candidate_neighbors
            score = round(len(shared) / len(union), 6) if union else 0.0
            if score <= 0.0:
                continue
            results.append(
                {
                    "node_id": candidate.node_id,
                    "label": candidate.label,
                    "similarity": score,
                    "shared_neighbor_count": len(shared),
                    "shared_neighbors": sorted(shared),
                    "shared_by_type": _shared_by_type(self._graph, shared),
                }
            )
        results.sort(
            key=lambda entry: (
                -entry["similarity"],
                str(entry["label"]),
                entry["node_id"],
            )
        )
        return results[: max(limit, 0)]

    # ---- Attribute lookups ----

    def companies_by_industry(self, industry: str) -> list[dict[str, Any]]:
        return self._companies_for_attribute(
            NodeType.INDUSTRY, industry, {EdgeType.OPERATES_IN}
        )

    def companies_by_country(self, country: str) -> list[dict[str, Any]]:
        return self._companies_for_attribute(
            NodeType.COUNTRY, country, {EdgeType.LOCATED_IN}
        )

    def companies_using_technology(self, technology: str) -> list[dict[str, Any]]:
        return self._companies_for_attribute(
            NodeType.TECHNOLOGY, technology, {EdgeType.USES_TECHNOLOGY}
        )

    # ---- Internal helpers ----

    def _node_dict(self, node_id: str) -> dict[str, Any]:
        node = self._graph.node(node_id)
        return node.to_dict() if node is not None else {}

    def _neighbor_set(self, node_id: str) -> set[str]:
        neighbors: set[str] = set()
        for neighbor, _kind in self._adjacent(node_id, True):
            neighbors.add(neighbor.node_id)
        return neighbors

    def _adjacent(
        self, node_id: str, bidirectional: bool
    ) -> list[tuple[GraphNode, EdgeType]]:
        """Return (neighbor, edge_type) pairs in deterministic order."""
        pairs: list[tuple[GraphNode, EdgeType]] = []
        for neighbor_id, kind in self._graph.adjacent(
            node_id, bidirectional=bidirectional
        ):
            node = self._graph.node(neighbor_id)
            if node is not None:
                pairs.append((node, kind))
        return pairs

    def _companies_for_attribute(
        self,
        node_type: NodeType,
        value: str,
        edge_types: set[EdgeType],
    ) -> list[dict[str, Any]]:
        node_id = self._lookup_attribute(node_type, value)
        if node_id is None:
            return []
        companies: dict[str, GraphNode] = {}
        for edge in self._graph.in_edges(node_id):
            if edge.edge_type not in edge_types:
                continue
            source = self._graph.node(edge.source_id)
            if source is not None and source.node_type == NodeType.COMPANY:
                companies[source.node_id] = source
        return [companies[key].to_dict() for key in sorted(companies)]

    def _lookup_attribute(self, node_type: NodeType, value: str) -> str | None:
        direct = attribute_node_id(node_type, value)
        if self._graph.has_node(direct):
            return direct
        key = entity_key(value)
        for node in self._graph.nodes_of_type(node_type):
            if entity_key(node.label) == key:
                return node.node_id
        return None


def _reconstruct(
    parent: dict[str, str], start_id: str, end_id: str
) -> list[str]:
    path = [end_id]
    current = end_id
    while current != start_id:
        current = parent[current]
        path.append(current)
    path.reverse()
    return path


def _shared_by_type(
    graph: KnowledgeGraph, shared: set[str]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node_id in shared:
        node = graph.node(node_id)
        if node is None:
            continue
        counts[node.node_type.value] = counts.get(node.node_type.value, 0) + 1
    return dict(sorted(counts.items()))
