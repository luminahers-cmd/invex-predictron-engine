"""Knowledge graph model (Company Knowledge Graph).

Defines the node and edge vocabulary used by the graph layer plus the
immutable value objects that carry provenance on every element.

The graph is intentionally schema-light: nodes and edges are plain
dataclasses with deterministic serialization.  Every node and edge keeps
the sorted list of ``DatasetRecord.record_id`` values that produced it
(``sources``), so the graph can be audited back to the dataset.

Nothing in this module invents data.  It only names the vocabulary that
:mod:`predictron_engine.dataset.graph.builder` populates from records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    """Node kinds in the company knowledge graph."""

    COMPANY = "company"
    FOUNDER = "founder"
    INVESTOR = "investor"
    ORGANIZATION = "organization"
    INDUSTRY = "industry"
    TECHNOLOGY = "technology"
    PRODUCT = "product"
    COUNTRY = "country"
    STATE = "state"
    CITY = "city"
    DOMAIN = "domain"
    IDENTIFIER = "identifier"


class EdgeType(str, Enum):
    """Relationship kinds in the company knowledge graph."""

    FOUNDED_BY = "FOUNDED_BY"
    INVESTED_BY = "INVESTED_BY"
    LOCATED_IN = "LOCATED_IN"
    OPERATES_IN = "OPERATES_IN"
    USES_TECHNOLOGY = "USES_TECHNOLOGY"
    BUILDS_PRODUCT = "BUILDS_PRODUCT"
    HAS_DOMAIN = "HAS_DOMAIN"
    HAS_IDENTIFIER = "HAS_IDENTIFIER"
    ALIAS_OF = "ALIAS_OF"
    ACQUIRED_BY = "ACQUIRED_BY"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    RELATED_TO = "RELATED_TO"


# Relationships whose semantics do not depend on direction.  The store
# canonicalizes their endpoints so lookups never miss a reverse edge.
SYMMETRIC_EDGE_TYPES: frozenset[EdgeType] = frozenset(
    {EdgeType.ALIAS_OF, EdgeType.RELATED_TO}
)

# Edge kinds that connect a Company node to another Company node.  Used
# by queries and metrics to distinguish structural edges from attribute
# edges.
COMPANY_TO_COMPANY_EDGE_TYPES: frozenset[EdgeType] = frozenset(
    {
        EdgeType.ALIAS_OF,
        EdgeType.ACQUIRED_BY,
        EdgeType.SUBSIDIARY_OF,
        EdgeType.RELATED_TO,
    }
)


def canonical_edge_key(
    source_id: str, edge_type: EdgeType, target_id: str
) -> tuple[str, str, str]:
    """Return the canonical key for an edge.

    Symmetric edge types are normalized so ``(a, X, b)`` and ``(b, X, a)``
    collapse to the same key regardless of insertion order.
    """
    if edge_type in SYMMETRIC_EDGE_TYPES and target_id < source_id:
        source_id, target_id = target_id, source_id
    return (source_id, edge_type.value, target_id)


@dataclass(frozen=True)
class GraphNode:
    """A single node in the knowledge graph.

    Attributes
    ----------
    node_id :
        Deterministic identifier, e.g. ``company:<record ids>`` or
        ``industry:fintech``.
    node_type :
        Vocabulary kind of the node.
    label :
        Human-readable display value exactly as grounded in the data (or
        the deterministic normalized form for attribute nodes).
    properties :
        Extra deterministic scalar/list attributes (never fabricated).
    sources :
        Sorted distinct record IDs that produced the node.
    """

    node_id: str
    node_type: NodeType
    label: str
    properties: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable representation."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label,
            "properties": _stable_properties(self.properties),
            "sources": sorted(set(self.sources)),
        }


@dataclass(frozen=True)
class GraphEdge:
    """A directed relationship between two nodes.

    Attributes
    ----------
    edge_type :
        Relationship kind.
    source_id / target_id :
        Endpoint node identifiers.  For symmetric edge types the store
        canonicalizes the endpoint order.
    properties :
        Deterministic relationship attributes (e.g. identifier ``kind``).
    sources :
        Sorted distinct record IDs that produced the edge.  Every edge in
        the graph must carry at least one source.
    """

    edge_type: EdgeType
    source_id: str
    target_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str, str]:
        """Return the store key for this edge."""
        return canonical_edge_key(self.source_id, self.edge_type, self.target_id)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable representation."""
        return {
            "edge_type": self.edge_type.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "properties": _stable_properties(self.properties),
            "sources": sorted(set(self.sources)),
        }


def _stable_properties(properties: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministically ordered copy of a property mapping."""
    stable: dict[str, Any] = {}
    for key in sorted(properties):
        value = properties[key]
        if isinstance(value, list | tuple | set):
            stable[key] = sorted(str(item) for item in value)
        else:
            stable[key] = value
    return stable
