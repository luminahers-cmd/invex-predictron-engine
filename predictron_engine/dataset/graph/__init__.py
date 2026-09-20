"""Company Knowledge Graph (Project E3).

A deterministic graph layer that transforms resolved companies into
connected entities.  The graph is generated entirely from
:class:`DatasetRecord` objects, preserves provenance on every node and
edge, and never fabricates relationships.

Public API
----------
model : NodeType / EdgeType vocabulary, GraphNode / GraphEdge values.
store : :class:`KnowledgeGraph` — in-memory store optimized for traversal.
builder : :class:`CompanyKnowledgeGraphBuilder` — deterministic builder.
queries : :class:`GraphQueries` — neighbors, paths, components, similarity.
metrics : :class:`GraphMetrics` — node/edge counts, degree, density.
reports : ``build_graph_report`` / ``build_graph_statistics`` /
    ``build_relationship_summary``.
persistence : JSON snapshot contract (``write_graph`` / ``read_graph``).
"""

from predictron_engine.dataset.graph.builder import (
    CompanyKnowledgeGraphBuilder,
    CompanyNameIndex,
    GraphBuildReport,
    GraphBuildResult,
    attribute_node_id,
    company_node_id,
)
from predictron_engine.dataset.graph.metrics import (
    DegreeDistribution,
    GraphMetrics,
    compute_graph_metrics,
)
from predictron_engine.dataset.graph.model import (
    COMPANY_TO_COMPANY_EDGE_TYPES,
    SYMMETRIC_EDGE_TYPES,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    canonical_edge_key,
)
from predictron_engine.dataset.graph.persistence import (
    GRAPH_SCHEMA_VERSION,
    canonical_graph_json,
    graph_from_dict,
    graph_to_dict,
    read_graph,
    write_graph,
)
from predictron_engine.dataset.graph.queries import GraphQueries
from predictron_engine.dataset.graph.reports import (
    build_graph_report,
    build_graph_statistics,
    build_relationship_summary,
    graph_key,
)
from predictron_engine.dataset.graph.store import KnowledgeGraph

__all__ = [
    "COMPANY_TO_COMPANY_EDGE_TYPES",
    "SYMMETRIC_EDGE_TYPES",
    "CompanyKnowledgeGraphBuilder",
    "CompanyNameIndex",
    "DegreeDistribution",
    "EdgeType",
    "GRAPH_SCHEMA_VERSION",
    "GraphBuildReport",
    "GraphBuildResult",
    "GraphEdge",
    "GraphMetrics",
    "GraphNode",
    "GraphQueries",
    "KnowledgeGraph",
    "NodeType",
    "attribute_node_id",
    "build_graph_report",
    "build_graph_statistics",
    "build_relationship_summary",
    "canonical_edge_key",
    "canonical_graph_json",
    "company_node_id",
    "compute_graph_metrics",
    "graph_from_dict",
    "graph_key",
    "graph_to_dict",
    "read_graph",
    "write_graph",
]
