"""Feature definitions for the KNOWLEDGE_GRAPH category.

Deterministic graph metrics derived from the company knowledge graph.
"""

from __future__ import annotations

from typing import Any

from predictron_engine.feature_store.models import (
    FeatureCategory,
    FeatureDefinition,
    ValueType,
)
from predictron_engine.feature_store.registry import FeatureComputer


def _make_feature(
    feature_id: str,
    feature_name: str,
    description: str,
    value_type: ValueType = ValueType.FLOAT,
    source_fields: list[str] | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    tags: list[str] | None = None,
) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=feature_id,
        feature_name=feature_name,
        category=FeatureCategory.KNOWLEDGE_GRAPH,
        description=description,
        value_type=value_type,
        source_fields=source_fields or [],
        min_value=min_value,
        max_value=max_value,
        tags=tags or ["knowledge_graph"],
    )


def _get_graph(ctx: dict[str, Any]) -> Any:
    return ctx.get("knowledge_graph")


def _get_queries(ctx: dict[str, Any]) -> Any:
    return ctx.get("graph_queries")


def _get_node_id(ctx: dict[str, Any], record: Any) -> str | None:
    node_id = ctx.get("company_node_id")
    if isinstance(node_id, str) and node_id:
        return node_id
    company_id = ctx.get("company_id")
    if isinstance(company_id, str) and company_id:
        return company_id
    return None


def _kg_ev(node_id: str, field: str) -> list[dict[str, str]]:
    return [{"source_type": "knowledge_graph", "source_id": node_id,
             "source_field": field}]


GRAPH_DEGREE_DEFINITION = _make_feature(
    feature_id="graph_degree",
    feature_name="Graph Degree",
    description="Total degree of the company node in the knowledge graph",
    value_type=ValueType.INT,
    source_fields=["knowledge_graph"],
    min_value=0.0,
    tags=["knowledge_graph", "degree"],
)


def compute_graph_degree(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    graph = _get_graph(ctx)
    if graph is None:
        return None, []
    node_id = _get_node_id(ctx, record)
    if node_id is None:
        return None, []
    try:
        degree = graph.degree(node_id)
    except (KeyError, ValueError):
        return 0, []
    return degree, _kg_ev(node_id, "degree")


GRAPH_DENSITY_DEFINITION = _make_feature(
    feature_id="graph_density",
    feature_name="Graph Density",
    description="Directed graph density (edges / possible edges)",
    value_type=ValueType.FLOAT,
    source_fields=["knowledge_graph"],
    min_value=0.0,
    max_value=1.0,
    tags=["knowledge_graph", "density"],
)


def compute_graph_density(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    graph = _get_graph(ctx)
    if graph is None:
        return None, []
    node_count = graph.node_count
    edge_count = graph.edge_count
    possible = node_count * (node_count - 1)
    density = round(edge_count / possible, 8) if possible else 0.0
    return density, _kg_ev("global", "density")


CONNECTED_COMPONENT_SIZE_DEFINITION = _make_feature(
    feature_id="connected_component_size",
    feature_name="Connected Component Size",
    description="Size of the connected component containing the company",
    value_type=ValueType.INT,
    source_fields=["knowledge_graph"],
    min_value=1,
    tags=["knowledge_graph", "components"],
)


def compute_connected_component_size(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    queries = _get_queries(ctx)
    if queries is None:
        return None, []
    node_id = _get_node_id(ctx, record)
    if node_id is None:
        return None, []
    try:
        components = queries.connected_components()
    except (AttributeError, ValueError):
        return 1, []
    for component in components:
        if node_id in component:
            return len(component), _kg_ev(node_id, "component")
    return 1, _kg_ev(node_id, "component")


ECOSYSTEM_CONNECTIONS_DEFINITION = _make_feature(
    feature_id="ecosystem_connections",
    feature_name="Ecosystem Connections",
    description="Number of direct neighbors in the knowledge graph",
    value_type=ValueType.INT,
    source_fields=["knowledge_graph"],
    min_value=0.0,
    tags=["knowledge_graph", "ecosystem", "connections"],
)


def compute_ecosystem_connections(
    record: Any,
    deps: dict[str, Any],
    ctx: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    queries = _get_queries(ctx)
    if queries is None:
        return None, []
    node_id = _get_node_id(ctx, record)
    if node_id is None:
        return None, []
    try:
        neighbors = queries.neighbors(node_id)
    except (KeyError, ValueError):
        return 0, []
    return len(neighbors), _kg_ev(node_id, "neighbors")


KNOWLEDGE_GRAPH_FEATURES: list[tuple[FeatureDefinition, FeatureComputer]] = [
    (GRAPH_DEGREE_DEFINITION, compute_graph_degree),
    (GRAPH_DENSITY_DEFINITION, compute_graph_density),
    (CONNECTED_COMPONENT_SIZE_DEFINITION, compute_connected_component_size),
    (ECOSYSTEM_CONNECTIONS_DEFINITION, compute_ecosystem_connections),
]
