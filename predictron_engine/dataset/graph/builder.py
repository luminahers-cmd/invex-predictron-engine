"""Deterministic company knowledge graph builder.

Transforms a collection of :class:`DatasetRecord` objects into a connected
:class:`KnowledgeGraph`.  The builder is the only place that creates graph
elements, and it never fabricates an edge:

* The set of **Company** nodes is exactly the set of canonical
  :class:`CompanyIdentity` objects produced by the entity resolver.
* Every attribute node (founder, investor, industry, technology, ...)
  is created from a value that literally appears in a record's profile
  or ``analysis_metadata`` under a documented key (see
  :mod:`predictron_engine.dataset.graph.extract`).
* Every node and edge records the sorted ``record_id`` values that
  produced it, so provenance is preserved end to end.

Determinism
-----------
Records and identities are processed in sorted order, attribute values
are normalized and sorted, and entity-name resolution always chooses a
deterministic winner.  Given the same records, the same graph (and the
same ``graph_key``) is produced regardless of input order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from predictron_engine.dataset.company_name import (
    canonical_name_key,
    core_name,
)
from predictron_engine.dataset.entity_resolution import EntityResolver
from predictron_engine.dataset.graph.extract import (
    acquirer_values,
    entity_key,
    founder_values,
    investor_values,
    organization_values,
    product_values,
    record_city,
    record_country,
    record_region,
    related_values,
    technology_values,
)
from predictron_engine.dataset.graph.model import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
)
from predictron_engine.dataset.graph.persistence import canonical_graph_json
from predictron_engine.dataset.graph.store import KnowledgeGraph
from predictron_engine.dataset.identity import CompanyIdentity
from predictron_engine.dataset.models import DatasetRecord
from predictron_engine.dataset.outcomes import OutcomeRecord


def company_node_id(identity: CompanyIdentity) -> str:
    """Return the deterministic node ID for a resolved company."""
    return "company:" + "+".join(sorted(identity.record_ids))


def attribute_node_id(node_type: NodeType, label: str) -> str:
    """Return the deterministic node ID for an attribute node."""
    return f"{node_type.value}:{entity_key(label)}"


def identifier_node_id(kind: str, value: str) -> str:
    return f"identifier:{kind}:{value}"


@dataclass
class GraphBuildReport:
    """Summary of a single deterministic build pass."""

    record_count: int = 0
    identity_count: int = 0
    node_count: int = 0
    edge_count: int = 0
    node_counts: dict[str, int] = field(default_factory=dict)
    edge_counts: dict[str, int] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    provenance_issues: list[str] = field(default_factory=list)
    graph_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_type": "graph_build_report",
            "record_count": self.record_count,
            "identity_count": self.identity_count,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "node_counts": dict(sorted(self.node_counts.items())),
            "edge_counts": dict(sorted(self.edge_counts.items())),
            "sources": list(self.sources),
            "provenance_issues": list(self.provenance_issues),
            "graph_key": self.graph_key,
        }


@dataclass
class GraphBuildResult:
    """A built graph together with its report and source identities."""

    graph: KnowledgeGraph
    report: GraphBuildReport
    identities: list[CompanyIdentity] = field(default_factory=list)


class CompanyNameIndex:
    """Deterministic lookup from an entity name to a resolved company node.

    Built once per graph from the canonical identities.  Company names
    and aliases are indexed by their canonical name key; domains are
    indexed separately.  Resolution always returns a single winner.
    """

    def __init__(self, identities: list[CompanyIdentity]) -> None:
        self._by_name: dict[str, list[tuple[str, str]]] = {}
        self._by_domain: dict[str, str] = {}
        for identity in identities:
            node_id = company_node_id(identity)
            for name in [identity.canonical_name, *identity.aliases]:
                key = canonical_name_key(name)
                if key:
                    self._by_name.setdefault(key, []).append(
                        (node_id, identity.canonical_name)
                    )
            for domain in [
                identity.canonical_domain,
                *identity.alternate_domains,
            ]:
                if domain:
                    self._by_domain.setdefault(domain.casefold(), node_id)
        for key, entries in self._by_name.items():
            self._by_name[key] = sorted(
                set(entries), key=lambda entry: (entry[1], entry[0])
            )

    def resolve(self, name: str) -> str | None:
        """Resolve a referenced name to a company node ID, or ``None``."""
        cleaned = name.strip()
        if not cleaned:
            return None
        key = canonical_name_key(cleaned)
        candidates = self._by_name.get(key)
        if candidates:
            exact = [
                entry
                for entry in candidates
                if entry[1].casefold() == cleaned.casefold()
            ]
            return (exact or candidates)[0][0]
        if "." in cleaned:
            return self._by_domain.get(cleaned.casefold())
        return None


class CompanyKnowledgeGraphBuilder:
    """Builds a company knowledge graph from dataset records.

    Parameters
    ----------
    resolver :
        Optional :class:`EntityResolver`; a default one is constructed
        when not supplied.
    include_outcomes :
        When True (default), the optional ``outcomes`` argument passed to
        :meth:`build` contributes grounded INVESTED_BY / ACQUIRED_BY
        edges.  Set False to ignore outcomes entirely.
    """

    def __init__(
        self,
        resolver: EntityResolver | None = None,
        *,
        include_outcomes: bool = True,
    ) -> None:
        self._resolver = resolver or EntityResolver()
        self._include_outcomes = include_outcomes

    def build(
        self,
        records: list[DatasetRecord],
        *,
        outcomes: Mapping[str, OutcomeRecord] | None = None,
    ) -> GraphBuildResult:
        """Build the graph, returning the result and its report."""
        ordered = sorted(records, key=lambda record: record.record_id)
        records_by_id = {record.record_id: record for record in ordered}

        resolution = self._resolver.resolve(ordered)
        identities = sorted(
            resolution.identities,
            key=lambda identity: (
                identity.canonical_name,
                sorted(identity.record_ids),
            ),
        )
        identity_by_record = _identity_by_record(identities)

        graph = KnowledgeGraph()
        self._name_index = CompanyNameIndex(identities)
        self._identity_by_node = {
            company_node_id(identity): identity for identity in identities
        }

        for identity in identities:
            self._emit_company(graph, identity, records_by_id)

        if self._include_outcomes and outcomes:
            self._emit_outcome_edges(graph, outcomes, identity_by_record)

        self._emit_alias_edges(graph, identities)

        report = self._build_report(
            ordered, identities, graph
        )
        return GraphBuildResult(graph=graph, report=report, identities=identities)

    # ---- Company nodes and identity-level attributes ----

    def _emit_company(
        self,
        graph: KnowledgeGraph,
        identity: CompanyIdentity,
        records_by_id: dict[str, DatasetRecord],
    ) -> None:
        node_id = company_node_id(identity)
        properties: dict[str, Any] = {}
        if identity.canonical_domain:
            properties["canonical_domain"] = identity.canonical_domain
        if identity.aliases:
            properties["aliases"] = list(identity.aliases)
        if identity.status:
            properties["status"] = identity.status
        if identity.founded_year is not None:
            properties["founded_year"] = identity.founded_year
        if identity.employee_range:
            properties["employee_range"] = identity.employee_range
        properties["merge_decision"] = identity.merge_decision
        if identity.sources:
            properties["record_sources"] = list(identity.sources)

        graph.add_node(
            GraphNode(
                node_id=node_id,
                node_type=NodeType.COMPANY,
                label=identity.canonical_name,
                properties=properties,
                sources=list(identity.record_ids),
            )
        )

        for industry in identity.industries:
            self._link_attribute(
                graph, node_id, NodeType.INDUSTRY, industry,
                EdgeType.OPERATES_IN, list(identity.record_ids),
            )
        for country in identity.country_codes:
            self._link_attribute(
                graph, node_id, NodeType.COUNTRY, country,
                EdgeType.LOCATED_IN, list(identity.record_ids),
            )
        for kind, values in identity.identifiers.items():
            for value in values:
                sources = identity.identifier_sources.get(
                    f"{kind}:{value}", list(identity.record_ids)
                )
                self._link_identifier(graph, node_id, kind, value, sources)
        self._link_domains(graph, identity)

        for record_id in identity.record_ids:
            record = records_by_id.get(record_id)
            if record is not None:
                self._emit_record_attributes(graph, node_id, record)

    def _link_domains(
        self, graph: KnowledgeGraph, identity: CompanyIdentity
    ) -> None:
        node_id = company_node_id(identity)
        members = list(identity.record_ids)
        canonical = identity.canonical_domain
        if canonical:
            self._link_attribute(
                graph, node_id, NodeType.DOMAIN, canonical,
                EdgeType.HAS_DOMAIN, members,
            )
        for domain in identity.alternate_domains:
            if not domain or domain == canonical:
                continue
            self._link_attribute(
                graph, node_id, NodeType.DOMAIN, domain,
                EdgeType.HAS_DOMAIN, members,
            )
        for record_id, domain in identity.record_domains.items():
            if not domain or domain == canonical:
                continue
            self._link_attribute(
                graph, node_id, NodeType.DOMAIN, domain,
                EdgeType.HAS_DOMAIN, [record_id],
            )

    def _emit_record_attributes(
        self,
        graph: KnowledgeGraph,
        company_id: str,
        record: DatasetRecord,
    ) -> None:
        rid = record.record_id

        for founder in founder_values(record):
            self._link_attribute(
                graph, company_id, NodeType.FOUNDER, founder,
                EdgeType.FOUNDED_BY, [rid],
            )
        for investor in investor_values(record):
            self._link_attribute(
                graph, company_id, NodeType.INVESTOR, investor,
                EdgeType.INVESTED_BY, [rid],
            )
        for technology in technology_values(record):
            self._link_attribute(
                graph, company_id, NodeType.TECHNOLOGY, technology,
                EdgeType.USES_TECHNOLOGY, [rid],
            )
        for product in product_values(record):
            self._link_attribute(
                graph, company_id, NodeType.PRODUCT, product,
                EdgeType.BUILDS_PRODUCT, [rid],
            )
        for city in _single(record_city(record)):
            self._link_attribute(
                graph, company_id, NodeType.CITY, city,
                EdgeType.LOCATED_IN, [rid],
            )
        for region in _single(record_region(record)):
            self._link_attribute(
                graph, company_id, NodeType.STATE, region,
                EdgeType.LOCATED_IN, [rid],
            )
        for country in _single(record_country(record)):
            self._link_attribute(
                graph, company_id, NodeType.COUNTRY, country,
                EdgeType.LOCATED_IN, [rid],
            )

        for organization in organization_values(record):
            target, node_type = self._resolve_external(organization)
            self._link_external(
                graph, company_id, target, node_type, organization,
                EdgeType.SUBSIDIARY_OF, rid,
            )
        for acquirer in acquirer_values(record):
            target, node_type = self._resolve_external(acquirer)
            self._link_external(
                graph, company_id, target, node_type, acquirer,
                EdgeType.ACQUIRED_BY, rid,
            )
        for related in related_values(record):
            related_id = self._name_index.resolve(related)
            if related_id is None or related_id == company_id:
                continue
            graph.add_edge(
                GraphEdge(
                    edge_type=EdgeType.RELATED_TO,
                    source_id=company_id,
                    target_id=related_id,
                    properties={"related_name": related},
                    sources=[rid],
                )
            )

    # ---- Outcome-derived edges ----

    def _emit_outcome_edges(
        self,
        graph: KnowledgeGraph,
        outcomes: Mapping[str, OutcomeRecord],
        identity_by_record: dict[str, str],
    ) -> None:
        for record_id in sorted(outcomes):
            company_id = identity_by_record.get(record_id)
            if company_id is None:
                continue
            outcome = outcomes[record_id]
            investors = set(outcome.outcome.investors)
            for event in outcome.outcome.funding_rounds:
                investors.update(event.investors)
            for investor in sorted(investors):
                self._link_attribute(
                    graph, company_id, NodeType.INVESTOR, investor,
                    EdgeType.INVESTED_BY, [record_id],
                )
            acquirer = outcome.outcome.acquisition
            if acquirer:
                target, node_type = self._resolve_external(acquirer)
                self._link_external(
                    graph, company_id, target, node_type, acquirer,
                    EdgeType.ACQUIRED_BY, record_id,
                )

    # ---- Alias edges between identities ----

    def _emit_alias_edges(
        self, graph: KnowledgeGraph, identities: list[CompanyIdentity]
    ) -> None:
        buckets: dict[str, list[CompanyIdentity]] = {}
        for identity in identities:
            keys: set[str] = set()
            core = core_name(identity.canonical_name)
            if core:
                keys.add(f"core:{core}")
            canonical_key = canonical_name_key(identity.canonical_name)
            if canonical_key:
                keys.add(f"name:{canonical_key}")
            for alias in identity.aliases:
                alias_key = canonical_name_key(alias)
                if alias_key:
                    keys.add(f"name:{alias_key}")
            for key in keys:
                buckets.setdefault(key, []).append(identity)

        pairs: dict[tuple[str, str], set[str]] = {}
        for entries in buckets.values():
            by_node = {company_node_id(entry): entry for entry in entries}
            node_ids = sorted(by_node)
            if len(node_ids) < 2:
                continue
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    pair = (node_ids[i], node_ids[j])
                    matched = pairs.setdefault(pair, set())
                    matched.add(by_node[pair[0]].canonical_name)
                    matched.add(by_node[pair[1]].canonical_name)

        for pair in sorted(pairs):
            left, right = pair
            left_identity = self._identity_by_node[left]
            right_identity = self._identity_by_node[right]
            sources = sorted(
                set(
                    [*left_identity.record_ids, *right_identity.record_ids]
                )
            )
            graph.add_edge(
                GraphEdge(
                    edge_type=EdgeType.ALIAS_OF,
                    source_id=left,
                    target_id=right,
                    properties={"matched_names": sorted(pairs[pair])},
                    sources=sources,
                )
            )

    # ---- Helpers ----

    def _link_attribute(
        self,
        graph: KnowledgeGraph,
        company_id: str,
        node_type: NodeType,
        label: str,
        edge_type: EdgeType,
        sources: list[str],
    ) -> None:
        node_id = attribute_node_id(node_type, label)
        graph.add_node(
            GraphNode(
                node_id=node_id,
                node_type=node_type,
                label=label,
                sources=list(sources),
            )
        )
        graph.add_edge(
            GraphEdge(
                edge_type=edge_type,
                source_id=company_id,
                target_id=node_id,
                sources=list(sources),
            )
        )

    def _link_identifier(
        self,
        graph: KnowledgeGraph,
        company_id: str,
        kind: str,
        value: str,
        sources: list[str],
    ) -> None:
        node_id = identifier_node_id(kind, value)
        graph.add_node(
            GraphNode(
                node_id=node_id,
                node_type=NodeType.IDENTIFIER,
                label=value,
                properties={"kind": kind},
                sources=list(sources),
            )
        )
        graph.add_edge(
            GraphEdge(
                edge_type=EdgeType.HAS_IDENTIFIER,
                source_id=company_id,
                target_id=node_id,
                properties={"kind": kind},
                sources=list(sources),
            )
        )

    def _resolve_external(self, name: str) -> tuple[str, NodeType]:
        """Resolve a referenced name to a company or new organization node."""
        resolved = self._name_index.resolve(name)
        if resolved is not None:
            return resolved, NodeType.COMPANY
        return attribute_node_id(NodeType.ORGANIZATION, name), NodeType.ORGANIZATION

    def _link_external(
        self,
        graph: KnowledgeGraph,
        company_id: str,
        target_id: str,
        target_type: NodeType,
        label: str,
        edge_type: EdgeType,
        record_id: str,
    ) -> None:
        if target_id == company_id:
            return
        if target_type == NodeType.ORGANIZATION:
            graph.add_node(
                GraphNode(
                    node_id=target_id,
                    node_type=NodeType.ORGANIZATION,
                    label=label,
                    sources=[record_id],
                )
            )
        graph.add_edge(
            GraphEdge(
                edge_type=edge_type,
                source_id=company_id,
                target_id=target_id,
                properties={_reference_property(edge_type): label},
                sources=[record_id],
            )
        )

    def _build_report(
        self,
        records: list[DatasetRecord],
        identities: list[CompanyIdentity],
        graph: KnowledgeGraph,
    ) -> GraphBuildReport:
        node_counts: dict[str, int] = {}
        for node in graph.nodes():
            node_counts[node.node_type.value] = (
                node_counts.get(node.node_type.value, 0) + 1
            )
        edge_counts: dict[str, int] = {}
        for edge in graph.edges():
            edge_counts[edge.edge_type.value] = (
                edge_counts.get(edge.edge_type.value, 0) + 1
            )
        return GraphBuildReport(
            record_count=len(records),
            identity_count=len(identities),
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            node_counts=node_counts,
            edge_counts=edge_counts,
            sources=sorted({record.source for record in records}),
            provenance_issues=graph.validate(),
            graph_key=hashlib.sha256(
                canonical_graph_json(graph).encode("utf-8")
            ).hexdigest(),
        )


def _identity_by_record(
    identities: list[CompanyIdentity],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for identity in identities:
        node_id = company_node_id(identity)
        for record_id in identity.record_ids:
            mapping[record_id] = node_id
    return mapping


def _reference_property(edge_type: EdgeType) -> str:
    if edge_type == EdgeType.ACQUIRED_BY:
        return "acquirer"
    if edge_type == EdgeType.SUBSIDIARY_OF:
        return "parent"
    return "reference"


def _single(value: str | None) -> list[str]:
    return [value] if value else []
