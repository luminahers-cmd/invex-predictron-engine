"""Tests for the read side of the research graph — Phase 8 Sprint 6.

Covers :class:`KnowledgeGraph`: alias-aware lookup, deterministic search,
breadth-first traversal, shortest paths, isolated subgraphs, structural
reporting, the content-addressed graph key, and serialization.
"""

from __future__ import annotations

import pytest

from predictron_engine.research.exceptions import (
    InvalidGraphModelError,
    InvalidMemoryInputError,
)
from predictron_engine.research.graph_models import (
    EntityType,
    GraphEntity,
    RelationshipType,
)
from predictron_engine.research.graph_store import GraphDirection
from predictron_engine.research.knowledge_graph import (
    KNOWLEDGE_GRAPH_SCHEMA_VERSION,
    GraphNeighbor,
    KnowledgeGraph,
    SearchScope,
    TraversalResult,
)
from predictron_engine.research.linking import LINKING_VERSION, entity_id

from .memory_helpers import make_entity, make_relationship

ACME = entity_id("company", "Acme")
OPENAI = entity_id("company", "OpenAI")
GPT = entity_id("technology", "GPT")
RIVAL = entity_id("competitor", "Rival")


def company(name: str, identifier: str, **overrides: object) -> GraphEntity:
    values: dict[str, object] = {"evidence_ids": ("ev-1",)}
    values.update(overrides)
    return make_entity(name, EntityType.COMPANY, identifier=identifier, **values)  # type: ignore[arg-type]


def edge(
    source_id: str,
    target_id: str,
    relationship_type: RelationshipType,
    **overrides: object,
) -> object:
    values: dict[str, object] = {"evidence_ids": ("ev-1",)}
    values.update(overrides)
    return make_relationship(source_id, target_id, relationship_type, **values)  # type: ignore[arg-type]


def build_graph() -> KnowledgeGraph:
    """A graph shaped Acme -> GPT -> Rival, with Acme also competing with Rival."""
    graph = KnowledgeGraph()
    graph.store.add_entity(company("Acme", ACME, topic_id="topic-1"))
    graph.store.add_entity(
        company("Acme Corporation", ACME, attributes={"country": "FR"})
    )
    graph.store.add_entity(company("OpenAI", OPENAI, topic_id="topic-2"))
    graph.store.add_entity(
        make_entity("GPT", EntityType.TECHNOLOGY, identifier=GPT, evidence_ids=("ev-1",))
    )
    graph.store.add_entity(
        make_entity("Rival", EntityType.COMPETITOR, identifier=RIVAL, evidence_ids=("ev-1",))
    )
    graph.store.add_relationship(  # type: ignore[arg-type]
        edge(ACME, GPT, RelationshipType.HAS_TECHNOLOGY)
    )
    graph.store.add_relationship(  # type: ignore[arg-type]
        edge(GPT, RIVAL, RelationshipType.COMPETES_ON_TECHNOLOGY)
    )
    graph.store.add_relationship(  # type: ignore[arg-type]
        edge(ACME, RIVAL, RelationshipType.COMPETES_WITH)
    )
    return graph


# -- construction and basics ----------------------------------------


def test_graph_starts_empty() -> None:
    graph = KnowledgeGraph()
    assert graph.is_empty() is True
    assert graph.entities() == ()
    assert graph.relationships() == ()
    assert graph.entity_ids() == ()
    assert graph.relationship_count() == 0
    assert graph.get_entity("ent_company_acme") is None
    assert graph.entity_type("ent_company_acme") is None
    assert graph.resolve(EntityType.COMPANY, "Acme") is None


def test_graph_reads_through_to_its_store() -> None:
    graph = build_graph()
    assert graph.is_empty() is False
    assert graph.get_entity(ACME) is not None
    assert graph.entity_type(ACME) is EntityType.COMPANY
    assert graph.entity_type(GPT) is EntityType.TECHNOLOGY
    assert graph.relationship_count() == 3
    assert [entity.entity_id for entity in graph.companies()] == sorted(
        (ACME, OPENAI)
    )
    assert len(graph.entities_of_type(EntityType.COMPANY)) == 2
    assert graph.validate().valid is True


# -- lookup ----------------------------------------------------------


def test_lookup_finds_a_node_by_its_canonical_name() -> None:
    assert build_graph().get_entity_by_name("Acme") is not None
    assert build_graph().get_company("acme") is not None


def test_lookup_normalizes_punctuation_and_case() -> None:
    graph = build_graph()
    assert graph.get_entity_by_name("ACME, Inc.") is not None
    assert graph.get_company("  acme  ") is not None


def test_lookup_finds_a_node_by_a_learned_alias() -> None:
    graph = build_graph()
    assert graph.get_entity_by_name("Acme Corp") is not None
    assert graph.get_company("ACME CORPORATION") is not None


def test_lookup_never_lets_types_shadow_each_other() -> None:
    graph = KnowledgeGraph()
    graph.store.add_entity(company("Nova", "ent_company_nova"))
    graph.store.add_entity(
        make_entity(
            "Nova",
            EntityType.PRODUCT,
            identifier=entity_id(EntityType.PRODUCT.value, "Nova"),
            evidence_ids=("ev-1",),
        )
    )
    assert graph.get_entity_by_name("Nova", EntityType.COMPANY) is not None
    assert graph.get_entity_by_name("Nova", EntityType.PRODUCT) is not None
    assert graph.get_company("Nova") is not None
    assert graph.get_entity_by_name("Nova", EntityType.RISK) is None


def test_lookup_of_an_unknown_or_blank_name_is_none() -> None:
    graph = build_graph()
    assert graph.get_entity_by_name("Nowhere") is None
    assert graph.get_entity_by_name("   ") is None
    assert graph.get_company("Nowhere") is None
    assert graph.get_entity_by_name(7) is None  # type: ignore[arg-type]


def test_resolution_rebuilds_from_a_serialized_graph() -> None:
    graph = KnowledgeGraph.from_dict(build_graph().to_dict())
    assert graph.resolve(EntityType.COMPANY, "Acme") == ACME
    assert graph.resolve(EntityType.COMPANY, "Acme Corporation") == ACME
    assert graph.resolve(EntityType.COMPANY, "Nowhere") is None


def test_resolution_of_an_unseen_name_records_nothing() -> None:
    graph = build_graph()
    before = graph.entity_ids()
    assert graph.resolve(EntityType.COMPANY, "Nowhere") is None
    assert graph.entity_ids() == before


def test_relationship_lookup_and_filtering() -> None:
    graph = build_graph()
    edge_type = RelationshipType.HAS_TECHNOLOGY
    assert graph.get_relationship(ACME, edge_type, GPT) is not None
    assert graph.get_relationship(ACME, RelationshipType.HAS_PRODUCT, GPT) is None
    assert len(graph.get_relationships()) == 3
    assert len(graph.get_relationships(relationship_type=edge_type)) == 1
    assert len(graph.get_relationships(entity_id=ACME)) == 2
    assert len(graph.outgoing_relationships(ACME)) == 2
    assert len(graph.incoming_relationships(ACME)) == 0
    assert (
        len(
            graph.get_relationships(
                entity_id=ACME, direction=GraphDirection.INCOMING
            )
        )
        == 0
    )
    assert (
        len(
            graph.get_relationships(
                entity_id=GPT, direction=GraphDirection.INCOMING
            )
        )
        == 1
    )
    assert (
        len(graph.get_relationships(entity_id=GPT, direction=GraphDirection.BOTH)) == 2
    )


def test_companies_can_be_filtered_by_topic() -> None:
    graph = build_graph()
    assert [entity.entity_id for entity in graph.companies_for_topic("topic-1")] == [ACME]
    assert graph.companies_for_topic(" topic-2 ") == graph.companies_for_topic("topic-2")
    assert graph.companies_for_topic("topic-9") == ()


# -- search ----------------------------------------------------------


def test_search_ranks_an_exact_name_first() -> None:
    results = build_graph().search("Acme")
    assert results[0].entity_id == ACME


def test_search_is_scope_limited_to_names() -> None:
    graph = build_graph()
    assert graph.search("Acme", scope=SearchScope.NAME)[0].entity_id == ACME
    assert graph.search("FR", scope=SearchScope.NAME) == ()
    assert graph.search("FR", scope=SearchScope.ATTRIBUTES)[0].entity_id == ACME
    assert graph.search("FR")[0].entity_id == ACME


def test_search_can_be_narrowed_to_a_type() -> None:
    graph = build_graph()
    assert [e.entity_id for e in graph.search("GPT", EntityType.TECHNOLOGY)] == [GPT]
    assert graph.search("GPT", EntityType.COMPANY) == ()
    assert graph.search_by_name("GPT", EntityType.RISK) == ()


def test_search_honours_its_limit() -> None:
    assert len(build_graph().search("a", limit=1)) == 1
    assert build_graph().search("nothing matches this", limit=5) == ()


def test_search_of_a_blank_query_is_empty() -> None:
    assert build_graph().search("   ") == ()


# -- traversal -------------------------------------------------------


def test_traversal_returns_neighbours_at_the_shallowest_depth() -> None:
    result = build_graph().traverse(ACME, max_depth=2)
    assert result.start_id == ACME
    assert result.has_entity(GPT)
    assert result.depth_of(GPT) == 1
    assert result.depth_of(RIVAL) == 1
    assert result.entity_ids() == tuple(sorted(result.entity_ids()))
    assert result.relationship_count() == 3
    assert result.evidence_ids() == ("ev-1",)


def test_traversal_reports_each_entity_once() -> None:
    result = build_graph().traverse(GPT, max_depth=3)
    assert len(result.neighbors) == len({n.entity_id for n in result.neighbors})
    assert result.of_type(EntityType.TECHNOLOGY) == ()
    assert result.depth_of(ACME) == 1


def test_traversal_can_follow_one_direction_only() -> None:
    graph = build_graph()
    assert len(graph.traverse(ACME, max_depth=1, direction=GraphDirection.OUTGOING).neighbors) == 2
    assert graph.traverse(ACME, max_depth=1, direction=GraphDirection.INCOMING).neighbors == ()
    assert len(graph.traverse(GPT, max_depth=1, direction=GraphDirection.INCOMING).neighbors) == 1


def test_traversal_can_follow_one_relationship_only() -> None:
    graph = build_graph()
    result = graph.traverse(
        GPT, max_depth=2, relationship_type=RelationshipType.HAS_TECHNOLOGY
    )
    assert [n.entity_id for n in result.neighbors] == [ACME]


def test_traversal_of_an_unknown_node_is_empty() -> None:
    result = build_graph().traverse("ent_missing", max_depth=2)
    assert result.neighbors == ()
    assert result.entity_ids() == ()


def test_traversal_rejects_a_malformed_depth() -> None:
    graph = build_graph()
    with pytest.raises(InvalidGraphModelError):
        graph.traverse(ACME, max_depth=-1)
    with pytest.raises(InvalidGraphModelError):
        graph.traverse(ACME, max_depth="two")  # type: ignore[arg-type]
    with pytest.raises(InvalidGraphModelError):
        graph.neighbors(ACME, depth=0)


def test_neighbors_are_deterministically_ordered() -> None:
    first = build_graph().neighbors(ACME, depth=2)
    second = build_graph().neighbors(ACME, depth=2)
    assert [n.sort_key for n in first] == [n.sort_key for n in second]
    assert [n.depth for n in first] == sorted(n.depth for n in first)
    assert build_graph().neighbors("ent_missing") == ()


def test_shortest_path_finds_the_bridge_between_two_nodes() -> None:
    assert build_graph().shortest_path(ACME, RIVAL) == (ACME, RIVAL)
    assert build_graph().shortest_path(ACME, GPT) == (ACME, GPT)


def test_shortest_path_of_a_disconnected_or_unknown_node_is_none() -> None:
    graph = build_graph()
    assert graph.shortest_path(ACME, OPENAI) is None
    assert graph.shortest_path(ACME, "ent_missing") is None
    assert graph.shortest_path("ent_missing", ACME) is None


def test_shortest_path_from_a_node_to_itself_is_that_node() -> None:
    assert build_graph().shortest_path(GPT, GPT) == (GPT,)


def test_shortest_path_respects_direction() -> None:
    graph = build_graph()
    assert graph.shortest_path(GPT, ACME, GraphDirection.OUTGOING) is None
    assert graph.shortest_path(GPT, ACME, GraphDirection.INCOMING) == (GPT, ACME)


# -- subgraph --------------------------------------------------------


def test_subgraph_keeps_only_the_requested_nodes_and_their_edges() -> None:
    graph = build_graph()
    sub = graph.subgraph([ACME, GPT])
    assert sub.entity_ids() == tuple(sorted((ACME, GPT)))
    assert sub.relationship_count() == 1
    assert sub.get_entity(ACME) == graph.get_entity(ACME)


def test_subgraph_ignores_unknown_nodes_and_duplicates() -> None:
    assert build_graph().subgraph([ACME, "ent_missing", ACME]).entity_ids() == (ACME,)


def test_subgraph_can_narrow_to_one_relationship_kind() -> None:
    graph = build_graph()
    sub = graph.subgraph([ACME, RIVAL, GPT], relationship_type=RelationshipType.COMPETES_WITH)
    assert sub.relationship_count() == 1
    assert sub.get_relationship(
        ACME, RelationshipType.COMPETES_WITH, RIVAL
    ) is not None


def test_subgraph_is_isolated_from_its_parent() -> None:
    graph = build_graph()
    sub = graph.subgraph([ACME, GPT])
    sub.store.update_entity(ACME, confidence=0.1)
    assert graph.get_entity(ACME).confidence == 0.9  # type: ignore[union-attr]
    assert sub.get_entity(ACME).confidence == 0.1  # type: ignore[union-attr]
    assert sub.get_entity_by_name("Acme Corp") is not None


# -- identity and reporting ------------------------------------------


def test_graph_key_is_stable_for_the_same_content() -> None:
    assert build_graph().graph_key() == build_graph().graph_key()
    assert build_graph().graph_key().startswith("graph_")


def test_graph_key_is_content_addressed_and_prefixed() -> None:
    empty = KnowledgeGraph()
    assert empty.graph_key() == KnowledgeGraph().graph_key()
    assert empty.graph_key() != build_graph().graph_key()


def test_graph_key_changes_when_the_graph_changes() -> None:
    graph = build_graph()
    before = graph.graph_key()
    graph.store.add_entity(company("Nova", "ent_company_nova"))
    assert graph.graph_key() != before


def test_canonical_json_is_stable_and_serializable() -> None:
    import json

    graph = build_graph()
    document = graph.canonical_json()
    assert document == build_graph().canonical_json()
    assert json.loads(document)["schema_version"]
    assert build_graph().graph_key() == graph.graph_key()


def test_report_summarises_structure_and_validity() -> None:
    report = build_graph().report()
    assert report["schema_version"] == KNOWLEDGE_GRAPH_SCHEMA_VERSION
    assert report["linking_version"] == LINKING_VERSION
    assert report["graph_key"] == build_graph().graph_key()
    assert report["stats"]["entity_count"] == 4  # type: ignore[index]
    assert report["validation"]["valid"] is True  # type: ignore[index]


def test_stats_are_exposed() -> None:
    stats = build_graph().stats()
    assert stats.entity_count == 4
    assert stats.relationship_count == 3
    assert stats.count_for_entity_type(EntityType.COMPANY) == 2


# -- serialization ---------------------------------------------------


def test_graph_round_trips_through_a_dictionary() -> None:
    graph = build_graph()
    document = graph.to_dict()
    rebuilt = KnowledgeGraph.from_dict(document)
    assert rebuilt.to_dict() == document
    assert rebuilt.graph_key() == graph.graph_key()
    assert rebuilt.get_entity(ACME) is not None
    assert rebuilt.get_company("Acme Corp") is not None


def test_graph_rejects_a_foreign_document() -> None:
    with pytest.raises(InvalidMemoryInputError):
        KnowledgeGraph.from_dict([])  # type: ignore[arg-type]


def test_neighbor_round_trips() -> None:
    neighbor = GraphNeighbor(
        entity_id=GPT,
        entity_type=EntityType.TECHNOLOGY,
        name="GPT",
        relationship_type=RelationshipType.HAS_TECHNOLOGY,
        confidence=0.9,
        depth=1,
        evidence_ids=("ev-1",),
    )
    assert GraphNeighbor.from_dict(neighbor.to_dict()) == neighbor


def test_traversal_result_round_trips() -> None:
    result = build_graph().traverse(ACME, max_depth=2)
    assert TraversalResult.from_dict(result.to_dict()) == result
