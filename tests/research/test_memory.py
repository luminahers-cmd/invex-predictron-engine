"""Tests for the public research memory facade — Phase 8 Sprint 6.

Covers :class:`ResearchMemory`: the single deterministic transaction that
turns one report plus its evidence and observations into nodes, edges,
history entries, and a stable memory key.
"""

from __future__ import annotations

import pytest

from predictron_engine.research.exceptions import (
    InvalidMemoryInputError,
    InvalidRelationshipError,
)
from predictron_engine.research.graph_models import (
    EntityType,
    RelationshipType,
    ValidationMarker,
    ValidationStatus,
)
from predictron_engine.research.graph_store import GraphDirection, MemoryIndex
from predictron_engine.research.history import HistoryLog
from predictron_engine.research.knowledge_graph import KnowledgeGraph
from predictron_engine.research.memory import (
    EntityObservation,
    RelationshipObservation,
    ResearchMemory,
)

from .memory_helpers import make_evidence, make_report

PASSED = ValidationMarker.from_status(ValidationStatus.PASSED, 0.9)


def gpt_observation(
    evidence_id: str,
    *,
    name: str = "GPT",
    **overrides: object,
) -> EntityObservation:
    values: dict[str, object] = {
        "attributes": {"generation": 4},
        "evidence_ids": (evidence_id,),
        "confidence": 0.8,
    }
    values.update(overrides)
    return EntityObservation(name, EntityType.TECHNOLOGY, **values)  # type: ignore[arg-type]


def has_technology(evidence_id: str, **overrides: object) -> RelationshipObservation:
    values: dict[str, object] = {"evidence_ids": (evidence_id,), "confidence": 0.7}
    values.update(overrides)
    return RelationshipObservation(
        relationship_type=RelationshipType.HAS_TECHNOLOGY,
        source_name="Acme",
        source_type=EntityType.COMPANY,
        target_name="GPT",
        target_type=EntityType.TECHNOLOGY,
        **values,  # type: ignore[arg-type]
    )


def ingest(
    memory: ResearchMemory,
    report_id: str = "rep-1",
    *,
    company_name: str = "Acme",
    **evidence_kwargs: object,
) -> tuple[object, str]:
    """Add one report with one evidence item, returning the evidence id."""
    evidence = make_evidence(**evidence_kwargs)  # type: ignore[arg-type]
    write = memory.add_report(
        make_report(report_id, company_name=company_name),
        evidence=[evidence],
    )
    return write, evidence.evidence_id


# -- construction ----------------------------------------------------


def test_memory_starts_empty() -> None:
    memory = ResearchMemory()
    assert memory.is_empty() is True
    assert memory.counts() == {
        "reports": 0,
        "evidence": 0,
        "duplicates": 0,
        "indexed": 0,
        "entities": 0,
        "relationships": 0,
        "history": 0,
    }
    assert memory.reports() == ()
    assert memory.entities() == ()
    assert memory.entity_ids() == ()
    assert memory.evidence_ids() == ()
    assert memory.get_company("Acme") is None
    assert memory.get_report("rep-1") is None
    assert memory.get_entity("Acme") is None
    assert memory.get_history("Acme") == ()
    assert memory.get_changes("Acme") == ()


def test_memory_rejects_foreign_collaborators() -> None:
    with pytest.raises(InvalidMemoryInputError):
        ResearchMemory(index="index")  # type: ignore[arg-type]
    with pytest.raises(InvalidMemoryInputError):
        ResearchMemory(graph="graph")  # type: ignore[arg-type]
    with pytest.raises(InvalidMemoryInputError):
        ResearchMemory(history="history")  # type: ignore[arg-type]


def test_memory_shares_one_index_across_its_stores() -> None:
    memory = ResearchMemory()
    assert memory.records.index is memory.index
    assert memory.graph.store.index is memory.index


# -- the ingestion transaction ---------------------------------------


def test_a_report_alone_creates_only_the_company() -> None:
    memory = ResearchMemory()
    write, _ = ingest(memory)
    assert write.created is True
    assert write.report_id == "rep-1"
    assert write.evidence_ids
    assert write.created_entity_ids
    assert memory.get_company("Acme") is not None
    assert memory.get_company("Acme").entity_type is EntityType.COMPANY  # type: ignore[union-attr]
    assert memory.get_relationships() == ()


def test_carrying_evidence_creates_an_evidence_node() -> None:
    memory = ResearchMemory()
    _, evidence_id = ingest(memory)
    evidence_nodes = [
        entity for entity in memory.entities() if entity.is_evidence
    ]
    assert len(evidence_nodes) == 1
    assert evidence_nodes[0].evidence_ids == (evidence_id,)
    assert evidence_nodes[0].entity_type is EntityType.EVIDENCE


def test_observations_create_nodes_and_edges() -> None:
    memory = ResearchMemory()
    _, evidence_id = ingest(memory)
    write = memory.add_report(
        make_report("rep-2"),
        evidence=[make_evidence(claim="Acme uses GPT")],
        entities=[gpt_observation(evidence_id)],
        relationships=[has_technology(evidence_id)],
    )
    assert write.created_entity_ids
    assert write.created_relationship_ids
    assert memory.get_entity("GPT") is not None
    edges = memory.get_relationships("Acme", RelationshipType.HAS_TECHNOLOGY)
    assert len(edges) == 1
    assert edges[0].evidence_ids == (evidence_id,)


def test_a_fact_is_grounded_in_the_evidence_the_report_carried() -> None:
    memory = ResearchMemory()
    with pytest.raises(InvalidMemoryInputError):
        memory.add_report(
            make_report("rep-1"),
            evidence=[make_evidence()],
            entities=[gpt_observation("ev_missing")],
        )
    assert memory.get_entity("GPT") is None


def test_an_entity_observation_must_cite_evidence() -> None:
    memory = ResearchMemory()
    with pytest.raises(InvalidMemoryInputError):
        memory.add_report(
            make_report("rep-1"),
            entities=[EntityObservation("GPT", EntityType.TECHNOLOGY)],
        )


def test_a_relationship_observation_must_cite_evidence() -> None:
    memory = ResearchMemory()
    _, evidence_id = ingest(memory)
    memory.add_report(make_report("rep-2"), entities=[gpt_observation(evidence_id)])
    with pytest.raises(InvalidRelationshipError):
        memory.add_report(
            make_report("rep-3"),
            relationships=[has_technology(evidence_id, evidence_ids=())],
        )


def test_a_relationship_must_join_nodes_memory_already_holds() -> None:
    memory = ResearchMemory()
    _, evidence_id = ingest(memory)
    with pytest.raises(InvalidMemoryInputError):
        memory.add_report(
            make_report("rep-2"),
            relationships=[has_technology(evidence_id)],
        )


def test_a_relationship_keeps_its_declared_endpoint_types() -> None:
    memory = ResearchMemory()
    _, evidence_id = ingest(memory)
    memory.add_report(
        make_report("rep-2"),
        entities=[
            EntityObservation(
                "Core", EntityType.TEAM, evidence_ids=(evidence_id,)
            )
        ],
    )
    with pytest.raises(InvalidRelationshipError):
        memory.add_report(
            make_report("rep-3"),
            relationships=[
                RelationshipObservation(
                    relationship_type=RelationshipType.HAS_TECHNOLOGY,
                    source_name="Acme",
                    source_type=EntityType.COMPANY,
                    target_name="Core",
                    target_type=EntityType.TEAM,
                    evidence_ids=(evidence_id,),
                )
            ],
        )


def test_replaying_a_report_writes_nothing_new() -> None:
    memory = ResearchMemory()
    evidence = make_evidence()
    first = memory.add_report(
        make_report("rep-1"),
        evidence=[evidence],
        entities=[gpt_observation(evidence.evidence_id)],
        relationships=[has_technology(evidence.evidence_id)],
    )
    counts = memory.counts()
    second = memory.add_report(
        make_report("rep-1"),
        evidence=[evidence],
        entities=[gpt_observation(evidence.evidence_id)],
        relationships=[has_technology(evidence.evidence_id)],
    )
    assert second.created is False
    assert second.report_id == first.report_id
    assert memory.counts() == counts
    assert len(memory.get_relationships()) == 1


def test_a_report_cannot_be_rewritten() -> None:
    memory = ResearchMemory()
    memory.add_report(make_report("rep-1"))
    with pytest.raises(Exception):
        memory.add_report(make_report("rep-1", company_name="OpenAI"))


def test_standalone_evidence_is_stored_and_returned_by_id() -> None:
    memory = ResearchMemory()
    evidence = make_evidence()
    identifier = memory.add_evidence(evidence)
    assert identifier == evidence.evidence_id
    assert memory.get_evidence(identifier) is evidence
    assert memory.evidence_ids() == (identifier,)
    assert memory.is_empty() is False


# -- merging and history ---------------------------------------------


def test_re_observing_an_entity_updates_it_and_records_history() -> None:
    memory = ResearchMemory()
    _, first = ingest(memory)
    memory.add_report(make_report("rep-2"), entities=[gpt_observation(first)])
    write = memory.add_report(
        make_report("rep-3"),
        entities=[gpt_observation(first, attributes={"generation": 5})],
    )
    assert write.updated_entity_ids
    assert write.history_sequences
    node = memory.get_entity("GPT")
    assert node is not None
    assert node.attribute("generation") == 5
    changes = memory.get_changes("GPT", "generation")
    assert changes
    assert changes[-1].change_for("generation").new_value == "5"  # type: ignore[union-attr]


def test_history_accumulates_across_reports() -> None:
    memory = ResearchMemory()
    _, first = ingest(memory)
    memory.add_report(make_report("rep-2"), entities=[gpt_observation(first)])
    memory.add_report(
        make_report("rep-3"),
        entities=[gpt_observation(first, attributes={"generation": 5})],
    )
    history = memory.get_history("GPT")
    assert len(history) >= 2
    assert history[0].is_creation is True
    assert [entry.sequence for entry in history] == sorted(
        entry.sequence for entry in history
    )


def test_a_new_name_for_a_known_entity_is_recorded_as_history() -> None:
    memory = ResearchMemory()
    _, first = ingest(memory)
    memory.add_report(make_report("rep-2"), entities=[gpt_observation(first)])
    node_id = memory.get_entity("GPT").entity_id  # type: ignore[union-attr]
    memory.add_report(
        make_report("rep-3"),
        entities=[gpt_observation(first, name="GPT Inc.", confidence=0.9)],
    )
    assert memory.get_entity("GPT").entity_id == node_id  # type: ignore[union-attr]
    assert memory.get_entity("GPT Inc.") is not None
    assert "GPT Inc." in memory.get_entity(node_id).aliases  # type: ignore[union-attr]
    assert len(memory.get_history("GPT")) > 1


def test_a_different_canonical_name_is_a_different_entity() -> None:
    memory = ResearchMemory()
    _, first = ingest(memory)
    memory.add_report(make_report("rep-2"), entities=[gpt_observation(first)])
    memory.add_report(
        make_report("rep-3"),
        entities=[gpt_observation(first, name="GPT-4")],
    )
    assert memory.get_entity("GPT") is not None
    assert memory.get_entity("GPT-4") is not None
    assert memory.get_entity("GPT").entity_id != memory.get_entity(  # type: ignore[union-attr]
        "GPT-4"
    ).entity_id

def test_re_observing_the_same_name_writes_no_history() -> None:
    memory = ResearchMemory()
    _, first = ingest(memory)
    memory.add_report(make_report("rep-2"), entities=[gpt_observation(first)])
    before = len(memory.history)
    memory.add_report(make_report("rep-3"), entities=[gpt_observation(first)])
    assert len(memory.history) == before


def test_strengthening_an_edge_is_recorded_in_history() -> None:
    memory = ResearchMemory()
    _, first = ingest(memory)
    second = make_evidence(url="https://example.test/b", claim="Acme files on GPT")
    memory.add_report(
        make_report("rep-2"),
        evidence=[second],
        entities=[gpt_observation(first)],
        relationships=[has_technology(first, confidence=0.5)],
    )
    memory.add_report(
        make_report("rep-3"),
        relationships=[has_technology(second.evidence_id, confidence=0.95)],
    )
    edge = memory.get_relationships("Acme", RelationshipType.HAS_TECHNOLOGY)[0]
    assert set(edge.evidence_ids) == {first, second.evidence_id}
    assert edge.confidence == 0.95
    assert memory.get_history("Acme")


def test_history_is_a_persistent_log() -> None:
    memory = ResearchMemory()
    ingest(memory)
    assert memory.history.is_empty() is False
    assert isinstance(memory.history, HistoryLog)


# -- reads -----------------------------------------------------------


def test_entities_are_readable_by_name_and_identifier() -> None:
    memory = ResearchMemory()
    _, first = ingest(memory)
    memory.add_report(make_report("rep-2"), entities=[gpt_observation(first)])
    node = memory.get_entity("GPT")
    assert node is not None
    assert memory.get_entity(node.entity_id) == node
    assert memory.get_entity("gpt") == node
    assert memory.get_entity("  ") is None
    assert memory.get_entity(7) is None  # type: ignore[arg-type]
    assert memory.get_entity("GPT", EntityType.RISK) is None
    assert memory.get_company("GPT") is None
    assert memory.get_company("acme, inc.") is not None


def test_relationships_can_be_read_for_one_node_or_all() -> None:
    memory = ResearchMemory()
    _, first = ingest(memory)
    memory.add_report(
        make_report("rep-2"),
        entities=[gpt_observation(first)],
        relationships=[has_technology(first)],
    )
    assert len(memory.get_relationships()) == 1
    assert len(memory.get_relationships("Acme")) == 1
    assert len(memory.get_relationships("GPT")) == 1
    assert (
        len(
            memory.get_relationships(
                "Acme", direction=GraphDirection.INCOMING
            )
        )
        == 0
    )
    assert memory.get_relationships("Nowhere") == ()


def test_search_and_stats_are_exposed() -> None:
    memory = ResearchMemory()
    _, first = ingest(memory)
    memory.add_report(make_report("rep-2"), entities=[gpt_observation(first)])
    assert memory.search_entities("GPT")[0].entity_id == memory.get_entity("GPT").entity_id  # type: ignore[union-attr]
    assert memory.search_entities("GPT", EntityType.COMPANY) == ()
    stats = memory.stats()
    assert stats.entity_count == len(memory.entities())
    assert stats.count_for_entity_type(EntityType.COMPANY) == 1


def test_retrieve_latest_normalizes_the_company_name() -> None:
    memory = ResearchMemory()
    ingest(memory, "rep-1", company_name="Acme")
    ingest(memory, "rep-2", company_name="OpenAI Inc.")
    assert [r.report_id for r in memory.retrieve_latest(company_name="Open AI")] == [
        "rep-2"
    ]
    assert memory.retrieve_latest(company_name="Nowhere") == ()
    assert len(memory.retrieve_latest(limit=10)) == 2


def test_retrieve_by_topic_spans_reports_evidence_and_the_graph() -> None:
    memory = ResearchMemory()
    evidence = make_evidence(topic_id="topic-7")
    memory.add_report(
        make_report("rep-1"),
        evidence=[evidence],
        entities=[
            gpt_observation(evidence.evidence_id, topic_id="topic-7"),
            EntityObservation(
                "Rival",
                EntityType.COMPETITOR,
                evidence_ids=(evidence.evidence_id,),
                topic_id="topic-7",
            ),
        ],
        relationships=[
            has_technology(evidence.evidence_id),
            RelationshipObservation(
                relationship_type=RelationshipType.COMPETES_ON_TECHNOLOGY,
                source_name="GPT",
                source_type=EntityType.TECHNOLOGY,
                target_name="Rival",
                target_type=EntityType.COMPETITOR,
                evidence_ids=(evidence.evidence_id,),
            ),
        ],
        topic_id="topic-7",
    )
    retrieval = memory.retrieve_by_topic("topic-7")
    assert retrieval.reports == (memory.get_report("rep-1"),)
    assert retrieval.evidence == (evidence,)
    assert len(retrieval.entity_ids) == 3
    assert len(retrieval.relationship_keys) == 1
    assert retrieval.is_empty() is False
    assert memory.retrieve_by_topic("topic-9").is_empty() is True


def test_a_company_node_is_not_scoped_to_one_topic() -> None:
    memory = ResearchMemory()
    evidence = make_evidence(topic_id="topic-7")
    memory.add_report(make_report("rep-1"), evidence=[evidence], topic_id="topic-7")
    company = memory.get_company("Acme")
    assert company is not None
    assert company.topic_id == ""
    assert memory.get_entity("Acme").entity_id not in memory.retrieve_by_topic(  # type: ignore[union-attr]
        "topic-7"
    ).entity_ids


def test_validation_of_an_ingested_memory_is_clean() -> None:
    memory = ResearchMemory()
    _, first = ingest(memory)
    memory.add_report(
        make_report("rep-2"),
        entities=[gpt_observation(first)],
        relationships=[has_technology(first)],
    )
    assert memory.validate().valid is True


# -- identity, reporting, serialization ------------------------------


def test_memory_key_is_stable_and_content_addressed() -> None:
    def build() -> ResearchMemory:
        memory = ResearchMemory()
        _, first = ingest(memory)
        memory.add_report(
            make_report("rep-2"),
            entities=[gpt_observation(first)],
            relationships=[has_technology(first)],
        )
        return memory

    assert build().memory_key() == build().memory_key()
    assert build().memory_key() != ResearchMemory().memory_key()
    assert build().memory_key().startswith("graph_")


def test_report_summarises_the_whole_memory() -> None:
    memory = ResearchMemory()
    _, first = ingest(memory)
    memory.add_report(
        make_report("rep-2"),
        entities=[gpt_observation(first)],
        relationships=[has_technology(first)],
    )
    report = memory.report()
    assert report["memory_key"] == memory.memory_key()
    assert report["counts"] == memory.counts()
    assert report["validation"]["valid"] is True  # type: ignore[index]


def test_memory_round_trips_through_a_dictionary() -> None:
    memory = ResearchMemory()
    _, first = ingest(memory)
    memory.add_report(
        make_report("rep-2"),
        entities=[gpt_observation(first)],
        relationships=[has_technology(first)],
    )
    document = memory.to_dict()
    rebuilt = ResearchMemory.from_dict(document)
    assert rebuilt.to_dict() == document
    assert rebuilt.memory_key() == memory.memory_key()
    assert rebuilt.counts() == memory.counts()
    assert rebuilt.get_entity("GPT") == memory.get_entity("GPT")
    assert rebuilt.get_company("Acme") == memory.get_company("Acme")
    assert len(rebuilt.get_history("GPT")) == len(memory.get_history("GPT"))
    assert rebuilt.get_evidence(first) is not None
    assert rebuilt.get_report("rep-1") == memory.get_report("rep-1")
    assert rebuilt.records.index is rebuilt.index
    assert rebuilt.index.next_sequence() == memory.index.next_sequence()


def test_a_round_tripped_memory_still_rejects_a_rewritten_report() -> None:
    memory = ResearchMemory()
    ingest(memory)
    rebuilt = ResearchMemory.from_dict(memory.to_dict())
    with pytest.raises(Exception):
        rebuilt.add_report(make_report("rep-1", company_name="OpenAI"))


def test_memory_rejects_a_foreign_snapshot() -> None:
    with pytest.raises(InvalidMemoryInputError):
        ResearchMemory.from_dict([])  # type: ignore[arg-type]
    with pytest.raises(InvalidMemoryInputError):
        ResearchMemory.from_dict({})
    with pytest.raises(InvalidMemoryInputError):
        ResearchMemory.from_dict({"graph": "nope"})


def test_a_memory_can_be_seeded_with_its_collaborators() -> None:
    index = MemoryIndex()
    graph = KnowledgeGraph()
    history = HistoryLog()
    memory = ResearchMemory(index=index, graph=graph, history=history)
    assert memory.index is index
    assert memory.graph is graph
    assert memory.history is history
