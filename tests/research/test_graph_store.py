"""Tests for the append-only graph store and its record index.

Covers :class:`MemoryIndex`, :class:`MemoryRecordKey`, and
:class:`GraphStore`: node merging, edge strengthening, endpoint
validation, structural stats, integrity validation, and serialization.
"""

from __future__ import annotations

import pytest

from predictron_engine.research.exceptions import (
    InvalidGraphModelError,
    InvalidMemoryInputError,
    InvalidRelationshipError,
)
from predictron_engine.research.graph_models import (
    EntityType,
    GraphEntity,
    GraphRelationship,
    RelationshipType,
    ValidationMarker,
    ValidationStatus,
)
from predictron_engine.research.graph_store import (
    GraphDirection,
    GraphStore,
    GraphValidation,
    MemoryIndex,
    MemoryRecordKey,
)

from .memory_helpers import make_entity, make_relationship

PASSED = ValidationMarker.from_status(ValidationStatus.PASSED, 0.9)


def acme(**overrides: object) -> GraphEntity:
    values: dict[str, object] = {"evidence_ids": ("ev-1",)}
    values.update(overrides)
    return make_entity(
        "Acme",
        EntityType.COMPANY,
        identifier="ent_company_acme",
        **values,  # type: ignore[arg-type]
    )


def gpt(**overrides: object) -> GraphEntity:
    values: dict[str, object] = {"evidence_ids": ("ev-1",)}
    values.update(overrides)
    return make_entity(
        "GPT",
        EntityType.TECHNOLOGY,
        identifier="ent_technology_gpt",
        **values,  # type: ignore[arg-type]
    )


def core(**overrides: object) -> GraphEntity:
    values: dict[str, object] = {"evidence_ids": ("ev-1",)}
    values.update(overrides)
    return make_entity(
        "Core",
        EntityType.TEAM,
        identifier="ent_team_core",
        **values,  # type: ignore[arg-type]
    )


def uses_technology(**overrides: object) -> GraphRelationship:
    values: dict[str, object] = {"evidence_ids": ("ev-1",)}
    values.update(overrides)
    return make_relationship(
        "ent_company_acme",
        "ent_technology_gpt",
        RelationshipType.HAS_TECHNOLOGY,
        **values,  # type: ignore[arg-type]
    )


# -- MemoryRecordKey -------------------------------------------------


def test_record_key_exposes_its_composite_identity() -> None:
    key = MemoryRecordKey(record_type="report", record_id="rep-1", sequence=1)
    assert key.composite == ("report", "rep-1")
    assert key.topic_id == ""
    assert key.stored_at == ""


def test_record_key_rejects_an_empty_identity() -> None:
    with pytest.raises(InvalidMemoryInputError):
        MemoryRecordKey(record_type="  ", record_id="rep-1", sequence=1)


def test_record_key_rejects_a_sequence_below_one() -> None:
    with pytest.raises(InvalidMemoryInputError):
        MemoryRecordKey(record_type="report", record_id="rep-1", sequence=0)


def test_record_key_rejects_a_non_integer_sequence() -> None:
    with pytest.raises(InvalidMemoryInputError):
        MemoryRecordKey(record_type="report", record_id="rep-1", sequence="1")  # type: ignore[arg-type]


def test_record_key_round_trips() -> None:
    key = MemoryRecordKey(
        record_type="report",
        record_id="rep-1",
        sequence=3,
        topic_id="topic-1",
        company_name="Acme",
        stored_at="2026-01-01",
    )
    assert MemoryRecordKey.from_dict(key.to_dict()) == key


def test_record_key_rejects_a_foreign_shape() -> None:
    with pytest.raises(InvalidMemoryInputError):
        MemoryRecordKey.from_dict({"record_type": "report"})


# -- MemoryIndex -----------------------------------------------------


def test_index_hands_out_strictly_increasing_sequences() -> None:
    index = MemoryIndex()
    assert index.next_sequence() == 1
    first = index.allocate("report", "rep-1")
    second = index.allocate("report", "rep-2")
    assert (first.sequence, second.sequence) == (1, 2)
    assert index.next_sequence() == 3


def test_index_is_idempotent_for_one_record() -> None:
    index = MemoryIndex()
    first = index.allocate("report", "rep-1", stored_at="2026-01-01")
    repeat = index.allocate("report", "rep-1", stored_at="2026-09-09")
    assert repeat is first
    assert len(index) == 1
    assert index.next_sequence() == 2


def test_index_lists_keys_in_sequence_order() -> None:
    index = MemoryIndex()
    index.allocate("report", "rep-2")
    index.allocate("entity", "ent-1")
    index.allocate("report", "rep-1")
    assert [key.record_id for key in index.keys()] == ["rep-2", "ent-1", "rep-1"]
    assert [key.record_id for key in index.keys("report")] == ["rep-2", "rep-1"]
    assert index.record_ids("report") == ("rep-2", "rep-1")
    assert index.keys("evidence") == ()


def test_index_reports_the_latest_key_of_a_kind() -> None:
    index = MemoryIndex()
    index.allocate("report", "rep-1")
    assert index.latest("report") is not None
    assert index.latest("report").record_id == "rep-1"
    assert index.latest("evidence") is None


def test_index_looks_keys_up_and_tests_membership() -> None:
    index = MemoryIndex()
    index.allocate("report", "rep-1", topic_id="topic-1")
    assert index.key_for("report", "rep-1") is not None
    assert index.key_for("report", "rep-1").topic_id == "topic-1"  # type: ignore[union-attr]
    assert index.key_for("report", "nope") is None
    assert index.contains("report", "rep-1") is True
    assert index.contains("report", "nope") is False
    assert ("report", "rep-1") in index
    assert ("report", "nope") not in index


def test_index_ignores_a_malformed_membership_probe() -> None:
    index = MemoryIndex()
    index.allocate("report", "rep-1")
    assert "report" not in index
    assert ("report",) not in index


def test_index_requires_an_identity_to_allocate() -> None:
    index = MemoryIndex()
    with pytest.raises(InvalidMemoryInputError):
        index.allocate("report", "  ")


def test_index_round_trips_with_its_sequences() -> None:
    index = MemoryIndex()
    index.allocate("report", "rep-1", company_name="Acme", stored_at="2026-01-01")
    index.allocate("entity", "ent-1")
    rebuilt = MemoryIndex.from_dict(index.to_dict())
    assert rebuilt.keys() == index.keys()
    assert rebuilt.next_sequence() == index.next_sequence()
    assert rebuilt.allocate("report", "rep-1").sequence == 1


def test_index_rejects_a_foreign_payload() -> None:
    with pytest.raises(InvalidMemoryInputError):
        MemoryIndex.from_dict({"keys": "nope"})


# -- GraphStore: nodes -----------------------------------------------


def test_store_starts_empty() -> None:
    store = GraphStore()
    assert store.entities() == ()
    assert store.relationships() == ()
    assert store.entity_ids() == ()
    assert store.relationship_count() == 0
    assert store.get_entity("ent_company_acme") is None
    assert store.has_entity("ent_company_acme") is False
    assert store.entity_type("ent_company_acme") is None


def test_add_entity_reports_a_creation() -> None:
    store = GraphStore()
    write = store.add_entity(acme(), stored_at="2026-01-01", record_id="rep-1")
    assert write.created is True
    assert write.changed is False
    assert write.previous is None
    assert write.sequence == 1
    assert store.get_entity("ent_company_acme") == write.entity
    assert store.entity_type("ent_company_acme") is EntityType.COMPANY
    assert store.has_entity("ent_company_acme") is True
    assert store.index.latest("entity") is not None
    assert store.index.latest("entity").stored_at == "2026-01-01"  # type: ignore[union-attr]


def test_add_entity_merges_rather_than_replaces() -> None:
    store = GraphStore()
    store.add_entity(acme(attributes={"headcount": 10}, confidence=0.4))
    write = store.add_entity(
        acme(
            attributes={"city": "Paris"},
            confidence=0.8,
            evidence_ids=("ev-2",),
        ),
        record_id="rep-2",
    )
    assert write.created is False
    assert write.changed is True
    assert write.previous is not None
    assert write.previous.attribute_map() == {"headcount": 10}
    assert write.entity.attribute_map() == {"city": "Paris", "headcount": 10}
    assert write.entity.confidence == 0.8
    assert write.entity.provenance.evidence_ids == ("ev-1", "ev-2")
    assert write.entity.record_id == "rep-2"
    assert store.entity_ids() == ("ent_company_acme",)


def test_add_entity_learns_a_new_name_as_an_alias() -> None:
    store = GraphStore()
    store.add_entity(acme())
    write = store.add_entity(
        make_entity(
            "Acme Corporation",
            EntityType.COMPANY,
            identifier="ent_company_acme",
            evidence_ids=("ev-2",),
        )
    )
    assert write.entity.aliases == ("Acme Corporation",)


def test_add_entity_does_not_learn_its_own_name_as_an_alias() -> None:
    store = GraphStore()
    store.add_entity(acme())
    write = store.add_entity(acme())
    assert write.entity.aliases == ()
    assert write.changed is False


def test_add_entity_keeps_the_strongest_verdict() -> None:
    store = GraphStore()
    store.add_entity(acme(validation=PASSED))
    write = store.add_entity(acme())
    assert write.entity.validation is PASSED


def test_add_entity_refuses_to_change_an_entity_type() -> None:
    store = GraphStore()
    store.add_entity(acme())
    with pytest.raises(InvalidGraphModelError):
        store.add_entity(
            make_entity("Acme", EntityType.TEAM, identifier="ent_company_acme")
        )


def test_add_entity_rejects_a_foreign_object() -> None:
    with pytest.raises(InvalidGraphModelError):
        GraphStore().add_entity("Acme")  # type: ignore[arg-type]


def test_store_filters_nodes_by_type() -> None:
    store = GraphStore()
    store.add_entity(acme())
    store.add_entity(gpt())
    assert [entity.entity_id for entity in store.entities_of_type(EntityType.COMPANY)] == [
        "ent_company_acme"
    ]
    assert store.entities_of_type(EntityType.RISK) == ()


def test_update_entity_merges_into_a_known_node() -> None:
    store = GraphStore()
    store.add_entity(acme())
    write = store.update_entity(
        "ent_company_acme",
        attributes={"headcount": 42},
        validation=PASSED,
        aliases=("Acme Inc",),
    )
    assert write.changed is True
    assert write.entity.attribute("headcount") == 42
    assert write.entity.validation is PASSED
    assert write.entity.aliases == ("Acme Inc",)
    assert store.index.latest("entity") is not None
    assert store.index.latest("entity").record_id == "ent_company_acme"  # type: ignore[union-attr]


def test_update_entity_refuses_to_invent_a_fact() -> None:
    with pytest.raises(InvalidGraphModelError):
        GraphStore().update_entity("ent_company_acme", confidence=0.5)


# -- GraphStore: edges -----------------------------------------------


def test_add_relationship_reports_a_creation() -> None:
    store = GraphStore()
    store.add_entity(acme())
    store.add_entity(gpt())
    write = store.add_relationship(uses_technology(), stored_at="2026-01-01")
    assert write.created is True
    assert write.strengthened is False
    assert write.previous is None
    assert store.relationship_count() == 1
    assert store.has_relationship(
        "ent_company_acme",
        RelationshipType.HAS_TECHNOLOGY,
        "ent_technology_gpt",
    )


def test_add_relationship_unions_support_without_rewriting() -> None:
    store = GraphStore()
    store.add_entity(acme())
    store.add_entity(gpt())
    store.add_relationship(uses_technology(evidence_ids=("ev-1",), confidence=0.4))
    write = store.add_relationship(
        uses_technology(evidence_ids=("ev-2",), confidence=0.8)
    )
    assert write.created is False
    assert write.strengthened is True
    assert write.previous is not None
    assert write.previous.evidence_ids == ("ev-1",)
    assert write.relationship.evidence_ids == ("ev-1", "ev-2")
    assert write.relationship.confidence == 0.8
    assert store.relationship_count() == 1


def test_add_relationship_requires_both_endpoints() -> None:
    store = GraphStore()
    store.add_entity(acme())
    with pytest.raises(InvalidRelationshipError):
        store.add_relationship(uses_technology())


def test_add_relationship_requires_permitted_endpoint_types() -> None:
    store = GraphStore()
    store.add_entity(acme())
    store.add_entity(core())
    with pytest.raises(InvalidRelationshipError):
        store.add_relationship(
            make_relationship(
                "ent_company_acme",
                "ent_team_core",
                RelationshipType.HAS_TECHNOLOGY,
                evidence_ids=("ev-1",),
            )
        )


def test_add_relationship_rejects_a_foreign_object() -> None:
    with pytest.raises(InvalidRelationshipError):
        GraphStore().add_relationship("acme -> gpt")  # type: ignore[arg-type]


def test_store_indexes_incident_edges_per_direction() -> None:
    store = GraphStore()
    store.add_entity(acme())
    store.add_entity(gpt())
    edge = store.add_relationship(uses_technology()).relationship
    assert store.incident_keys("ent_company_acme") == (edge.key,)
    assert store.incident_keys("ent_technology_gpt") == ()
    assert store.incident_keys(
        "ent_technology_gpt", GraphDirection.INCOMING
    ) == (edge.key,)
    assert store.incident_keys("ent_technology_gpt", GraphDirection.BOTH) == (
        edge.key,
    )
    assert store.incident_keys("ent_missing") == ()


def test_store_rejects_a_foreign_direction() -> None:
    with pytest.raises(InvalidGraphModelError):
        GraphStore().incident_keys("ent_company_acme", "outgoing")  # type: ignore[arg-type]


def test_store_walks_an_edge_to_its_other_end() -> None:
    store = GraphStore()
    store.add_entity(acme())
    store.add_entity(gpt())
    edge = store.add_relationship(uses_technology()).relationship
    assert store.endpoints(edge.key) == ("ent_company_acme", "ent_technology_gpt")
    assert (
        store.other_endpoint(edge.key, "ent_company_acme") == "ent_technology_gpt"
    )
    assert store.other_endpoint(edge.key, "ent_technology_gpt") == "ent_company_acme"


def test_store_looks_edges_up_by_key() -> None:
    store = GraphStore()
    store.add_entity(acme())
    store.add_entity(gpt())
    edge = store.add_relationship(uses_technology()).relationship
    assert store.relationships_by_key(edge.key) == edge
    assert store.get_relationship(
        "ent_company_acme", RelationshipType.HAS_TECHNOLOGY, "ent_technology_gpt"
    ) == edge
    assert store.relationships_by_key(("x", "has_technology", "y")) is None
    assert store.has_relationship(
        "ent_company_acme", RelationshipType.HAS_PRODUCT, "ent_technology_gpt"
    ) is False


# -- GraphStore: stats and validation -------------------------------


def test_store_counts_its_structure() -> None:
    store = GraphStore()
    store.add_entity(acme(validation=PASSED, topic_id="topic-1"))
    store.add_entity(gpt())
    store.add_entity(
        make_entity("ev-1", EntityType.EVIDENCE, identifier="ent_evidence_ev-1")
    )
    store.add_relationship(uses_technology())
    stats = store.stats()
    assert stats.entity_count == 3
    assert stats.business_entity_count == 2
    assert stats.evidence_node_count == 1
    assert stats.relationship_count == 1
    assert stats.validated_entity_count == 1
    assert stats.referenced_evidence_count == 1
    assert stats.topics == ("topic-1",)
    assert stats.count_for_entity_type(EntityType.COMPANY) == 1
    assert stats.count_for_relationship(RelationshipType.HAS_TECHNOLOGY) == 1


def test_store_validates_a_well_formed_graph() -> None:
    store = GraphStore()
    store.add_entity(acme())
    store.add_entity(gpt())
    store.add_relationship(uses_technology())
    report = store.validate()
    assert report.valid is True
    assert report.issues() == ()
    assert report.issue_count == 0


def test_store_rejects_a_document_with_a_dangling_edge() -> None:
    document = GraphStore().to_dict()
    document["relationships"] = [uses_technology().to_dict()]
    with pytest.raises(InvalidRelationshipError):
        GraphStore.from_dict(document)


def test_store_rejects_a_document_with_an_ungrounded_node() -> None:
    document = GraphStore().to_dict()
    document["entities"] = [
        make_entity(
            "Acme",
            EntityType.COMPANY,
            identifier="ent_company_acme",
            report_ids=(),
        ).to_dict()
    ]
    report = GraphStore.from_dict(document).validate()
    assert report.valid is False
    assert "ent_company_acme" in report.missing_provenance
    assert report.issue_count == 1


def test_graph_validation_round_trips() -> None:
    report = GraphValidation(dangling=("a",), missing_provenance=("b",))
    assert GraphValidation.from_dict(report.to_dict()) == report
    assert report.valid is False


# -- GraphStore: serialization ---------------------------------------


def test_store_round_trips_through_a_dictionary() -> None:
    store = GraphStore()
    store.add_entity(acme(attributes={"headcount": 10}), record_id="rep-1")
    store.add_entity(gpt())
    store.add_relationship(uses_technology())
    document = store.to_dict()
    rebuilt = GraphStore.from_dict(document)
    assert rebuilt.entities() == store.entities()
    assert rebuilt.relationships() == store.relationships()
    assert rebuilt.to_dict() == document
    assert rebuilt.validate().valid is True
    assert rebuilt.index.keys() == store.index.keys()


def test_store_rejects_a_foreign_document() -> None:
    with pytest.raises(InvalidMemoryInputError):
        GraphStore.from_dict([])  # type: ignore[arg-type]


def test_store_rejects_a_foreign_entity_list() -> None:
    with pytest.raises(InvalidMemoryInputError):
        GraphStore.from_dict({"entities": "nope"})


def test_store_rejects_a_foreign_relationship_list() -> None:
    with pytest.raises(InvalidMemoryInputError):
        GraphStore.from_dict({"relationships": [7]})
