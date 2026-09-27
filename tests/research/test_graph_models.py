"""Tests for the immutable graph value objects — Phase 8 Sprint 6.

Covers :class:`Provenance`, :class:`ValidationMarker`,
:class:`EntitySnapshot`, :class:`GraphEntity`,
:class:`GraphRelationship`, :class:`GraphStats`, and the
``RELATIONSHIP_SPECS`` vocabulary that governs endpoint typing.
"""

from __future__ import annotations

import dataclasses

import pytest

from predictron_engine.research.exceptions import (
    InvalidGraphModelError,
    InvalidRelationshipError,
)
from predictron_engine.research.graph_models import (
    BUSINESS_ENTITY_TYPES,
    GRAPH_SCHEMA_VERSION,
    RELATIONSHIP_ID_PREFIX,
    RELATIONSHIP_SPECS,
    EntitySnapshot,
    EntityType,
    EvidenceConfidence,
    GraphEntity,
    GraphRelationship,
    GraphStats,
    Provenance,
    RelationshipType,
    ValidationMarker,
    ValidationStatus,
)

PASSED = ValidationMarker.from_status(ValidationStatus.PASSED, 0.82)
FAILED = ValidationMarker.from_status(ValidationStatus.FAILED, 0.2)
MEDIUM = ValidationMarker.from_status(ValidationStatus.PASSED, 0.4)
SKIPPED = ValidationMarker()


def test_schema_version_is_exported() -> None:
    assert GRAPH_SCHEMA_VERSION == "1.0.0"


def test_vocabulary_is_sixteen_entities_and_thirteen_relationships() -> None:
    assert len(list(EntityType)) == 16
    assert len(list(RelationshipType)) == 13
    assert len(RELATIONSHIP_SPECS) == 13


def test_every_relationship_type_has_a_spec() -> None:
    assert {spec_key.value for spec_key in RELATIONSHIP_SPECS} == {
        relationship_type.value for relationship_type in RelationshipType
    }


def test_evidence_is_the_only_non_business_entity_type() -> None:
    assert BUSINESS_ENTITY_TYPES == frozenset(set(EntityType) - {EntityType.EVIDENCE})
    assert EntityType.EVIDENCE not in BUSINESS_ENTITY_TYPES


def test_every_relationship_spec_declares_both_ends() -> None:
    for spec in RELATIONSHIP_SPECS.values():
        assert spec.source_types
        assert spec.target_types
        assert spec.description.strip()


def test_evidence_can_point_at_any_business_entity() -> None:
    spec = RELATIONSHIP_SPECS[RelationshipType.EVIDENCE_ENTITY]
    assert spec.source_types == frozenset({EntityType.EVIDENCE})
    assert spec.target_types == BUSINESS_ENTITY_TYPES


# -- Provenance ------------------------------------------------------


def test_provenance_starts_empty() -> None:
    assert Provenance().is_empty is True
    assert Provenance(evidence_ids=("ev-1",)).is_empty is False


def test_provenance_union_preserves_order_and_keeps_duplicates_at_source() -> None:
    merged = Provenance(evidence_ids=("ev-1",)).merged_with(
        Provenance(evidence_ids=("ev-2",), urls=("u",))
    )
    assert merged.evidence_ids == ("ev-1", "ev-2")
    assert merged.urls == ("u",)


def test_provenance_rejects_a_foreign_operand() -> None:
    with pytest.raises(InvalidGraphModelError):
        Provenance().merged_with("nope")  # type: ignore[arg-type]


def test_provenance_rejects_a_non_string_evidence_id() -> None:
    with pytest.raises(InvalidGraphModelError):
        Provenance(evidence_ids=(7,))  # type: ignore[arg-type]


def test_provenance_round_trips() -> None:
    provenance = Provenance(
        evidence_ids=("ev-1",),
        source_ids=("src-1",),
        urls=("https://example.invalid/a",),
        titles=("A",),
        report_ids=("rep-1",),
    )
    assert Provenance.from_dict(provenance.to_dict()) == provenance


def test_provenance_rejects_a_foreign_shape() -> None:
    with pytest.raises(InvalidGraphModelError):
        Provenance.from_dict({"evidence_ids": 5})


# -- ValidationMarker ------------------------------------------------


def test_marker_band_follows_the_score() -> None:
    assert PASSED.confidence is EvidenceConfidence.HIGH
    assert MEDIUM.confidence is EvidenceConfidence.LOW
    assert FAILED.confidence is EvidenceConfidence.UNKNOWN


def test_marker_reports_whether_a_fact_was_validated() -> None:
    assert SKIPPED.is_validated is False
    assert PASSED.is_validated is True
    assert SKIPPED.status is ValidationStatus.SKIPPED


def test_marker_rejects_a_foreign_status() -> None:
    with pytest.raises(InvalidGraphModelError):
        ValidationMarker(status="passed")  # type: ignore[arg-type]


def test_marker_rejects_a_score_outside_the_unit_range() -> None:
    with pytest.raises(InvalidGraphModelError):
        ValidationMarker(score=1.5)


def test_marker_round_trips() -> None:
    assert ValidationMarker.from_dict(PASSED.to_dict()) == PASSED


def test_marker_rejects_a_foreign_shape() -> None:
    with pytest.raises(InvalidGraphModelError):
        ValidationMarker.from_dict({"status": "maybe"})


# -- EntitySnapshot --------------------------------------------------


def make_snapshot(**overrides: object) -> EntitySnapshot:
    values: dict[str, object] = {
        "entity_id": "ent_company_acme",
        "entity_type": EntityType.COMPANY,
        "version": 1,
        "recorded_at": "2026-01-01",
        "name": "Acme",
        "attributes": (("headcount", 10),),
        "provenance": Provenance(evidence_ids=("ev-1",)),
        "confidence": 0.7,
        "validation": SKIPPED,
    }
    values.update(overrides)
    return EntitySnapshot(**values)  # type: ignore[arg-type]


def test_snapshot_exposes_its_attributes_as_a_mapping() -> None:
    assert make_snapshot().attribute_map() == {"headcount": 10}


def test_snapshot_rejects_a_version_below_one() -> None:
    with pytest.raises(InvalidGraphModelError):
        make_snapshot(version=0)


def test_snapshot_rejects_malformed_attributes() -> None:
    with pytest.raises(InvalidGraphModelError):
        make_snapshot(attributes=("headcount",))


def test_snapshot_round_trips_with_aliases() -> None:
    snapshot = make_snapshot(aliases=("Acme Inc",))
    assert EntitySnapshot.from_dict(snapshot.to_dict()) == snapshot
    assert EntitySnapshot.from_dict(snapshot.to_dict()).aliases == ("Acme Inc",)


def test_snapshot_treats_absent_aliases_as_empty() -> None:
    payload = make_snapshot().to_dict()
    payload.pop("aliases")
    assert EntitySnapshot.from_dict(payload).aliases == ()


def test_snapshot_rejects_malformed_aliases() -> None:
    with pytest.raises(InvalidGraphModelError):
        make_snapshot(aliases=("ok", 7))


def test_snapshot_rejects_a_foreign_shape() -> None:
    with pytest.raises(InvalidGraphModelError):
        EntitySnapshot.from_dict({"entity_id": "x", "version": "one"})


# -- GraphEntity -----------------------------------------------------


def make_entity(**overrides: object) -> GraphEntity:
    values: dict[str, object] = {
        "entity_id": "ent_company_acme",
        "entity_type": EntityType.COMPANY,
        "name": "Acme",
        "canonical_name": "acme",
    }
    values.update(overrides)
    return GraphEntity(**values)  # type: ignore[arg-type]


def test_entity_is_frozen() -> None:
    entity = make_entity()
    with pytest.raises(dataclasses.FrozenInstanceError):
        entity.name = "Other"  # type: ignore[misc]


def test_entity_rejects_an_unknown_type() -> None:
    with pytest.raises(InvalidGraphModelError):
        make_entity(entity_type="company")


def test_entity_rejects_a_confidence_outside_the_unit_range() -> None:
    with pytest.raises(InvalidGraphModelError):
        make_entity(confidence=1.2)


def test_entity_rejects_malformed_attributes() -> None:
    with pytest.raises(InvalidGraphModelError):
        make_entity(attributes=("headcount", 10))


def test_entity_reads_one_attribute_and_defaults_the_rest() -> None:
    entity = make_entity(attributes=(("headcount", 10),))
    assert entity.attribute("headcount") == 10
    assert entity.attribute("missing") is None
    assert entity.attribute("missing", "fallback") == "fallback"
    assert entity.attribute_map() == {"headcount": 10}


def test_entity_answers_whether_it_is_an_evidence_node() -> None:
    assert make_entity(entity_type=EntityType.EVIDENCE, name="ev-1").is_evidence
    assert make_entity().is_evidence is False


def test_entity_lists_every_name_it_is_known_by() -> None:
    entity = make_entity(aliases=("Acme Inc",))
    assert entity.names == ("Acme", "Acme Inc")
    assert entity.has_alias("Acme Inc") is True
    assert entity.has_alias("Nope") is False


def test_entity_exposes_its_evidence() -> None:
    entity = make_entity(provenance=Provenance(evidence_ids=("ev-1", "ev-2")))
    assert entity.evidence_ids == ("ev-1", "ev-2")


def test_entity_knows_the_relationships_it_may_hold() -> None:
    entity = make_entity()
    assert entity.supports_relationship(RelationshipType.HAS_TECHNOLOGY) is True
    assert entity.accepts_relationship(RelationshipType.COMPETES_WITH) is False
    assert entity.supports_relationship(RelationshipType.EVIDENCE_ENTITY) is False


def test_entity_updates_merge_without_mutating() -> None:
    entity = make_entity(attributes=(("headcount", 10),))
    updated = entity.with_updates(attributes={"headcount": 20, "city": "Paris"})
    assert updated is not entity
    assert entity.attribute_map() == {"headcount": 10}
    assert updated.attribute_map() == {"headcount": 20, "city": "Paris"}


def test_entity_updates_union_provenance() -> None:
    entity = make_entity(provenance=Provenance(evidence_ids=("ev-1",)))
    updated = entity.with_updates(provenance=Provenance(evidence_ids=("ev-2",)))
    assert entity.provenance.evidence_ids == ("ev-1",)
    assert updated.provenance.evidence_ids == ("ev-1", "ev-2")


def test_entity_updates_extend_aliases_without_repeating() -> None:
    entity = make_entity(aliases=("Acme Inc",))
    updated = entity.with_updates(aliases=("Acme Inc", "Acme Corporation"))
    assert updated.aliases == ("Acme Inc", "Acme Corporation")


def test_entity_updates_ignore_its_own_name_as_an_alias() -> None:
    assert make_entity().with_updates(aliases=("Acme",)).aliases == ()


def test_entity_updates_keep_the_strongest_verdict() -> None:
    entity = make_entity(validation=PASSED)
    assert entity.with_updates(validation=SKIPPED).validation is PASSED
    assert entity.with_updates(validation=FAILED).validation is FAILED


def test_entity_updates_accept_a_verdict_when_none_was_held() -> None:
    entity = make_entity()
    assert entity.with_updates(validation=FAILED).validation is FAILED


def test_entity_updates_carry_the_writing_record() -> None:
    assert make_entity().with_updates(record_id="rep-1").record_id == "rep-1"
    assert make_entity(record_id="rep-1").with_updates().record_id == "rep-1"


def test_entity_snapshot_carries_names_and_version() -> None:
    entity = make_entity(aliases=("Acme Inc",), attributes=(("city", "Paris"),))
    snapshot = entity.snapshot(4, "2026-04-01")
    assert snapshot.version == 4
    assert snapshot.recorded_at == "2026-04-01"
    assert snapshot.aliases == ("Acme Inc",)
    assert snapshot.attribute_map() == {"city": "Paris"}
    assert snapshot.provenance == entity.provenance


def test_entity_round_trips() -> None:
    entity = make_entity(
        aliases=("Acme Inc",),
        attributes=(("headcount", 10),),
        provenance=Provenance(evidence_ids=("ev-1",), urls=("u",)),
        confidence=0.6,
        validation=PASSED,
        topic_id="topic-1",
        record_id="rep-1",
    )
    assert GraphEntity.from_dict(entity.to_dict()) == entity


def test_entity_rejects_a_foreign_shape() -> None:
    with pytest.raises(InvalidGraphModelError):
        GraphEntity.from_dict({"entity_id": "x", "entity_type": "widget"})


# -- GraphRelationship -----------------------------------------------


def make_relationship(**overrides: object) -> GraphRelationship:
    values: dict[str, object] = {
        "source_id": "ent_company_acme",
        "target_id": "ent_technology_gpt",
        "relationship_type": RelationshipType.HAS_TECHNOLOGY,
        "evidence_ids": ("ev-1",),
    }
    values.update(overrides)
    return GraphRelationship(**values)  # type: ignore[arg-type]


def test_relationship_is_frozen() -> None:
    relationship = make_relationship()
    with pytest.raises(dataclasses.FrozenInstanceError):
        relationship.confidence = 0.9  # type: ignore[misc]


def test_relationship_rejects_identical_endpoints() -> None:
    with pytest.raises(InvalidRelationshipError):
        make_relationship(target_id="ent_company_acme")


def test_relationship_requires_at_least_one_evidence_id() -> None:
    with pytest.raises(InvalidRelationshipError):
        make_relationship(evidence_ids=())


def test_relationship_identifier_is_stable_and_prefixed() -> None:
    relationship = make_relationship()
    assert relationship.relationship_id() == relationship.relationship_id()
    assert relationship.relationship_id().startswith(RELATIONSHIP_ID_PREFIX)
    assert relationship.key == (
        "ent_company_acme",
        RelationshipType.HAS_TECHNOLOGY.value,
        "ent_technology_gpt",
    )


def test_relationship_identifier_is_direction_sensitive() -> None:
    relationship = make_relationship()
    reverse = make_relationship(
        source_id="ent_technology_gpt",
        target_id="ent_company_acme",
    )
    assert relationship.is_symmetric is False
    assert relationship.relationship_id() != reverse.relationship_id()


def test_relationship_accepts_a_permitted_pair_of_ends() -> None:
    make_relationship().check_endpoints(EntityType.COMPANY, EntityType.TECHNOLOGY)


def test_relationship_rejects_a_forbidden_pair_of_ends() -> None:
    with pytest.raises(InvalidRelationshipError):
        make_relationship().check_endpoints(EntityType.TEAM, EntityType.TECHNOLOGY)


def test_relationship_rejects_unknown_endpoint_types() -> None:
    with pytest.raises(InvalidRelationshipError):
        make_relationship().check_endpoints("company", EntityType.TECHNOLOGY)  # type: ignore[arg-type]


def test_relationship_support_can_only_strengthen() -> None:
    relationship = make_relationship(confidence=0.8, evidence_ids=("ev-1",))
    strengthened = relationship.with_support(
        evidence_ids=("ev-2",),
        confidence=0.4,
        timestamp="2026-02-01",
    )
    assert relationship.evidence_ids == ("ev-1",)
    assert strengthened.evidence_ids == ("ev-1", "ev-2")
    assert strengthened.confidence == 0.8
    assert strengthened.timestamp == "2026-02-01"


def test_relationship_support_raises_confidence_when_evidence_improves() -> None:
    assert (
        make_relationship(confidence=0.3).with_support(confidence=0.9).confidence
        == 0.9
    )


def test_relationship_support_ignores_repeated_evidence() -> None:
    relationship = make_relationship(evidence_ids=("ev-1",))
    assert relationship.with_support(evidence_ids=("ev-1",)).evidence_ids == ("ev-1",)


def test_relationship_support_merges_attributes_and_provenance() -> None:
    relationship = make_relationship(
        attributes=(("since", 2024),),
        provenance=Provenance(evidence_ids=("ev-1",), urls=("u1",)),
    )
    strengthened = relationship.with_support(
        evidence_ids=("ev-2",),
        attributes={"role": "core"},
        provenance=Provenance(urls=("u2",)),
    )
    assert strengthened.attribute_map() == {"role": "core", "since": 2024}
    assert strengthened.provenance.evidence_ids == ("ev-1", "ev-2")
    assert strengthened.provenance.urls == ("u1", "u2")


def test_relationship_rejects_a_confidence_outside_the_unit_range() -> None:
    with pytest.raises(InvalidRelationshipError):
        make_relationship(confidence=-0.1)


def test_relationship_round_trips() -> None:
    relationship = make_relationship(
        confidence=0.5,
        provenance=Provenance(evidence_ids=("ev-1",), urls=("u",)),
        timestamp="2026-01-01",
        evidence_ids=("ev-1",),
        attributes=(("role", "core"),),
    )
    assert GraphRelationship.from_dict(relationship.to_dict()) == relationship


def test_relationship_rejects_a_foreign_shape() -> None:
    with pytest.raises(InvalidGraphModelError):
        GraphRelationship.from_dict({"source_id": "a", "target_id": "a"})


# -- GraphStats ------------------------------------------------------


def make_stats() -> GraphStats:
    return GraphStats(
        entity_count=3,
        relationship_count=1,
        evidence_node_count=1,
        business_entity_count=2,
        validated_entity_count=1,
        referenced_evidence_count=1,
        entity_counts=((EntityType.COMPANY.value, 2), (EntityType.EVIDENCE.value, 1)),
        relationship_counts=((RelationshipType.HAS_TECHNOLOGY.value, 1),),
        topics=("topic-1",),
    )


def test_stats_count_by_type_and_default_to_zero() -> None:
    stats = make_stats()
    assert stats.count_for_entity_type(EntityType.COMPANY) == 2
    assert stats.count_for_entity_type(EntityType.RISK) == 0
    assert stats.count_for_relationship(RelationshipType.HAS_TECHNOLOGY) == 1
    assert stats.count_for_relationship(RelationshipType.HAS_RISK) == 0


def test_stats_reject_negative_counts() -> None:
    with pytest.raises(InvalidGraphModelError):
        GraphStats(
            entity_count=-1,
            relationship_count=0,
            evidence_node_count=0,
            business_entity_count=0,
            validated_entity_count=0,
            referenced_evidence_count=0,
        )


def test_stats_round_trip() -> None:
    assert GraphStats.from_dict(make_stats().to_dict()) == make_stats()


def test_stats_reject_a_foreign_shape() -> None:
    with pytest.raises(InvalidGraphModelError):
        GraphStats.from_dict({"entity_count": "three"})
