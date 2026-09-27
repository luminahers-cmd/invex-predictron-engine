"""Tests for the immutable history layer — Phase 8 Sprint 6.

Covers :class:`HistoryChange`, :class:`HistoryEntry`, the persistent
:class:`HistoryLog`, and the deterministic :func:`derive_changes` diff.
"""

from __future__ import annotations

import pytest

from predictron_engine.research.exceptions import (
    ImmutableRecordError,
    InvalidGraphModelError,
    InvalidMemoryInputError,
)
from predictron_engine.research.graph_models import EntityType, ValidationMarker
from predictron_engine.research.history import (
    CHANGE_KIND_ADDED,
    CHANGE_KIND_REMOVED,
    CHANGE_KIND_UNCHANGED,
    CHANGE_KIND_UPDATED,
    HISTORY_ENTRY_PREFIX,
    HISTORY_SCHEMA_VERSION,
    HistoryChange,
    HistoryEntry,
    HistoryLog,
    derive_changes,
    history_entry_id,
)
from predictron_engine.research.validation_models import ValidationStatus

from .memory_helpers import make_entity

MARKER = ValidationMarker.from_status(ValidationStatus.PASSED, 0.8)


def test_history_entry_id_is_stable_and_prefixed() -> None:
    first = history_entry_id("ent_company_1", 4)
    assert first == history_entry_id("ent_company_1", 4)
    assert first != history_entry_id("ent_company_1", 5)
    assert first.startswith(HISTORY_ENTRY_PREFIX)


def test_schema_version_is_exported() -> None:
    assert HISTORY_SCHEMA_VERSION == "1.0.0"


def test_change_kinds_are_the_documented_strings() -> None:
    assert CHANGE_KIND_ADDED == "added"
    assert CHANGE_KIND_REMOVED == "removed"
    assert CHANGE_KIND_UPDATED == "updated"
    assert CHANGE_KIND_UNCHANGED == "unchanged"


def make_change(previous, new, field_name="headcount") -> HistoryChange:
    return HistoryChange(
        field_name=field_name,
        previous_value=previous,
        new_value=new,
        timestamp="2026-01-01",
        source="rep-1",
    )


def test_change_reports_an_added_field() -> None:
    change = make_change(None, "1200")
    assert change.kind == CHANGE_KIND_ADDED
    assert change.was_absent is True
    assert change.is_creation is True
    assert change.changed is True


def test_change_reports_a_removed_field() -> None:
    change = make_change("1200", None)
    assert change.kind == CHANGE_KIND_REMOVED
    assert change.is_removed is True
    assert change.is_removal is True


def test_change_reports_an_updated_field() -> None:
    change = make_change("1200", "1500")
    assert change.kind == CHANGE_KIND_UPDATED
    assert change.is_transition is True


def test_change_reports_an_unchanged_field() -> None:
    change = make_change("1200", "1200")
    assert change.kind == CHANGE_KIND_UNCHANGED
    assert change.changed is False


def test_change_distinguishes_an_empty_value_from_absence() -> None:
    change = make_change("", "1200")
    assert change.kind == CHANGE_KIND_UPDATED
    assert change.was_absent is False


def test_change_requires_a_field_name() -> None:
    with pytest.raises(InvalidGraphModelError):
        make_change(None, "1", field_name="   ")


def test_change_requires_a_timestamp() -> None:
    with pytest.raises(InvalidGraphModelError):
        HistoryChange(
            field_name="headcount",
            previous_value=None,
            new_value="1",
            timestamp=None,  # type: ignore[arg-type]
            source="rep-1",
        )


def test_change_rejects_a_non_string_value() -> None:
    with pytest.raises(InvalidGraphModelError):
        make_change(1200, "1500")  # type: ignore[arg-type]


def test_change_round_trips() -> None:
    change = HistoryChange(
        field_name="headcount",
        previous_value="1200",
        new_value="1500",
        timestamp="2026-01-01",
        source="rep-1",
        evidence_ids=("ev-1",),
        validation_status=ValidationStatus.PASSED,
    )
    assert HistoryChange.from_dict(change.to_dict()) == change


def test_change_rejects_a_foreign_shape() -> None:
    with pytest.raises(InvalidGraphModelError):
        HistoryChange.from_dict({"field_name": 7})


def make_entry(
    entity,
    sequence: int = 1,
    previous=None,
    changes: tuple[HistoryChange, ...] = (),
    evidence_ids: tuple[str, ...] = (),
) -> HistoryEntry:
    return HistoryEntry(
        entity_id=entity.entity_id,
        entity_type=entity.entity_type,
        sequence=sequence,
        recorded_at=f"2026-01-{sequence:02d}",
        source=f"rep-{sequence}",
        previous=previous,
        current=entity.snapshot(sequence, f"2026-01-{sequence:02d}"),
        changes=changes,
        evidence_ids=evidence_ids,
    )


def test_entry_of_a_creation_has_no_previous_snapshot() -> None:
    entity = make_entity("GPT", evidence_ids=("ev-1",))
    entry = make_entry(entity, evidence_ids=("ev-1",))
    assert entry.is_creation is True
    assert entry.version == 1
    assert entry.entry_id == history_entry_id(entity.entity_id, 1)


def test_entry_exposes_its_derived_version() -> None:
    entity = make_entity("GPT")
    entry = make_entry(entity, sequence=7)
    assert entry.version == 7


def test_entry_finds_one_change_by_field() -> None:
    entity = make_entity("GPT")
    change = make_change("4", "5", field_name="generation")
    entry = make_entry(
        entity,
        sequence=2,
        previous=make_entity("GPT").snapshot(1, "2026-01-01"),
        changes=(change,),
    )
    assert entry.change_for("generation") is change
    assert entry.change_for("missing") is None
    assert entry.changed_fields() == ("generation",)


def test_entry_reports_no_changed_fields_for_a_creation() -> None:
    assert make_entry(make_entity("GPT")).changed_fields() == ()


def test_entry_requires_a_known_entity_type() -> None:
    entity = make_entity("GPT")
    with pytest.raises(InvalidGraphModelError):
        HistoryEntry(
            entity_id=entity.entity_id,
            entity_type="technology",  # type: ignore[arg-type]
            sequence=1,
            recorded_at="2026-01-01",
            source="rep-1",
            previous=None,
            current=entity.snapshot(1, "2026-01-01"),
            changes=(),
        )


def test_entry_round_trips() -> None:
    entity = make_entity("GPT", evidence_ids=("ev-1",), validation=MARKER)
    entry = make_entry(
        entity,
        sequence=2,
        previous=make_entity("GPT", evidence_ids=("ev-1",)).snapshot(
            1, "2026-01-01"
        ),
        changes=(make_change("4", "5", field_name="generation"),),
        evidence_ids=("ev-1",),
    )
    document = entry.to_dict()
    assert HistoryEntry.from_dict(document) == entry


def test_entry_rejects_a_foreign_shape() -> None:
    with pytest.raises(InvalidGraphModelError):
        HistoryEntry.from_dict({"entity_id": "x", "sequence": "one"})


def test_derive_changes_of_a_creation_reports_nothing() -> None:
    entity = make_entity("GPT", attributes={"headcount": 10})
    assert (
        derive_changes(
            previous=None,
            current=entity.snapshot(1, "2026-01-01"),
            timestamp="2026-01-01",
            source="rep-1",
        )
        == ()
    )


def test_derive_changes_reports_a_changed_attribute() -> None:
    before = make_entity("GPT", attributes={"headcount": 10}).snapshot(
        1, "2026-01-01"
    )
    after = make_entity("GPT", attributes={"headcount": 20}).snapshot(2, "2026-01-02")
    changes = derive_changes(
        previous=before,
        current=after,
        timestamp="2026-01-02",
        source="rep-2",
    )
    assert [change.field_name for change in changes] == ["headcount"]
    assert changes[0].previous_value == "10"
    assert changes[0].new_value == "20"
    assert changes[0].kind == CHANGE_KIND_UPDATED


def test_derive_changes_reports_an_added_attribute() -> None:
    before = make_entity("GPT", attributes={}).snapshot(1, "2026-01-01")
    after = make_entity("GPT", attributes={"generation": 5}).snapshot(2, "2026-01-02")
    changes = derive_changes(
        previous=before,
        current=after,
        timestamp="2026-01-02",
        source="rep-2",
    )
    assert changes[0].kind == CHANGE_KIND_ADDED
    assert changes[0].was_absent is True


def test_derive_changes_reports_a_removed_attribute() -> None:
    before = make_entity("GPT", attributes={"generation": 5}).snapshot(
        1, "2026-01-01"
    )
    after = make_entity("GPT", attributes={}).snapshot(2, "2026-01-02")
    changes = derive_changes(
        previous=before,
        current=after,
        timestamp="2026-01-02",
        source="rep-2",
    )
    assert changes[0].kind == CHANGE_KIND_REMOVED


def test_derive_changes_reports_a_confidence_move() -> None:
    before = make_entity("GPT", confidence=0.5).snapshot(1, "2026-01-01")
    after = make_entity("GPT", confidence=0.9).snapshot(2, "2026-01-02")
    changes = derive_changes(
        previous=before,
        current=after,
        timestamp="2026-01-02",
        source="rep-2",
    )
    assert [change.field_name for change in changes] == ["confidence"]


def test_derive_changes_reports_a_validation_move() -> None:
    before = make_entity("GPT").snapshot(1, "2026-01-01")
    after = make_entity("GPT", validation=MARKER).snapshot(2, "2026-01-02")
    changes = derive_changes(
        previous=before,
        current=after,
        timestamp="2026-01-02",
        source="rep-2",
    )
    assert [change.field_name for change in changes] == ["validation"]


def test_derive_changes_reports_nothing_when_nothing_moved() -> None:
    before = make_entity("GPT", attributes={"headcount": 10}).snapshot(
        1, "2026-01-01"
    )
    after = make_entity("GPT", attributes={"headcount": 10}).snapshot(2, "2026-01-02")
    assert (
        derive_changes(
            previous=before,
            current=after,
            timestamp="2026-01-02",
            source="rep-2",
        )
        == ()
    )


def test_derive_changes_is_ordered_by_field_name() -> None:
    before = make_entity("GPT", attributes={"a": 1, "b": 2}).snapshot(1, "2026-01-01")
    after = make_entity("GPT", attributes={"a": 9, "b": 9}).snapshot(2, "2026-01-02")
    changes = derive_changes(
        previous=before,
        current=after,
        timestamp="2026-01-02",
        source="rep-2",
    )
    assert [change.field_name for change in changes] == ["a", "b"]


def test_derive_changes_treats_a_type_change_as_a_change() -> None:
    before = make_entity("GPT", attributes={"mature": 1}).snapshot(1, "2026-01-01")
    after = make_entity("GPT", attributes={"mature": True}).snapshot(2, "2026-01-02")
    changes = derive_changes(
        previous=before,
        current=after,
        timestamp="2026-01-02",
        source="rep-2",
    )
    assert [change.field_name for change in changes] == ["mature"]


def record(log: HistoryLog, entity, sequence: int, previous=None) -> HistoryLog:
    return log.record(
        entity_id=entity.entity_id,
        entity_type=entity.entity_type,
        sequence=sequence,
        recorded_at=f"2026-01-{sequence:02d}",
        source=f"rep-{sequence}",
        previous=previous,
        current=entity.snapshot(sequence, f"2026-01-{sequence:02d}"),
    )


def test_log_records_entries_in_sequence_order() -> None:
    entity = make_entity("GPT")
    log = record(HistoryLog(), entity, 1)
    log = record(log, entity, 2, previous=entity.snapshot(1, "2026-01-01"))
    assert [entry.sequence for entry in log.ordered()] == [1, 2]
    assert log.entries() == log.ordered()
    assert [entry.sequence for entry in log] == [1, 2]


def test_log_is_persistent_rather_than_mutated() -> None:
    entity = make_entity("GPT")
    empty = HistoryLog()
    first = record(empty, entity, 1)
    assert empty.is_empty() is True
    assert first.is_empty() is False


def test_log_derives_changes_when_none_are_supplied() -> None:
    entity = make_entity("GPT", attributes={"headcount": 10})
    log = record(HistoryLog(), entity, 1)
    log = log.record(
        entity_id=entity.entity_id,
        entity_type=EntityType.TECHNOLOGY,
        sequence=2,
        recorded_at="2026-01-02",
        source="rep-2",
        previous=entity.snapshot(1, "2026-01-01"),
        current=make_entity("GPT", attributes={"headcount": 20}).snapshot(
            2, "2026-01-02"
        ),
    )
    assert log.latest_for(entity.entity_id).changed_fields() == ("headcount",)


def test_log_collects_the_evidence_behind_an_entry() -> None:
    entity = make_entity("GPT", evidence_ids=("ev-1",))
    log = record(HistoryLog(), entity, 1)
    assert log.latest_for(entity.entity_id).evidence_ids == ("ev-1",)


def test_log_is_empty_before_anything_is_recorded() -> None:
    log = HistoryLog()
    assert log.is_empty() is True
    assert len(log) == 0
    assert log.ordered() == ()
    assert list(log) == []


def test_log_filters_by_entity() -> None:
    first = make_entity("GPT")
    second = make_entity("Acme", EntityType.COMPANY)
    log = record(HistoryLog(), first, 1)
    log = record(log, second, 2)
    assert log.for_entity(first.entity_id) == log.ordered()[:1]
    assert log.for_entity("ent_missing") == ()
    assert log.entities() == tuple(sorted((first.entity_id, second.entity_id)))


def test_log_reports_the_latest_and_first_entry() -> None:
    entity = make_entity("GPT")
    log = record(HistoryLog(), entity, 1)
    log = record(log, entity, 2, previous=entity.snapshot(1, "2026-01-01"))
    assert log.latest_for(entity.entity_id).sequence == 2
    assert log.first_for(entity.entity_id).sequence == 1
    assert log.latest_for("ent_missing") is None
    assert log.first_for("ent_missing") is None


def test_log_lists_snapshots_for_one_entity() -> None:
    entity = make_entity("GPT")
    log = HistoryLog()
    for version in (1, 2, 3):
        log = record(log, entity, version)
    assert [snap.version for snap in log.snapshots_for(entity.entity_id)] == [1, 2, 3]
    assert log.snapshots_for("ent_missing") == ()


def test_log_filters_changes_by_field() -> None:
    entity = make_entity("GPT", attributes={"headcount": 10})
    log = record(HistoryLog(), entity, 1)
    log = log.record(
        entity_id=entity.entity_id,
        entity_type=EntityType.TECHNOLOGY,
        sequence=2,
        recorded_at="2026-01-02",
        source="rep-2",
        previous=entity.snapshot(1, "2026-01-01"),
        current=make_entity("GPT", attributes={"headcount": 20}).snapshot(
            2, "2026-01-02"
        ),
    )
    matched = log.changes_for(entity.entity_id, "headcount")
    assert len(matched) == 1
    assert matched[0].new_value == "20"
    assert log.changes_for(entity.entity_id, "confidence") == ()
    assert log.changes_for("ent_missing", "headcount") == ()


def test_log_reports_the_last_value_of_a_field() -> None:
    entity = make_entity("GPT", attributes={"headcount": 10})
    log = record(HistoryLog(), entity, 1)
    log = log.record(
        entity_id=entity.entity_id,
        entity_type=EntityType.TECHNOLOGY,
        sequence=2,
        recorded_at="2026-01-02",
        source="rep-2",
        previous=entity.snapshot(1, "2026-01-01"),
        current=make_entity("GPT", attributes={"headcount": 20}).snapshot(
            2, "2026-01-02"
        ),
    )
    assert log.last_value_for(entity.entity_id, "headcount") == "20"
    assert log.last_value_for(entity.entity_id, "missing") is None
    assert log.last_value_for("ent_missing", "headcount") is None


def test_log_reports_the_next_sequence() -> None:
    assert HistoryLog().next_sequence() == 1
    entity = make_entity("GPT")
    log = record(HistoryLog(), entity, 4)
    assert log.next_sequence() == 5


def test_log_supports_membership_by_sequence() -> None:
    log = record(HistoryLog(), make_entity("GPT"), 1)
    assert 1 in log
    assert 99 not in log


def test_log_compares_by_content() -> None:
    def build() -> HistoryLog:
        return record(HistoryLog(), make_entity("GPT"), 1)

    assert build() == build()
    assert hash(build()) == hash(build())
    assert build() != HistoryLog()
    assert build() != "not a log"


def test_log_round_trips_through_a_dictionary() -> None:
    entity = make_entity("GPT", attributes={"headcount": 10})
    log = record(HistoryLog(), entity, 1)
    log = record(log, entity, 2, previous=entity.snapshot(1, "2026-01-01"))
    document = log.to_dict()
    rebuilt = HistoryLog.from_dict(document)
    assert rebuilt.to_dict() == document
    assert rebuilt.ordered() == log.ordered()
    assert rebuilt.next_sequence() == log.next_sequence()


def test_log_rejects_a_foreign_payload() -> None:
    with pytest.raises(InvalidMemoryInputError):
        HistoryLog.from_dict({"entries": "nope"})


def test_log_rejects_a_non_mapping() -> None:
    with pytest.raises(InvalidMemoryInputError):
        HistoryLog.from_dict([])  # type: ignore[arg-type]


def test_log_rejects_a_duplicate_sequence() -> None:
    log = record(HistoryLog(), make_entity("GPT"), 1)
    with pytest.raises(ImmutableRecordError):
        log.append(log.latest_for(log.entities()[0]))


def test_log_rejects_a_foreign_entry_on_construction() -> None:
    with pytest.raises(InvalidMemoryInputError):
        HistoryLog(["nope"])  # type: ignore[list-item]


def test_log_append_returns_a_new_log_for_chaining() -> None:
    entity = make_entity("GPT")
    log = HistoryLog()
    appended = log.append(make_entry(entity))
    assert appended is not log
    assert log.is_empty() is True
    assert len(appended) == 1
