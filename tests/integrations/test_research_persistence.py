"""Tests for the durable research-memory backend — Phase 8 Sprint 9, WI-7.

These tests pin the eight properties WI-7 must deliver:

1. the in-memory implementation behaves identically whether or not persistence
   is enabled;
2. persisted state survives a process restart (a fresh backend over the same
   file reconstructs the same memory);
3. serialization round-trips byte-identically;
4. ordering stays deterministic;
5. identifiers are unchanged;
6. provenance is unchanged;
7. canonical content hashes are unchanged; and
8. the existing research regression suite still passes (proved by the suite
   itself).

The backend is exercised through the public ports only: the persistent index,
graph store, and record store are injected into the unchanged
:class:`~predictron_engine.research.memory.ResearchMemory`, exactly as a caller
would.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from predictron_engine.canonical import canonical_json
from predictron_engine.integrations.research_persistence import (
    PERSISTENCE_SCHEMA_VERSION as PERSISTENCE_VERSION,
)
from predictron_engine.integrations.research_persistence import (
    FileDocumentStore,
    PersistentGraphStore,
    PersistentHistoryStore,
    PersistentMemoryIndex,
    PersistentMemoryStore,
    PersistentResearchMemory,
    ResearchPersistenceError,
    open_persistent_memory,
)
from predictron_engine.research.exceptions import (
    ResearchError,
    ResearchMemoryError,
)
from predictron_engine.research.graph_models import (
    EntityType,
    RelationshipType,
)
from predictron_engine.research.graph_store import GraphStore, MemoryIndex
from predictron_engine.research.memory import (
    EntityObservation,
    RelationshipObservation,
    ResearchMemory,
)
from predictron_engine.research.memory_store import MemoryStore
from predictron_engine.research.ports import (
    GraphStorePort,
    HistoryPort,
    MemoryIndexPort,
    MemoryStorePort,
)
from tests.research.memory_helpers import make_evidence, make_report

#: The frozen ``1.0.0`` golden document from the Commit 1 compatibility suite.
GOLDEN_DOCUMENT_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "research_memory_v1.json"
)


def gpt_observation(evidence_id: str) -> EntityObservation:
    """Build a grounded technology observation, as the memory tests do."""
    return EntityObservation(
        "GPT",
        EntityType.TECHNOLOGY,
        attributes={"generation": 4},
        evidence_ids=(evidence_id,),
        confidence=0.8,
    )


def has_technology(evidence_id: str) -> RelationshipObservation:
    """Build a grounded relationship observation, as the memory tests do."""
    return RelationshipObservation(
        relationship_type=RelationshipType.HAS_TECHNOLOGY,
        source_name="Acme",
        source_type=EntityType.COMPANY,
        target_name="GPT",
        target_type=EntityType.TECHNOLOGY,
        evidence_ids=(evidence_id,),
        confidence=0.7,
    )


def populate(memory: ResearchMemory) -> str:
    """Record two deterministic reports and return the evidence identifier.

    The shape mirrors :func:`tests.research.test_ports.populate`: one report
    that only records the company, then one that establishes a technology node
    and an edge to it.
    """
    evidence = make_evidence()
    memory.add_report(make_report("rep-1"), evidence=[evidence])
    memory.add_report(
        make_report("rep-2"),
        entities=[gpt_observation(evidence.evidence_id)],
        relationships=[has_technology(evidence.evidence_id)],
    )
    return evidence.evidence_id


def write_variant(
    path: Path,
    document: dict[str, object],
    *,
    version: str,
    history: object | None = None,
) -> None:
    """Write ``document`` under a chosen version, optionally with history."""
    variant = {
        key: value
        for key, value in document.items()
        if key not in {"schema_version", "history"}
    }
    variant["schema_version"] = version
    if history is not None:
        variant["history"] = history
    path.write_text(canonical_json(variant), encoding="utf-8")


# -- the persistent ports satisfy the protocols ------------------------


def test_the_persistent_ports_satisfy_their_protocols(tmp_path: Path) -> None:
    handle = PersistentResearchMemory(tmp_path / "memory.json")
    assert isinstance(handle.index, MemoryIndexPort)
    assert isinstance(handle.graph, GraphStorePort)
    assert isinstance(handle.records, MemoryStorePort)
    assert isinstance(handle.memory, ResearchMemory)
    assert isinstance(handle.memory.history, HistoryPort)


def test_the_persistent_ports_share_one_index(tmp_path: Path) -> None:
    handle = PersistentResearchMemory(tmp_path / "memory.json")
    assert handle.memory.store.index is handle.memory.index
    assert handle.memory.records.index is handle.memory.index
    assert handle.graph.index is handle.index
    assert handle.records.index is handle.index


def test_open_persistent_memory_returns_the_wired_facade(
    tmp_path: Path,
) -> None:
    memory = open_persistent_memory(tmp_path / "memory.json")
    assert isinstance(memory, ResearchMemory)
    assert isinstance(memory.index, PersistentMemoryIndex)
    assert isinstance(memory.store, PersistentGraphStore)
    assert isinstance(memory.records, PersistentMemoryStore)
    assert isinstance(memory.history, PersistentHistoryStore)


def test_a_fresh_memory_is_empty_and_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    handle = PersistentResearchMemory(path)
    assert handle.memory.is_empty() is True
    assert path.exists() is False


def test_the_in_memory_default_is_unchanged() -> None:
    memory = ResearchMemory()
    assert isinstance(memory.index, MemoryIndex)
    assert isinstance(memory.store, GraphStore)
    assert isinstance(memory.records, MemoryStore)


# -- requirement 1: the in-memory implementation behaves identically ---


def test_a_persistent_memory_matches_the_in_memory_memory(
    tmp_path: Path,
) -> None:
    plain = ResearchMemory()
    populate(plain)

    handle = PersistentResearchMemory(tmp_path / "memory.json")
    populate(handle.memory)

    assert canonical_json(handle.memory.to_dict()) == canonical_json(
        plain.to_dict()
    )
    assert handle.memory.memory_key() == plain.memory_key()
    assert handle.memory.counts() == plain.counts()


def test_a_persistent_graph_matches_a_plain_graph(tmp_path: Path) -> None:
    plain = ResearchMemory()
    populate(plain)

    handle = PersistentResearchMemory(tmp_path / "memory.json")
    populate(handle.memory)

    assert canonical_json(handle.graph.to_dict()) == canonical_json(
        plain.graph.to_dict()
    )
    assert canonical_json(handle.records.to_dict()) == canonical_json(
        plain.records.to_dict()
    )
    assert handle.graph.stats() == plain.graph.stats()


# -- requirement 2: persistence survives a restart ---------------------


def test_memory_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    first = PersistentResearchMemory(path)
    populate(first.memory)

    second = PersistentResearchMemory(path)

    assert canonical_json(second.graph.to_dict()) == canonical_json(
        first.graph.to_dict()
    )
    assert canonical_json(second.records.to_dict()) == canonical_json(
        first.records.to_dict()
    )
    assert second.memory.memory_key() == first.memory.memory_key()
    assert second.records.counts() == first.records.counts()
    assert second.graph.stats() == first.graph.stats()


def test_writes_continue_after_a_restart_without_collisions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.json"
    first = PersistentResearchMemory(path)
    populate(first.memory)
    high_water = first.index.next_sequence()

    second = PersistentResearchMemory(path)
    assert second.index.next_sequence() == high_water

    second.memory.add_report(
        make_report("rep-3"),
        evidence=[make_evidence(claim="Acme ships Y", url="https://x.test/b")],
    )
    assert second.memory.get_report("rep-3") is not None

    third = PersistentResearchMemory(path)
    assert third.memory.get_report("rep-3") is not None
    assert third.index.next_sequence() == second.index.next_sequence()
    assert third.records.counts() == second.records.counts()
    assert third.graph.stats() == second.graph.stats()


def test_history_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    first = PersistentResearchMemory(path)
    populate(first.memory)
    before_document = first.memory.history.to_dict()
    before_entries = first.memory.history.ordered()

    second = PersistentResearchMemory(path)

    assert second.memory.history.to_dict() == before_document
    assert second.memory.history.ordered() == before_entries
    assert len(second.memory.history) == len(before_entries)


def test_history_counts_survive_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    first = PersistentResearchMemory(path)
    populate(first.memory)
    before = first.memory.counts()["history"]

    second = PersistentResearchMemory(path)

    assert before > 0
    assert second.memory.counts()["history"] == before


def test_a_post_restart_record_continues_version_numbering(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.json"
    first = PersistentResearchMemory(path)
    populate(first.memory)
    technology = first.memory.get_entity("GPT")
    assert technology is not None

    second = PersistentResearchMemory(path)
    evidence = make_evidence(
        claim="Acme ships GPT-5",
        url="https://example.test/c",
    )
    second.memory.add_report(
        make_report("rep-3"),
        evidence=[evidence],
        entities=[
            EntityObservation(
                "GPT",
                EntityType.TECHNOLOGY,
                attributes={"generation": 5},
                evidence_ids=(evidence.evidence_id,),
                confidence=0.8,
            )
        ],
    )

    versions = [
        entry.version
        for entry in second.memory.history.for_entity(
            technology.entity_id
        )
    ]
    assert versions == [1, 2]


# -- requirement 3: round-trip serialization is byte-identical ---------


def test_the_document_round_trips_byte_identically(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    handle = PersistentResearchMemory(path)
    populate(handle.memory)

    original_bytes = path.read_bytes()
    original_document = handle.document()

    reopened = PersistentResearchMemory(path)
    reopened.flush()

    assert path.read_bytes() == original_bytes
    assert canonical_json(reopened.document()) == canonical_json(
        original_document
    )
    assert canonical_json(reopened.document()["history"]) == canonical_json(
        original_document["history"]
    )


def test_writing_the_same_state_twice_is_byte_stable(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    handle = PersistentResearchMemory(path)
    populate(handle.memory)

    first = path.read_bytes()
    handle.flush()

    assert path.read_bytes() == first


def test_the_document_carries_only_the_existing_schema_sections(
    tmp_path: Path,
) -> None:
    handle = PersistentResearchMemory(tmp_path / "memory.json")
    populate(handle.memory)
    document = handle.document()
    assert document["schema_version"] == PERSISTENCE_VERSION
    assert set(document) == {
        "schema_version",
        "index",
        "graph",
        "records",
        "history",
    }
    assert document["graph"]["schema_version"] == "1.0.0"
    assert document["records"]["schema_version"] == "1.0.0"
    assert document["index"]["schema_version"] == "1.0.0"
    assert document["history"]["schema_version"] == "1.0.0"


# -- requirement 4: ordering is deterministic --------------------------


def test_ordering_is_deterministic(tmp_path: Path) -> None:
    first = PersistentResearchMemory(tmp_path / "a.json")
    second = PersistentResearchMemory(tmp_path / "b.json")
    populate(first.memory)
    populate(second.memory)

    assert first.index.keys() == second.index.keys()
    assert first.graph.entity_ids() == second.graph.entity_ids()
    assert first.graph.relationships() == second.graph.relationships()
    assert [report.report_id for report in first.records.reports()] == [
        "rep-1",
        "rep-2",
    ]
    assert [item.evidence_id for item in first.records.evidence()] == [
        item.evidence_id for item in second.records.evidence()
    ]


def test_reads_are_sorted_after_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    handle = PersistentResearchMemory(path)
    populate(handle.memory)

    reloaded = PersistentResearchMemory(path)
    entity_ids = reloaded.graph.entity_ids()
    assert list(entity_ids) == sorted(entity_ids)
    keys = reloaded.index.keys()
    assert [key.sequence for key in keys] == sorted(
        key.sequence for key in keys
    )


# -- requirement 5: identifiers are unchanged --------------------------


def test_entity_identifiers_are_unchanged_across_a_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.json"
    first = PersistentResearchMemory(path)
    populate(first.memory)
    before = first.memory.entity_ids()

    second = PersistentResearchMemory(path)

    assert second.memory.entity_ids() == before


def test_record_keys_are_unchanged_across_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    first = PersistentResearchMemory(path)
    populate(first.memory)
    before = first.index.keys()

    second = PersistentResearchMemory(path)

    assert second.index.keys() == before


# -- requirement 6: provenance is unchanged ----------------------------


def test_provenance_is_unchanged_across_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    first = PersistentResearchMemory(path)
    populate(first.memory)
    before = {
        entity.entity_id: entity.provenance
        for entity in first.memory.entities()
    }

    second = PersistentResearchMemory(path)
    after = {
        entity.entity_id: entity.provenance
        for entity in second.memory.entities()
    }

    assert after == before


def test_evidence_and_reports_are_unchanged_across_a_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.json"
    first = PersistentResearchMemory(path)
    populate(first.memory)

    second = PersistentResearchMemory(path)

    assert second.memory.reports() == first.memory.reports()
    assert second.memory.records.evidence() == (
        first.memory.records.evidence()
    )


# -- requirement 7: canonical content hashes are unchanged -------------


def test_canonical_hashes_are_unchanged_across_a_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.json"
    first = PersistentResearchMemory(path)
    populate(first.memory)
    graph_key = first.memory.graph.graph_key()
    memory_key = first.memory.memory_key()

    second = PersistentResearchMemory(path)

    assert second.memory.graph.graph_key() == graph_key
    assert second.memory.memory_key() == memory_key


# -- the persistence schema-version gate -------------------------------


def test_a_1_0_0_document_loads_with_empty_history(tmp_path: Path) -> None:
    source = PersistentResearchMemory(tmp_path / "source.json")
    populate(source.memory)
    path = tmp_path / "memory.json"
    write_variant(path, source.document(), version="1.0.0")
    before = path.read_bytes()

    reopened = PersistentResearchMemory(path)

    assert reopened.memory.history.is_empty() is True
    assert reopened.memory.counts()["history"] == 0
    assert [report.report_id for report in reopened.memory.reports()] == [
        "rep-1",
        "rep-2",
    ]
    assert path.read_bytes() == before


def test_a_1_1_0_document_loads_with_its_history(tmp_path: Path) -> None:
    source = PersistentResearchMemory(tmp_path / "source.json")
    populate(source.memory)
    path = tmp_path / "memory.json"
    write_variant(
        path,
        source.document(),
        version="1.1.0",
        history=source.document()["history"],
    )

    reopened = PersistentResearchMemory(path)

    assert reopened.memory.history.to_dict() == (
        source.memory.history.to_dict()
    )
    assert reopened.memory.counts()["history"] == (
        source.memory.counts()["history"]
    )


def test_a_1_1_0_document_without_history_loads_empty(
    tmp_path: Path,
) -> None:
    source = PersistentResearchMemory(tmp_path / "source.json")
    populate(source.memory)
    path = tmp_path / "memory.json"
    write_variant(path, source.document(), version="1.1.0")

    reopened = PersistentResearchMemory(path)

    assert reopened.memory.history.is_empty() is True
    assert reopened.memory.counts()["history"] == 0


def test_a_stray_history_section_on_a_1_0_0_document_is_ignored(
    tmp_path: Path,
) -> None:
    source = PersistentResearchMemory(tmp_path / "source.json")
    populate(source.memory)
    path = tmp_path / "memory.json"
    write_variant(
        path,
        source.document(),
        version="1.0.0",
        history=source.document()["history"],
    )
    before = path.read_bytes()

    reopened = PersistentResearchMemory(path)

    assert reopened.memory.history.is_empty() is True
    assert reopened.memory.counts()["history"] == 0
    assert path.read_bytes() == before


def test_an_unsupported_schema_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    path.write_text('{"schema_version": "2.0.0"}', encoding="utf-8")

    with pytest.raises(ResearchPersistenceError) as error:
        PersistentResearchMemory(path)

    message = str(error.value)
    assert "2.0.0" in message
    assert "1.0.0" in message
    assert "1.1.0" in message


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        '{"schema_version": 110}',
        '{"schema_version": null}',
    ],
)
def test_a_missing_or_non_string_schema_version_is_rejected(
    tmp_path: Path,
    raw: str,
) -> None:
    path = tmp_path / "memory.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ResearchPersistenceError) as error:
        PersistentResearchMemory(path)

    message = str(error.value)
    assert "schema_version" in message
    assert "1.0.0" in message
    assert "1.1.0" in message


def test_the_commit_1_fixture_loads_with_empty_history() -> None:
    before = GOLDEN_DOCUMENT_PATH.read_bytes()

    reopened = PersistentResearchMemory(GOLDEN_DOCUMENT_PATH)

    assert reopened.memory.history.is_empty() is True
    assert reopened.memory.counts()["history"] == 0
    assert [report.report_id for report in reopened.memory.reports()] == [
        "rep-1",
        "rep-2",
    ]
    assert GOLDEN_DOCUMENT_PATH.read_bytes() == before


# -- filesystem durability details -------------------------------------


def test_no_temporary_files_are_left_behind(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    handle = PersistentResearchMemory(path)
    populate(handle.memory)

    leftovers = [
        entry.name
        for entry in tmp_path.iterdir()
        if entry.name.endswith(".tmp")
    ]
    assert leftovers == []


def test_the_document_store_reports_absence_and_reads_back(
    tmp_path: Path,
) -> None:
    store = FileDocumentStore(tmp_path / "memory.json")
    assert store.exists() is False
    assert store.read() is None

    store.write({"schema_version": PERSISTENCE_VERSION, "value": 1})
    assert store.exists() is True
    assert store.read() == {"schema_version": PERSISTENCE_VERSION, "value": 1}


def test_a_malformed_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ResearchPersistenceError):
        PersistentResearchMemory(path)


def test_a_non_object_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ResearchPersistenceError):
        PersistentResearchMemory(path)


def test_the_persistence_error_is_a_research_memory_error() -> None:
    assert issubclass(ResearchPersistenceError, ResearchMemoryError)
    assert issubclass(ResearchPersistenceError, ResearchError)
