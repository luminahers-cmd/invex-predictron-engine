"""Tests for the report and evidence ledger — Phase 8 Sprint 6.

Covers :class:`MemoryStore`: appending reports and evidence, idempotent
replay, refusal to rewrite a content-addressed record, duplicate
acknowledgement, and company/topic retrieval.
"""

from __future__ import annotations

import pytest

from predictron_engine.research.deduplication import DeduplicationIndex
from predictron_engine.research.exceptions import (
    ImmutableRecordError,
    InvalidMemoryInputError,
)
from predictron_engine.research.execution_report import ResearchReport
from predictron_engine.research.graph_store import MemoryIndex
from predictron_engine.research.memory_store import (
    MEMORY_SCHEMA_VERSION,
    RECORD_EVIDENCE,
    RECORD_REPORT,
    MemoryStore,
    TopicRetrieval,
)
from predictron_engine.research.models import Evidence

from .memory_helpers import make_evidence, make_report

ACME = "rep-1"
SECOND = "rep-2"


def test_schema_version_is_exported() -> None:
    assert MEMORY_SCHEMA_VERSION == "1.0.0"
    assert (RECORD_REPORT, RECORD_EVIDENCE) == ("report", "evidence")


def test_store_starts_empty() -> None:
    store = MemoryStore()
    assert store.is_empty() is True
    assert store.counts() == {
        "reports": 0,
        "evidence": 0,
        "duplicates": 0,
        "indexed": 0,
    }
    assert store.duplicate_count() == 0
    assert store.reports() == ()
    assert store.evidence() == ()
    assert store.get_report("rep-1") is None
    assert store.get_evidence("ev-1") is None
    assert store.has_report("rep-1") is False
    assert store.has_evidence("ev-1") is False
    assert store.latest_report() is None
    assert store.report_for_evidence("ev-1") is None


def test_adding_a_report_allocates_a_sequence() -> None:
    store = MemoryStore()
    write, writes = store.add_report(make_report(), stored_at="2026-01-01")
    assert write.created is True
    assert write.replayed is False
    assert write.previous is None
    assert write.report_id == ACME
    assert write.company_name == "Acme"
    assert write.sequence == 1
    assert writes == ()
    assert store.is_empty() is False
    assert store.counts() == {
        "reports": 1,
        "evidence": 0,
        "duplicates": 0,
        "indexed": 1,
    }
    assert store.has_report(ACME)
    assert store.get_report(ACME) is not None
    assert store.index.latest(RECORD_REPORT) is not None
    assert store.index.latest(RECORD_REPORT).company_name == "Acme"  # type: ignore[union-attr]


def test_adding_evidence_allocates_and_indexes_it() -> None:
    store = MemoryStore()
    write = store.add_evidence(make_evidence(), stored_at="2026-01-01")
    assert write.created is True
    assert write.replayed is False
    assert write.is_duplicate is False
    assert write.duplicate_of is None
    assert write.topic_id == "topic-1"
    assert write.sequence == 1
    assert store.has_evidence(write.evidence_id)
    assert store.get_evidence(write.evidence_id) is not None
    assert write.to_dict()["replayed"] is False


def test_adding_the_same_evidence_twice_is_a_replay() -> None:
    store = MemoryStore()
    first = store.add_evidence(make_evidence())
    second = store.add_evidence(make_evidence())
    assert first.sequence == second.sequence == 1
    assert second.created is False
    assert second.replayed is True
    assert second.is_duplicate is False
    assert store.counts()["evidence"] == 1


def test_distinct_claims_are_distinct_evidence_items() -> None:
    store = MemoryStore()
    store.add_evidence(make_evidence())
    write = store.add_evidence(make_evidence(claim="Acme sells widgets"))
    assert write.evidence_id != make_evidence().evidence_id
    assert write.is_duplicate is True
    assert store.counts()["evidence"] == 2


def test_adding_a_report_with_its_evidence_links_them() -> None:
    store = MemoryStore()
    evidence = make_evidence()
    write, writes = store.add_report(make_report(), evidence=[evidence])
    assert len(writes) == 1
    assert writes[0].created is True
    assert store.evidence_for_report(ACME) == (evidence,)
    assert store.report_for_evidence(evidence.evidence_id) == ACME
    assert store.counts() == {
        "reports": 1,
        "evidence": 1,
        "duplicates": 0,
        "indexed": 2,
    }


def test_a_replayed_report_is_a_no_op() -> None:
    store = MemoryStore()
    store.add_report(make_report(), evidence=[make_evidence()])
    write, writes = store.add_report(make_report(), evidence=[make_evidence()])
    assert write.created is False
    assert write.replayed is True
    assert write.previous is not None
    assert write.report == write.previous
    assert len(writes) == 1
    assert writes[0].replayed is True
    assert writes[0].created is False
    assert writes[0].sequence == 2
    assert store.counts() == {
        "reports": 1,
        "evidence": 1,
        "duplicates": 0,
        "indexed": 2,
    }


def test_a_report_cannot_be_rewritten_under_one_identifier() -> None:
    store = MemoryStore()
    store.add_report(make_report())
    with pytest.raises(ImmutableRecordError):
        store.add_report(make_report(company_name="OpenAI"))


def test_replay_refuses_evidence_the_report_never_carried() -> None:
    store = MemoryStore()
    store.add_report(make_report(), evidence=[make_evidence()])
    with pytest.raises(ImmutableRecordError):
        store.add_report(
            make_report(), evidence=[make_evidence(), make_evidence(claim="New claim")]
        )
    assert store.counts()["evidence"] == 1


def test_duplicate_evidence_is_recorded_once_and_acknowledged() -> None:
    store = MemoryStore()
    first = make_evidence()
    second = make_evidence(claim="A different claim about Acme")
    _, writes = store.add_report(make_report(), evidence=[first, second])
    assert [write.is_duplicate for write in writes] == [False, True]
    assert writes[1].duplicate_of == first.evidence_id
    assert writes[1].replayed is False
    assert writes[1].created is False
    assert store.counts() == {
        "reports": 1,
        "evidence": 2,
        "duplicates": 2,
        "indexed": 3,
    }
    assert store.evidence_for_report(ACME) == (first, second)


def test_a_standalone_duplicate_is_also_acknowledged() -> None:
    store = MemoryStore()
    first = store.add_evidence(make_evidence())
    write = store.add_evidence(make_evidence(claim="Another Acme claim"))
    assert write.is_duplicate is True
    assert write.duplicate_of == first.evidence_id
    assert write.created is False
    assert store.counts()["evidence"] == 2


def test_store_rejects_a_foreign_report_or_evidence() -> None:
    store = MemoryStore()
    with pytest.raises(InvalidMemoryInputError):
        store.add_report("rep-1")  # type: ignore[arg-type]
    with pytest.raises(InvalidMemoryInputError):
        store.add_evidence("ev-1")  # type: ignore[arg-type]
    with pytest.raises(InvalidMemoryInputError):
        store.add_report(make_report(), evidence=["ev-1"])  # type: ignore[list-item]


# -- retrieval -------------------------------------------------------


def test_reports_can_be_read_back_in_order() -> None:
    store = MemoryStore()
    store.add_report(make_report(ACME))
    store.add_report(make_report(SECOND))
    assert [report.report_id for report in store.reports()] == [ACME, SECOND]


def test_company_lookup_normalizes_names() -> None:
    store = MemoryStore()
    store.add_report(make_report(ACME, company_name="Acme"))
    store.add_report(make_report(SECOND, company_name="OpenAI Inc."))
    assert store.reports_for_company("Acme") == (store.get_report(ACME),)
    assert store.reports_for_company("acme, inc.") == (store.get_report(ACME),)
    assert store.reports_for_company("  ACME  ") == (store.get_report(ACME),)
    assert store.reports_for_company("Nowhere") == ()


def test_latest_report_is_the_newest_match() -> None:
    store = MemoryStore()
    store.add_report(make_report(ACME, company_name="Acme"))
    store.add_report(make_report(SECOND, company_name="Acme"))
    store.add_report(make_report("rep-3", company_name="OpenAI"))
    assert store.latest_report("Acme") == store.get_report(SECOND)
    assert store.latest_report("open ai") == store.get_report("rep-3")
    assert store.latest_report("Nowhere") is None
    assert store.latest_report() == store.get_report("rep-3")


def test_retrieve_latest_returns_newest_first_up_to_a_limit() -> None:
    store = MemoryStore()
    store.add_report(make_report(ACME, company_name="Acme"))
    store.add_report(make_report(SECOND, company_name="Acme"))
    assert [r.report_id for r in store.retrieve_latest(company_name="Acme")] == [SECOND]
    assert [r.report_id for r in store.retrieve_latest(company_name="Acme", limit=5)] == [
        SECOND,
        ACME,
    ]
    assert len(store.retrieve_latest(limit=1)) == 1
    assert len(store.retrieve_latest(limit=10)) == 2


def test_retrieve_latest_rejects_a_malformed_limit() -> None:
    store = MemoryStore()
    with pytest.raises(InvalidMemoryInputError):
        store.retrieve_latest(limit=0)
    with pytest.raises(InvalidMemoryInputError):
        store.retrieve_latest(limit="2")  # type: ignore[arg-type]


def test_evidence_can_be_retrieved_by_report_and_topic() -> None:
    store = MemoryStore()
    first = make_evidence()
    second = make_evidence(claim="Acme hires", topic_id="topic-2")
    store.add_report(make_report(ACME), evidence=[first, second])
    assert store.evidence_for_report(ACME) == (first, second)
    assert store.evidence_for_report("rep-9") == ()
    assert store.evidence_for_topic("topic-1") == (first,)
    assert store.evidence_for_topic(" topic-2 ") == (second,)
    assert store.evidence_for_topic("topic-9") == ()


def test_reports_can_be_retrieved_by_topic() -> None:
    store = MemoryStore()
    store.add_report(make_report(ACME), topic_id="topic-1")
    store.add_report(make_report(SECOND), topic_id="topic-2")
    assert store.reports_for_topic("topic-1") == (store.get_report(ACME),)
    assert store.reports_for_topic(" topic-2 ") == (store.get_report(SECOND),)
    assert store.reports_for_topic("topic-9") == ()


def test_a_report_is_found_by_the_topic_of_its_evidence() -> None:
    store = MemoryStore()
    store.add_report(make_report(ACME), evidence=[make_evidence(topic_id="topic-7")])
    assert store.reports_for_topic("topic-7") == ()
    assert store.covers_topic(store.get_report(ACME), "topic-7") is True  # type: ignore[arg-type]
    assert store.retrieve_latest(topic_id="topic-7") == (store.get_report(ACME),)


def test_a_write_can_declare_the_topic_its_report_covers() -> None:
    store = MemoryStore()
    store.add_report(make_report(ACME), topic_id="topic-3")
    assert store.reports_for_topic("topic-3") == (store.get_report(ACME),)
    assert store.covers_topic(store.get_report(ACME), "topic-3") is True  # type: ignore[arg-type]
    assert store.covers_topic(store.get_report(ACME), "topic-9") is False  # type: ignore[arg-type]
    assert store.covers_topic(store.get_report(ACME), "  ") is False  # type: ignore[arg-type]


def test_topic_records_bundles_the_ledger_half_of_a_topic() -> None:
    store = MemoryStore()
    evidence = make_evidence()
    store.add_report(make_report(ACME), evidence=[evidence], topic_id="topic-1")
    retrieval = store.topic_records(" topic-1 ")
    assert retrieval.topic_id == "topic-1"
    assert retrieval.reports == (store.get_report(ACME),)
    assert retrieval.evidence == (evidence,)
    assert retrieval.is_empty() is False
    assert store.topic_records("topic-9").is_empty() is True
    assert store.topic_records("topic-1").to_dict()["topic_id"] == "topic-1"


def test_topic_records_requires_a_topic() -> None:
    with pytest.raises(InvalidMemoryInputError):
        MemoryStore().topic_records("  ")


def test_key_lookup_is_exposed_for_callers_that_track_writes() -> None:
    store = MemoryStore()
    write, _ = store.add_report(make_report(), evidence=[make_evidence()])
    key = store.key_for(RECORD_EVIDENCE, write.report_id)
    assert key is None or key.record_type == RECORD_EVIDENCE


# -- serialization ---------------------------------------------------


def test_store_round_trips_through_a_dictionary() -> None:
    store = MemoryStore()
    evidence = make_evidence()
    store.add_report(make_report(ACME), evidence=[evidence, make_evidence(claim="x")])
    store.add_report(make_report(SECOND))
    document = store.to_dict()
    rebuilt = MemoryStore.from_dict(document)
    assert rebuilt.to_dict() == document
    assert rebuilt.counts() == store.counts()
    assert rebuilt.reports() == store.reports()
    assert rebuilt.evidence() == store.evidence()
    assert rebuilt.reports_for_company("Acme") == store.reports_for_company("Acme")
    assert rebuilt.index.next_sequence() == store.index.next_sequence()
    assert rebuilt.duplicate_count() == store.duplicate_count()


def test_store_round_trips_onto_caller_supplied_indexes() -> None:
    store = MemoryStore()
    store.add_report(make_report(), evidence=[make_evidence()])
    document = store.to_dict()
    index = MemoryIndex.from_dict(document["index"])
    deduplication = DeduplicationIndex()
    rebuilt = MemoryStore.from_dict(
        document, index=index, deduplication=deduplication
    )
    assert rebuilt.index is index
    assert rebuilt.deduplication is deduplication
    assert rebuilt.evidence() == store.evidence()
    assert rebuilt.add_evidence(make_evidence()).replayed is True


def test_store_rejects_a_foreign_document() -> None:
    with pytest.raises(InvalidMemoryInputError):
        MemoryStore.from_dict([])  # type: ignore[arg-type]
    with pytest.raises(InvalidMemoryInputError):
        MemoryStore.from_dict({"reports": "nope"})


def test_evidence_and_report_values_are_still_the_original_objects() -> None:
    store = MemoryStore()
    report = make_report()
    evidence = make_evidence()
    store.add_report(report, evidence=[evidence])
    assert store.get_report(ACME) is report
    assert store.get_evidence(evidence.evidence_id) is evidence
    assert isinstance(store.get_report(ACME), ResearchReport)
    assert isinstance(store.get_evidence(evidence.evidence_id), Evidence)


def test_topic_retrieval_serializes_its_ledger_half() -> None:
    retrieval = TopicRetrieval(topic_id="topic-1")
    assert retrieval.to_dict() == {
        "topic_id": "topic-1",
        "reports": [],
        "evidence": [],
        "entity_ids": [],
        "relationship_keys": [],
    }
