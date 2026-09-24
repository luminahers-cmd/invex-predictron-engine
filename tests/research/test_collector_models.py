"""Tests for the Evidence Collection immutable models.

Covers validation, immutability, determinism, and to_dict/from_dict
serialization for the Sprint 3 value objects (EvidenceReference,
EvidenceMetadata, Evidence, EvidenceCollection, CollectionStatus,
CollectionResult).
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from predictron_engine.research import (
    COLLECTION_SCHEMA_VERSION,
    CollectionResult,
    CollectionStatus,
    Evidence,
    EvidenceCollection,
    EvidenceMetadata,
    EvidenceReference,
)
from predictron_engine.research.exceptions import InvalidEvidence


def _reference(**overrides: object) -> EvidenceReference:
    base: dict[str, object] = {
        "source_category": "crunchbase",
        "source_identifier": "crunchbase",
        "url": "https://placeholder.predictron.local/funding/crunchbase",
        "description": "Placeholder reference for Funding research.",
    }
    base.update(overrides)
    return EvidenceReference(**base)  # type: ignore[arg-type]


def _evidence(**overrides: object) -> Evidence:
    base: dict[str, object] = {
        "task_id": "research_funding",
        "topic_id": "funding",
        "collector_id": "funding_collector",
        "category": "funding_rounds",
        "claim": "Recorded the funding rounds of Acme.",
        "confidence": 0.9,
        "reference": _reference(),
    }
    base.update(overrides)
    return Evidence(**base)  # type: ignore[arg-type]


def _collection(items: tuple[Evidence, ...] = ()) -> EvidenceCollection:
    return EvidenceCollection(
        evidence=items,
        topic_order=tuple(dict.fromkeys(item.topic_id for item in items)),
    )


def _result(**overrides: object) -> CollectionResult:
    base: dict[str, object] = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "plan_id": "plan_abc123",
        "company_name": "Acme",
        "collection_id": "collection_abc123",
        "status": CollectionStatus.SUCCESS,
        "collection": EvidenceCollection(
            evidence=(_evidence(),), topic_order=("funding",)
        ),
        "executed_tasks": ("research_funding",),
        "failed_tasks": (),
        "unresolved_tasks": (),
    }
    base.update(overrides)
    return CollectionResult(**base)  # type: ignore[arg-type]


class TestCollectionStatus:
    """The status enum carries the four deterministic outcomes."""

    def test_members(self) -> None:
        assert {status.value for status in CollectionStatus} == {
            "success",
            "partial",
            "failed",
            "empty",
        }

    def test_str_enum_lookup_by_value(self) -> None:
        assert CollectionStatus("partial") is CollectionStatus.PARTIAL
        with pytest.raises(ValueError):
            CollectionStatus("bogus")


class TestEvidenceReference:
    """Source provenance metadata validates and serializes."""

    def test_minimal_reference(self) -> None:
        reference = _reference()
        assert reference.source_category == "crunchbase"
        assert reference.source_identifier == "crunchbase"
        assert reference.url
        assert reference.description

    def test_requires_category_and_identifier(self) -> None:
        for field in ("source_category", "source_identifier"):
            for value in ("", "   "):
                with pytest.raises(InvalidEvidence):
                    _reference(**{field: value})

    def test_fields_are_stripped(self) -> None:
        reference = _reference(
            source_category="  crunchbase  ",
            source_identifier="  crunchbase  ",
            url="  https://example.test  ",
        )
        assert reference.source_category == "crunchbase"
        assert reference.source_identifier == "crunchbase"
        assert reference.url == "https://example.test"

    def test_serialization_roundtrip(self) -> None:
        reference = _reference(description="Placeholder.")
        assert EvidenceReference.from_dict(reference.to_dict()) == reference
        raw = json.dumps(reference.to_dict())
        assert EvidenceReference.from_dict(json.loads(raw)) == reference

    def test_invalid_dict_raises(self) -> None:
        data = _reference().to_dict()
        data.pop("source_identifier")
        with pytest.raises(ValueError):
            EvidenceReference.from_dict(data)

    def test_immutability(self) -> None:
        reference = _reference()
        with pytest.raises(FrozenInstanceError):
            reference.url = "https://other.test"  # type: ignore[misc]


class TestEvidenceMetadata:
    """Collector metadata validates identity and topic coverage."""

    def test_minimal_metadata(self) -> None:
        metadata = EvidenceMetadata(
            collector_id="funding_collector",
            display_name="Funding",
            supported_topics=("funding",),
        )
        assert metadata.deterministic is True
        assert metadata.description == ""

    def test_requires_identity_fields(self) -> None:
        for field in ("collector_id", "display_name"):
            base: dict[str, object] = {
                "collector_id": "funding_collector",
                "display_name": "Funding",
                "supported_topics": ("funding",),
            }
            base[field] = ""
            with pytest.raises(InvalidEvidence):
                EvidenceMetadata(**base)  # type: ignore[arg-type]

    def test_supported_topics_required_and_deduped(self) -> None:
        with pytest.raises(InvalidEvidence):
            EvidenceMetadata(
                collector_id="funding_collector",
                display_name="Funding",
                supported_topics=(),
            )
        metadata = EvidenceMetadata(
            collector_id="funding_collector",
            display_name="Funding",
            supported_topics=("funding", "team", "funding"),
        )
        assert metadata.supported_topics == ("funding", "team")

    def test_serialization_roundtrip(self) -> None:
        metadata = EvidenceMetadata(
            collector_id="funding_collector",
            display_name="Funding",
            description="Capital raised.",
            supported_topics=("funding",),
            deterministic=True,
        )
        restored = EvidenceMetadata.from_dict(metadata.to_dict())
        assert restored == metadata
        assert EvidenceMetadata.from_dict(
            json.loads(json.dumps(metadata.to_dict()))
        ) == metadata

    def test_from_dict_requires_fields(self) -> None:
        data = {
            "collector_id": "funding_collector",
            "display_name": "Funding",
            "supported_topics": [],
            "deterministic": True,
        }
        with pytest.raises(InvalidEvidence):
            EvidenceMetadata.from_dict(data)

    def test_immutability(self) -> None:
        metadata = EvidenceMetadata(
            collector_id="funding_collector",
            display_name="Funding",
            supported_topics=("funding",),
        )
        with pytest.raises(FrozenInstanceError):
            metadata.deterministic = False  # type: ignore[misc]


class TestEvidence:
    """Evidence items are immutable and self-identifying."""

    def test_evidence_id_is_stable_content_hash(self) -> None:
        first = _evidence()
        second = _evidence()
        assert first.evidence_id == second.evidence_id
        assert first.evidence_id.startswith("ev_")
        changed = _evidence(claim="A different claim.")
        assert changed.evidence_id != first.evidence_id

    def test_requires_identity_fields(self) -> None:
        for field in ("task_id", "topic_id", "collector_id", "category", "claim"):
            with pytest.raises(InvalidEvidence):
                _evidence(**{field: ""})

    def test_confidence_within_unit_range(self) -> None:
        for value in (-0.1, 1.1):
            with pytest.raises(InvalidEvidence):
                _evidence(confidence=value)

    def test_reference_must_be_typed(self) -> None:
        with pytest.raises(InvalidEvidence):
            _evidence(reference="crunchbase")  # type: ignore[arg-type]

    def test_serialization_roundtrip(self) -> None:
        item = _evidence()
        restored = Evidence.from_dict(item.to_dict())
        assert restored == item
        assert restored.evidence_id == item.evidence_id
        assert Evidence.from_dict(json.loads(json.dumps(item.to_dict()))) == item

    def test_invalid_dict_raises(self) -> None:
        data = _evidence().to_dict()
        data["confidence"] = "high"
        with pytest.raises(ValueError):
            Evidence.from_dict(data)

    def test_fields_stripped(self) -> None:
        item = _evidence(claim="  Recorded funding rounds.  ")
        assert item.claim == "Recorded funding rounds."

    def test_immutability(self) -> None:
        item = _evidence()
        with pytest.raises(FrozenInstanceError):
            item.confidence = 1.0  # type: ignore[misc]


class TestEvidenceCollection:
    """The collection aggregates and orders evidence deterministically."""

    def test_empty_collection(self) -> None:
        collection = _collection()
        assert len(collection) == 0
        assert collection.evidence == ()
        assert collection.topics_covered() == ()
        assert not collection.has_topic("funding")
        assert collection.for_topic("funding") == ()

    def test_orders_by_topic_order_then_evidence_id(self) -> None:
        items = (_evidence(claim="B claim."), _evidence(claim="A claim."))
        reverse_items = (items[1], items[0])
        ordered = _collection(items)
        also_ordered = EvidenceCollection(
            evidence=reverse_items, topic_order=("funding",)
        )
        assert ordered == also_ordered
        assert ordered.evidence == tuple(
            sorted(items, key=lambda item: item.evidence_id)
        )
        assert [item.evidence_id for item in ordered.evidence] == sorted(
            item.evidence_id for item in items
        )

    def test_for_topic_and_has_topic(self) -> None:
        founders = _evidence(
            task_id="research_founders",
            topic_id="founders",
            collector_id="founder_collector",
        )
        collection = _collection((_evidence(), founders))
        assert collection.has_topic("funding")
        assert collection.has_topic("founders")
        assert [i.topic_id for i in collection.for_topic("founders")] == [
            "founders"
        ]
        assert collection.evidence_count("funding") == 1
        assert collection.evidence_count("bogus") == 0

    def test_topics_covered_follows_topic_order(self) -> None:
        founders = _evidence(
            task_id="research_founders",
            topic_id="founders",
            collector_id="founder_collector",
        )
        collection = EvidenceCollection(
            evidence=(_evidence(), founders),
            topic_order=("founders", "funding"),
        )
        assert collection.topics_covered() == ("founders", "funding")
        assert collection.items() == collection.evidence

    def test_non_evidence_entry_rejected(self) -> None:
        with pytest.raises(InvalidEvidence):
            EvidenceCollection(evidence=("not-evidence",))  # type: ignore[arg-type]

    def test_topic_order_deduped(self) -> None:
        collection = EvidenceCollection(
            evidence=(_evidence(),), topic_order=("funding", "funding")
        )
        assert collection.topic_order == ("funding",)

    def test_serialization_roundtrip(self) -> None:
        collection = _collection((_evidence(),))
        restored = EvidenceCollection.from_dict(collection.to_dict())
        assert restored == collection
        assert EvidenceCollection.from_dict(
            json.loads(json.dumps(collection.to_dict()))
        ) == collection
        assert collection.to_dict()["topics_covered"] == ["funding"]

    def test_immutability(self) -> None:
        collection = _collection((_evidence(),))
        with pytest.raises(FrozenInstanceError):
            collection.evidence = ()  # type: ignore[misc]


class TestCollectionResult:
    """The engine result model is serializable and self-describing."""

    def test_fields_and_helpers(self) -> None:
        result = _result()
        assert result.schema_version == COLLECTION_SCHEMA_VERSION
        assert result.status is CollectionStatus.SUCCESS
        assert result.topics_covered == ("funding",)
        assert result.evidence_count == 1
        assert not result.has_failures()
        assert result.successfully_executed()

    def test_failure_helpers(self) -> None:
        result = _result(
            status=CollectionStatus.PARTIAL,
            executed_tasks=(),
            failed_tasks=("research_team",),
            unresolved_tasks=("research_team",),
        )
        assert result.has_failures()
        assert not result.successfully_executed()

    def test_requires_fields(self) -> None:
        for field in (
            "schema_version",
            "plan_id",
            "company_name",
            "collection_id",
        ):
            data = _result().to_dict()
            data[field] = ""
            with pytest.raises(InvalidEvidence):
                CollectionResult.from_dict(data)

    def test_status_validated(self) -> None:
        with pytest.raises(InvalidEvidence):
            _result(status="success")  # type: ignore[arg-type]
        data = _result().to_dict()
        data["status"] = "bogus"
        with pytest.raises(ValueError):
            CollectionResult.from_dict(data)

    def test_collection_must_be_evidence_collection(self) -> None:
        with pytest.raises(InvalidEvidence):
            _result(collection=())  # type: ignore[arg-type]

    def test_task_lists_deduped(self) -> None:
        result = _result(
            executed_tasks=("research_funding", "research_funding"),
            failed_tasks=("a", "a"),
            unresolved_tasks=("b", "b"),
        )
        assert result.executed_tasks == ("research_funding",)
        assert result.failed_tasks == ("a",)
        assert result.unresolved_tasks == ("b",)

    def test_serialization_roundtrip(self) -> None:
        result = _result()
        restored = CollectionResult.from_dict(result.to_dict())
        assert restored == result
        assert CollectionResult.from_dict(
            json.loads(json.dumps(result.to_dict()))
        ) == result
        assert result.to_dict()["status"] == "success"

    def test_immutability(self) -> None:
        result = _result()
        with pytest.raises(FrozenInstanceError):
            result.collection_id = "changed"  # type: ignore[misc]
