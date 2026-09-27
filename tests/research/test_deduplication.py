"""Tests for evidence deduplication — Phase 8 Sprint 6.

Covers the deterministic key derivations, the append-only
:class:`DeduplicationIndex`, and duplicate-group reporting.
"""

from __future__ import annotations

import pytest

from predictron_engine.research.deduplication import (
    DEDUPLICATION_SCHEMA_VERSION,
    DeduplicationIndex,
    DuplicateGroup,
    EvidenceKeys,
    duplicate_key,
    evidence_claim_key,
    evidence_fingerprint_key,
    evidence_keys,
    evidence_source_key,
    evidence_title_key,
    evidence_url_key,
    find_duplicate_groups,
)
from predictron_engine.research.exceptions import DeduplicationError

from .memory_helpers import make_evidence


def test_fingerprint_key_is_stable_for_one_fingerprint() -> None:
    first = evidence_fingerprint_key("abc123")
    assert first == evidence_fingerprint_key("abc123")
    assert first != evidence_fingerprint_key("def456")


def test_fingerprint_key_rejects_an_empty_fingerprint() -> None:
    with pytest.raises(DeduplicationError):
        evidence_fingerprint_key("   ")


def test_url_key_normalizes_before_hashing() -> None:
    assert evidence_url_key("https://x.test/a#frag") == evidence_url_key(
        "HTTPS://X.test/a"
    )


def test_url_key_of_a_blank_url_is_absent() -> None:
    assert evidence_url_key("  ") is None


def test_title_key_normalizes_before_hashing() -> None:
    assert evidence_title_key("Acme  Filing!") == evidence_title_key("acme filing")


def test_title_key_of_a_blank_title_is_absent() -> None:
    assert evidence_title_key("  ") is None


def test_source_key_combines_category_identifier_and_url() -> None:
    item = make_evidence(source="sec-1")
    key = evidence_source_key(item)
    assert key is not None
    assert key == evidence_source_key(make_evidence(source="sec-1"))
    assert key != evidence_source_key(make_evidence(source="sec-2"))


def test_source_key_is_absent_without_a_url() -> None:
    item = make_evidence()
    object.__setattr__(item.reference, "url", "   ")
    assert evidence_source_key(item) is None


def test_url_and_title_branches_on_a_reference_without_them() -> None:
    item = make_evidence()
    object.__setattr__(item.reference, "url", "  ")
    object.__setattr__(item.reference, "description", "  ")
    keys = evidence_keys(item)
    assert keys.url is None
    assert keys.title is None
    assert keys.claim


def test_claim_key_is_always_present() -> None:
    assert evidence_claim_key(make_evidence(claim="Acme ships X"))


def test_claim_key_differentiates_claims() -> None:
    assert evidence_claim_key(make_evidence(claim="A")) != evidence_claim_key(
        make_evidence(claim="B")
    )


def test_evidence_keys_reports_only_present_keys() -> None:
    keys = evidence_keys(make_evidence(source="sec-1", url="https://x.test/a"))
    assert keys.claim and keys.fingerprint and keys.source
    assert keys.url and keys.title


def test_evidence_keys_of_a_bare_item_drops_the_url() -> None:
    item = make_evidence()
    object.__setattr__(item.reference, "url", "")
    object.__setattr__(item.reference, "description", "")
    keys = evidence_keys(item)
    assert keys.url is None
    assert keys.title is None
    assert keys.source is None, "the source key needs a URL"
    assert keys.claim


def test_present_lists_present_keys_in_a_stable_order() -> None:
    keys = evidence_keys(make_evidence())
    assert keys.present() == tuple(
        value
        for value in (keys.fingerprint, keys.source, keys.url, keys.title, keys.claim)
        if value is not None
    )
    assert keys.duplicate not in keys.present()


def test_evidence_keys_round_trip() -> None:
    keys = evidence_keys(make_evidence())
    assert EvidenceKeys.from_dict(keys.to_dict()) == keys


def test_evidence_keys_from_dict_rejects_a_foreign_shape() -> None:
    with pytest.raises(DeduplicationError):
        EvidenceKeys.from_dict({"claim": 7})


def test_duplicate_key_is_a_digest_of_the_keys() -> None:
    item = make_evidence()
    assert duplicate_key(item) == duplicate_key(make_evidence())


def test_duplicate_key_differs_between_items() -> None:
    assert duplicate_key(make_evidence(claim="A")) != duplicate_key(
        make_evidence(claim="B")
    )


def test_index_records_and_recognises_the_same_item() -> None:
    index = DeduplicationIndex()
    item = make_evidence()
    key = index.record(item)
    assert key == index.key_for(item)
    assert index.existing_duplicate(item) is None, "nothing was seen before it"


def test_index_recognises_an_item_it_has_already_seen() -> None:
    index = DeduplicationIndex()
    item = make_evidence()
    index.record(item)
    assert index.is_duplicate(item) is True
    assert index.existing_duplicate(item) is None, "it repeats nothing else"


def test_index_recognises_a_different_item_with_the_same_keys() -> None:
    index = DeduplicationIndex()
    original = make_evidence()
    index.record(original)
    repeat = make_evidence(task_id="task-9")
    assert repeat.evidence_id != original.evidence_id
    assert index.is_duplicate(repeat) is True
    assert index.existing_duplicate(repeat) == original.evidence_id


def test_index_records_the_same_item_twice_without_growing() -> None:
    index = DeduplicationIndex()
    item = make_evidence()
    index.record(item)
    index.record(item)
    assert len(index) == 1
    assert index.indexed_evidence_ids() == (item.evidence_id,)


def test_index_does_not_mutate_a_recorded_item() -> None:
    index = DeduplicationIndex()
    item = make_evidence()
    index.record(item)
    assert item.reference.url == "https://example.test/a"


def test_index_record_all_returns_keys_in_order() -> None:
    index = DeduplicationIndex()
    keys = index.record_all([make_evidence(claim="A"), make_evidence(claim="B")])
    assert len(keys) == 2
    assert len(index) == 2


def test_index_reports_evidence_per_key() -> None:
    index = DeduplicationIndex()
    first = make_evidence(claim="A", url="https://x.test/a", source="s1")
    second = make_evidence(claim="A", url="https://x.test/a", source="s1", task_id="t2")
    index.record(first)
    index.record(second)
    claim_key = index.keys_for(first).claim
    assert index.evidence_ids_for_key(claim_key) == tuple(
        sorted((first.evidence_id, second.evidence_id))
    )


def test_index_reports_the_keys_of_a_recorded_item() -> None:
    index = DeduplicationIndex()
    item = make_evidence()
    index.record(item)
    assert index.keys_for_evidence(item.evidence_id) == index.keys_for(item)


def test_index_reports_nothing_for_an_unknown_item() -> None:
    index = DeduplicationIndex()
    assert index.keys_for_evidence("ev_missing") is None
    assert index.evidence_ids_for_key("missing") == ()


def test_index_reports_shared_identifiers_with_a_stored_item() -> None:
    index = DeduplicationIndex()
    original = make_evidence(claim="A", url="https://x.test/a", source="s1")
    index.record(original)
    other = make_evidence(claim="B", url="https://x.test/a", source="s1", task_id="t2")
    assert index.shared_with(other) == (original.evidence_id,)
    assert index.shared_with(original) == ()


def test_index_counts_no_duplicates_for_distinct_items() -> None:
    index = DeduplicationIndex()
    index.record_all(
        [
            make_evidence(
                claim="A",
                url="https://x.test/a",
                source="s1",
                title="Doc A",
                task_id="t1",
            ),
            make_evidence(
                claim="B",
                url="https://x.test/b",
                source="s2",
                title="Doc B",
                task_id="t2",
            ),
        ]
    )
    assert index.duplicate_count() == 0
    assert index.duplicate_groups() == ()


def test_index_counts_every_identifier_in_a_collision() -> None:
    index = DeduplicationIndex()
    first = make_evidence(
        claim="A", url="https://x.test/a", source="s1", task_id="t1"
    )
    second = make_evidence(
        claim="A", url="https://x.test/a", source="s1", task_id="t2"
    )
    index.record(first)
    index.record(second)
    assert index.duplicate_count() == 2, "both identifiers took part"
    groups = index.duplicate_groups()
    assert all(
        set(group.evidence_ids) == {first.evidence_id, second.evidence_id}
        for group in groups
    )
    assert all(group.reason for group in groups)


def test_index_reports_one_group_per_collision_bucket() -> None:
    index = DeduplicationIndex()
    index.record_all(
        [
            make_evidence(
                claim="A", url="https://x.test/a", source="s1", task_id="t1"
            ),
            make_evidence(
                claim="A", url="https://x.test/a", source="s1", task_id="t2"
            ),
            make_evidence(
                claim="B", url="https://x.test/b", source="s2", task_id="t3"
            ),
            make_evidence(
                claim="B", url="https://x.test/b", source="s2", task_id="t4"
            ),
        ]
    )
    groups = index.duplicate_groups()
    claims = {group.evidence_ids for group in groups if group.reason.endswith("claim")}
    assert len(claims) == 2, "one bucket per claim, however many keys collide"
    assert index.duplicate_count() == 4


def test_index_groups_are_ordered_deterministically() -> None:
    index = DeduplicationIndex()
    index.record_all(
        [
            make_evidence(
                claim="A", url="https://x.test/a", source="s1", task_id="t1"
            ),
            make_evidence(
                claim="A", url="https://x.test/a", source="s1", task_id="t2"
            ),
            make_evidence(
                claim="B", url="https://x.test/b", source="s2", task_id="t3"
            ),
            make_evidence(
                claim="B", url="https://x.test/b", source="s2", task_id="t4"
            ),
        ]
    )
    first = [group.to_dict() for group in index.duplicate_groups()]
    rebuilt = DeduplicationIndex.from_dict(index.to_dict())
    assert [group.to_dict() for group in rebuilt.duplicate_groups()] == first


def test_index_lists_recorded_evidence_ids_sorted() -> None:
    index = DeduplicationIndex()
    items = [
        make_evidence(
            claim=str(n),
            url=f"https://x.test/{n}",
            source=f"s{n}",
            task_id=f"t{n}",
        )
        for n in range(3)
    ]
    index.record_all(items)
    assert index.indexed_evidence_ids() == tuple(
        sorted(item.evidence_id for item in items)
    )


def test_index_supports_membership_and_length() -> None:
    index = DeduplicationIndex()
    item = make_evidence()
    index.record(item)
    assert len(index) == 1
    assert item.evidence_id in index
    assert "ev_missing" not in index


def test_index_round_trips_through_a_dictionary() -> None:
    index = DeduplicationIndex()
    first = make_evidence(
        claim="A", url="https://x.test/a", source="s1", title="Doc A", task_id="t1"
    )
    second = make_evidence(
        claim="A", url="https://x.test/a", source="s1", title="Doc A", task_id="t2"
    )
    index.record(first)
    index.record(second)
    document = index.to_dict()
    rebuilt = DeduplicationIndex.from_dict(document)
    assert rebuilt.to_dict() == document
    assert rebuilt.duplicate_count() == 2
    assert rebuilt.existing_duplicate(second) == first.evidence_id


def test_index_schema_version_is_exported() -> None:
    assert DEDUPLICATION_SCHEMA_VERSION == "1.0.0"
    assert DeduplicationIndex().to_dict()["schema_version"] == (
        DEDUPLICATION_SCHEMA_VERSION
    )


def test_find_duplicate_groups_accepts_a_sequence_of_items() -> None:
    items = [
        make_evidence(claim="A", url="https://x.test/a", source="s1", task_id="t1"),
        make_evidence(claim="A", url="https://x.test/a", source="s1", task_id="t2"),
    ]
    groups = find_duplicate_groups(items)
    assert groups
    assert isinstance(groups[0], DuplicateGroup)
    assert set(groups[0].evidence_ids) == {
        items[0].evidence_id,
        items[1].evidence_id,
    }


def test_find_duplicate_groups_of_distinct_items_is_empty() -> None:
    items = [
        make_evidence(
            claim=str(n),
            url=f"https://x.test/{n}",
            source=f"s{n}",
            title=f"Doc {n}",
            task_id=f"t{n}",
        )
        for n in range(3)
    ]
    assert find_duplicate_groups(items) == ()


def test_duplicate_group_reports_its_size() -> None:
    index = DeduplicationIndex()
    index.record_all(
        [
            make_evidence(claim="A", task_id="task-1"),
            make_evidence(claim="A", task_id="task-2"),
            make_evidence(claim="A", task_id="task-3"),
        ]
    )
    assert index.duplicate_groups()[0].size == 3


def test_duplicate_group_round_trips() -> None:
    group = DuplicateGroup(
        key="clm_abc",
        reason="normalized_claim",
        evidence_ids=("ev_1", "ev_2"),
    )
    assert DuplicateGroup.from_dict(group.to_dict()) == group


def test_duplicate_group_needs_at_least_two_identifiers() -> None:
    with pytest.raises(DeduplicationError):
        DuplicateGroup(
            key="clm_abc",
            reason="normalized_claim",
            evidence_ids=("ev_1",),
        )


def test_duplicate_group_rejects_a_foreign_shape() -> None:
    with pytest.raises(DeduplicationError):
        DuplicateGroup.from_dict(
            {"key": "k", "reason": "r", "evidence_ids": "nope"}
        )


def test_index_rejects_a_non_mapping() -> None:
    with pytest.raises((DeduplicationError, TypeError, ValueError)):
        DeduplicationIndex.from_dict([])  # type: ignore[arg-type]


def test_index_treats_an_unknown_payload_as_empty() -> None:
    rebuilt = DeduplicationIndex.from_dict({})
    assert len(rebuilt) == 0
    assert rebuilt.to_dict() == DeduplicationIndex().to_dict()
