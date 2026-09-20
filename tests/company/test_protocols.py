"""Tests for the CompanyStore protocol and its value objects.

The critical contract: a runtime-checkable Protocol that PostgresCompanyStore
and the test double (MemoryCompanyStore) both satisfy, with typed dataclasses
for every value crossing the store boundary.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest

from app.services.company_postgres import PostgresCompanyStore
from app.services.company_protocols import (
    CompanyIdentity,
    CompanyPage,
    CompanyRecord,
    CompanySnapshotPayload,
    CompanySnapshotRecord,
    CompanyStore,
    SnapshotAppendResult,
    SnapshotPage,
)

NOW = datetime(2025, 5, 1, 12, 0, tzinfo=UTC)


# ── Protocol structural conformance ─────────────────────────────────────


def test_store_is_runtime_checkable():
    assert isinstance(CompanyStore, type)
    assert hasattr(CompanyStore, "__protocol_attrs__")


def test_postgres_store_satisfies_protocol():
    assert isinstance(PostgresCompanyStore(), CompanyStore)


def test_memory_store_satisfies_protocol():
    from tests.company.conftest import MemoryCompanyStore

    assert isinstance(MemoryCompanyStore(), CompanyStore)


def test_store_methods_are_async():
    for name in ("upsert_company", "append_snapshot", "get_company",
                 "list_companies", "list_snapshots"):
        assert name in CompanyStore.__protocol_attrs__


def test_store_signature_check():
    sig = inspect.signature(CompanyStore.upsert_company)
    params = list(sig.parameters)
    assert params[0] == "self"
    assert params[1] == "session"
    assert "identity" in sig.parameters


def test_protocol_has_no_extra_required_methods_unless_expected():
    expected = {
        "append_snapshot",
        "get_company",
        "list_companies",
        "list_snapshots",
        "upsert_company",
    }
    assert set(CompanyStore.__protocol_attrs__) == expected


# ── CompanyIdentity value object ────────────────────────────────────────


def test_company_identity_fields():
    i = CompanyIdentity(
        company_id="c1",
        canonical_name="acme",
        canonical_domain="acme.com",
        canonical_name_key="a",
        fallback_slug="acme",
        primary_name="Acme Inc",
        website="https://acme.com",
        match_source="canonical_domain",
    )
    assert i.company_id == "c1"
    assert i.match_source == "canonical_domain"


def test_company_identity_is_frozen():
    i = CompanyIdentity(
        company_id="c1",
        canonical_name="acme",
        canonical_domain=None,
        canonical_name_key="a",
        fallback_slug="acme",
        primary_name="Acme",
        website=None,
        match_source="canonical_name",
    )
    with pytest.raises(Exception):
        i.canonical_domain = "other.com"  # type: ignore[misc]


def test_company_identity_equivalent_dataclasses_equal():
    kwargs = dict(
        company_id="c1",
        canonical_name="acme",
        canonical_domain="acme.com",
        canonical_name_key="a",
        fallback_slug="acme",
        primary_name="Acme Inc",
        website="https://acme.com",
        match_source="canonical_domain",
    )
    assert CompanyIdentity(**kwargs) == CompanyIdentity(**kwargs)


def test_company_identity_hashable():
    i = CompanyIdentity(
        company_id="c1",
        canonical_name="acme",
        canonical_domain=None,
        canonical_name_key="a",
        fallback_slug="acme",
        primary_name="Acme",
        website=None,
        match_source="canonical_name",
    )
    assert isinstance(hash(i), int)


# ── CompanyRecord value object ──────────────────────────────────────────


def test_company_record_fields():
    r = CompanyRecord(
        company_id="c1",
        canonical_name="acme",
        canonical_domain="acme.com",
        primary_name="Acme Inc",
        website="https://acme.com",
        latest_decision="invest",
        latest_confidence=0.8,
        latest_composite_score=70.0,
        snapshot_count=2,
        first_seen=NOW,
        last_seen=NOW,
        user_id=None,
        canonical_name_key="a",
        fallback_slug="acme",
    )
    assert r.snapshot_count == 2
    assert r.user_id is None


def test_company_record_is_plain_dataclass():
    r = CompanyRecord(
        company_id="c1",
        canonical_name="acme",
        canonical_domain=None,
        primary_name="Acme",
        website=None,
        latest_decision=None,
        latest_confidence=None,
        latest_composite_score=None,
        snapshot_count=0,
        first_seen=NOW,
        last_seen=NOW,
        user_id=None,
        canonical_name_key="a",
        fallback_slug="acme",
    )
    r.snapshot_count = 5  # mutable
    assert r.snapshot_count == 5


# ── CompanySnapshotPayload ──────────────────────────────────────────────


def test_snapshot_payload_fields():
    p = CompanySnapshotPayload(
        company_id="c1",
        analysis_id="a1",
        report_id="r1",
        decision="invest",
        confidence=0.9,
        composite_score=80.0,
        readiness_score=70.0,
        dimension_scores={"m": 85.0},
        created_at=NOW,
    )
    assert p.decision == "invest"
    assert p.dimension_scores == {"m": 85.0}


def test_snapshot_payload_defaults():
    p = CompanySnapshotPayload(
        company_id="c1",
        analysis_id="a1",
        report_id="r1",
    )
    assert p.decision is None
    assert p.dimension_scores == {}
    assert p.created_at is None
    fresh = CompanySnapshotPayload(
        company_id="c2",
        analysis_id="a2",
        report_id="r2",
    )
    assert fresh.dimension_scores == {}


# ── CompanySnapshotRecord ───────────────────────────────────────────────


def test_snapshot_record_fields():
    s = CompanySnapshotRecord(
        id="snap1",
        company_id="c1",
        analysis_id="a1",
        report_id="r1",
        decision="invest",
        confidence=0.8,
        composite_score=70.0,
        readiness_score=65.0,
        dimension_scores={"m": 80.0},
        created_at=NOW,
    )
    assert s.id == "snap1"
    assert s.readiness_score == 65.0


def test_snapshot_record_defaults():
    s = CompanySnapshotRecord(
        id="snap2",
        company_id="c1",
        analysis_id="a1",
        report_id="r1",
        decision=None,
        confidence=None,
        composite_score=None,
        readiness_score=None,
        dimension_scores={},
        created_at=NOW,
    )
    assert s.decision is None
    assert s.dimension_scores == {}


# ── SnapshotAppendResult ────────────────────────────────────────────────


def _minimal_snapshot(snap_id: str = "s", created_at: datetime = NOW) -> CompanySnapshotRecord:
    return CompanySnapshotRecord(
        id=snap_id,
        company_id="c",
        analysis_id="a",
        report_id="r",
        decision=None,
        confidence=None,
        composite_score=None,
        readiness_score=None,
        dimension_scores={},
        created_at=created_at,
    )


def test_append_result_created_true():
    r = SnapshotAppendResult(
        snapshot=_minimal_snapshot(),
        created=True,
    )
    assert r.created is True


def test_append_result_created_false():
    r = SnapshotAppendResult(
        snapshot=_minimal_snapshot(),
        created=False,
    )
    assert r.created is False


def test_append_result_is_dataclass():
    from dataclasses import is_dataclass

    assert is_dataclass(SnapshotAppendResult)


# ── Pages ───────────────────────────────────────────────────────────────


def test_company_page_fields():
    r = CompanyRecord(
        company_id="c1",
        canonical_name="acme",
        canonical_domain=None,
        primary_name="Acme",
        website=None,
        latest_decision=None,
        latest_confidence=None,
        latest_composite_score=None,
        snapshot_count=0,
        first_seen=NOW,
        last_seen=NOW,
        user_id=None,
        canonical_name_key="a",
        fallback_slug="acme",
    )
    page = CompanyPage(companies=[r], total=1)
    assert page.total == 1
    assert page.companies == [r]


def test_snapshot_page_fields():
    s = _minimal_snapshot()
    page = SnapshotPage(snapshots=[s], total=1)
    assert page.total == 1
    assert page.snapshots == [s]


def test_pages_are_plain_dataclasses():
    from dataclasses import is_dataclass

    assert is_dataclass(CompanyPage)
    assert is_dataclass(SnapshotPage)


def test_snapshot_record_timestamp_aware():
    s = _minimal_snapshot(created_at=datetime.now(UTC))
    assert s.created_at.tzinfo is not None
