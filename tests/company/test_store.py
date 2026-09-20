"""Behavioral tests for PostgresCompanyStore against in-memory SQLite.

These tests own identity adoption, snapshot idempotency, pagination, and
ownership scoping — the correctness-critical behaviours of Phase 1.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.company import Company, CompanySnapshot
from app.services.company_postgres import PostgresCompanyStore, compute_snapshot_id
from app.services.company_protocols import CompanyIdentity, CompanySnapshotPayload

pytestmark = pytest.mark.asyncio

NOW = datetime(2025, 5, 1, 12, 0, tzinfo=UTC)


def _uid(material: str) -> str:
    import hashlib

    return hashlib.sha256(material.encode()).hexdigest()


def _identity(
    name: str = "Acme Inc",
    website: str = "acme.com",
    company_id: str | None = None,
) -> CompanyIdentity:
    from app.services.companies import CompanyIdentityResolver

    resolver = CompanyIdentityResolver()
    identity = resolver.resolve(name, website)
    if company_id is not None:
        return CompanyIdentity(
            company_id=company_id,
            canonical_name=identity.canonical_name,
            canonical_domain=identity.canonical_domain,
            canonical_name_key=identity.canonical_name_key,
            fallback_slug=identity.fallback_slug,
            primary_name=identity.primary_name,
            website=identity.website,
            match_source=identity.match_source,
        )
    return identity


def _payload(
    company_id: str,
    analysis_id: str = "aid-1",
    decision: str | None = "invest",
    created_at: datetime | None = None,
) -> CompanySnapshotPayload:
    return CompanySnapshotPayload(
        company_id=company_id,
        analysis_id=analysis_id,
        report_id=f"report-{analysis_id}",
        decision=decision,
        confidence=0.8,
        composite_score=75.0,
        readiness_score=60.0,
        dimension_scores={"market": 80.0},
        created_at=created_at or NOW,
    )


# ── upsert_company: create ──────────────────────────────────────────────


async def test_upsert_creates_new_company(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    rec = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    assert rec.company_id
    assert rec.user_id == "u1"
    assert rec.snapshot_count == 0


async def test_upsert_sets_first_last_seen(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    seen = NOW
    rec = await store.upsert_company(
        sqlite_session, _identity(), user_id="u1", seen_at=seen
    )
    assert rec.first_seen == seen
    assert rec.last_seen == seen


async def test_upsert_latest_metrics_stored(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    rec = await store.upsert_company(
        sqlite_session,
        _identity(),
        user_id="u1",
        latest_decision="invest",
        latest_confidence=0.85,
        latest_composite_score=72.0,
    )
    assert rec.latest_decision == "invest"
    assert rec.latest_confidence == 0.85
    assert rec.latest_composite_score == 72.0


async def test_upsert_unique_company_ids_distinct_rows(
    sqlite_session: AsyncSession,
):
    store = PostgresCompanyStore()
    await store.upsert_company(
        sqlite_session, _identity(name="Acme", website=None), user_id="u1"
    )
    await store.upsert_company(
        sqlite_session, _identity(name="Betty", website=None), user_id="u1"
    )
    count = (
        await sqlite_session.execute(select(func.count()).select_from(Company))
    ).scalar_one()
    assert count == 2


# ── upsert_company: identity adoption ───────────────────────────────────


async def test_upsert_adopts_existing_by_id(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    first = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    second = await store.upsert_company(
        sqlite_session,
        _identity(name="Acme Incorporated", website="acme.com"),
        user_id="u1",
    )
    assert second.company_id == first.company_id


async def test_upsert_adopts_existing_same_name_different_website(
    sqlite_session: AsyncSession,
):
    store = PostgresCompanyStore()
    first = await store.upsert_company(
        sqlite_session, _identity(name="Acme", website=None), user_id="u1"
    )
    second = await store.upsert_company(
        sqlite_session, _identity(name="Acme", website="acme.com"), user_id="u1"
    )
    assert second.company_id == first.company_id


async def test_upsert_adopts_by_domain_when_id_differs(
    sqlite_session: AsyncSession,
):
    store = PostgresCompanyStore()
    first = await store.upsert_company(
        sqlite_session,
        _identity(name="Acme", website="acme.com", company_id=_uid("oldid")),
        user_id="u1",
    )
    second = await store.upsert_company(
        sqlite_session,
        _identity(name="Acme", website="acme.com", company_id=_uid("newid")),
        user_id="u1",
    )
    assert second.company_id == first.company_id


async def test_upsert_does_not_merge_different_companies(
    sqlite_session: AsyncSession,
):
    store = PostgresCompanyStore()
    a = await store.upsert_company(
        sqlite_session, _identity(name="Acme", website="acme.com"), user_id="u1"
    )
    b = await store.upsert_company(
        sqlite_session, _identity(name="Betty", website="betty.com"), user_id="u1"
    )
    assert a.company_id != b.company_id


async def test_upsert_adopts_keeps_creator_user_id(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    first = await store.upsert_company(sqlite_session, _identity(), user_id="creator")
    second = await store.upsert_company(
        sqlite_session,
        _identity(name="Acme Inc 2", website="acme.com"),
        user_id="other",
    )
    assert first.user_id == "creator"
    assert second.company_id == first.company_id


async def test_upsert_adopts_by_fallback_slug(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    first = await store.upsert_company(
        sqlite_session, _identity(name="", website=None), user_id="u1"
    )
    second = await store.upsert_company(
        sqlite_session, _identity(name="", website=None, company_id=_uid("z")),
        user_id="u1",
    )
    assert second.company_id == first.company_id


async def test_upsert_backfills_domain_on_adoption(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    await store.upsert_company(
        sqlite_session, _identity(name="Acme", website=None), user_id="u1"
    )
    before = await store.get_company(sqlite_session, _uid("acme"), user_id="u1")
    await store.upsert_company(
        sqlite_session, _identity(name="Acme", website="acme.com"), user_id="u1"
    )
    after = await store.get_company(sqlite_session, before.company_id, user_id="u1")
    assert after is not None
    assert after.canonical_domain == "acme.com"


async def test_upsert_updates_last_seen_on_adoption(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    t1 = datetime(2025, 1, 1, tzinfo=UTC)
    t2 = datetime(2025, 6, 1, tzinfo=UTC)
    await store.upsert_company(sqlite_session, _identity(), user_id="u1", seen_at=t1)
    rec = await store.upsert_company(
        sqlite_session, _identity(name="Acme Inc 2", website="acme.com"),
        user_id="u1", seen_at=t2,
    )
    assert rec.last_seen == t2


# ── append_snapshot: create + dedup ─────────────────────────────────────


async def test_append_snapshot_creates_row(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    result = await store.append_snapshot(sqlite_session, _payload(comp.company_id))
    assert result.created is True
    assert result.snapshot.analysis_id == "aid-1"


async def test_append_snapshot_increments_count(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    await store.append_snapshot(sqlite_session, _payload(comp.company_id))
    await store.append_snapshot(
        sqlite_session, _payload(comp.company_id, analysis_id="aid-2")
    )
    fresh = await store.get_company(sqlite_session, comp.company_id, user_id="u1")
    assert fresh is not None
    assert fresh.snapshot_count == 2


async def test_append_duplicate_analysis_id_is_noop(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    first = await store.append_snapshot(sqlite_session, _payload(comp.company_id))
    second = await store.append_snapshot(sqlite_session, _payload(comp.company_id))
    assert first.created is True
    assert second.created is False
    assert second.snapshot.id == first.snapshot.id


async def test_append_duplicate_does_not_increment_count(
    sqlite_session: AsyncSession,
):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    await store.append_snapshot(sqlite_session, _payload(comp.company_id))
    await store.append_snapshot(sqlite_session, _payload(comp.company_id))
    fresh = await store.get_company(sqlite_session, comp.company_id, user_id="u1")
    assert fresh is not None
    assert fresh.snapshot_count == 1


async def test_append_different_analysis_ids_are_both_kept(
    sqlite_session: AsyncSession,
):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    await store.append_snapshot(sqlite_session, _payload(comp.company_id, "aid-1"))
    await store.append_snapshot(sqlite_session, _payload(comp.company_id, "aid-2"))
    page = await store.list_snapshots(sqlite_session, comp.company_id, user_id="u1")
    assert page.total == 2


async def test_snapshot_id_deterministic(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    r1 = await store.append_snapshot(sqlite_session, _payload(comp.company_id, "aid-x"))
    r2 = await store.append_snapshot(sqlite_session, _payload(comp.company_id, "aid-x"))
    assert r1.snapshot.id == compute_snapshot_id(comp.company_id, "aid-x")
    assert r2.snapshot.id == r1.snapshot.id


async def test_same_analysis_id_globally_deduplicated(
    sqlite_session: AsyncSession,
):
    store = PostgresCompanyStore()
    ca = await store.upsert_company(
        sqlite_session, _identity(name="Acme", website="acme.com"), user_id="u1"
    )
    cb = await store.upsert_company(
        sqlite_session, _identity(name="Betty", website="betty.com"), user_id="u1"
    )
    # analysis_id is globally unique: appending the same analysis to a
    # different company is a no-op (idempotent), never a second snapshot.
    ra = await store.append_snapshot(sqlite_session, _payload(ca.company_id, "aid-1"))
    rb = await store.append_snapshot(sqlite_session, _payload(cb.company_id, "aid-1"))
    assert ra.created is True
    assert rb.created is False
    assert rb.snapshot.id == ra.snapshot.id
    assert rb.snapshot.company_id == ca.company_id


async def test_append_snapshot_sets_last_seen(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    later = NOW + timedelta(days=1)
    await store.append_snapshot(
        sqlite_session, _payload(comp.company_id, created_at=later)
    )
    fresh = await store.get_company(sqlite_session, comp.company_id, user_id="u1")
    assert fresh is not None
    assert fresh.last_seen == later.replace(tzinfo=None)


async def test_append_duplicate_reads_back_existing_row(
    sqlite_session: AsyncSession,
):
    """Read-after-write: a duplicate append returns the surviving row."""
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    first = await store.append_snapshot(sqlite_session, _payload(comp.company_id))
    second = await store.append_snapshot(sqlite_session, _payload(comp.company_id))
    assert second.created is False
    assert second.snapshot.id == first.snapshot.id
    page = await store.list_snapshots(sqlite_session, comp.company_id, user_id="u1")
    assert page.total == 1
    assert page.snapshots[0].id == first.snapshot.id
    assert page.snapshots[0].analysis_id == "aid-1"


def test_pg_append_uses_on_conflict_do_nothing():
    """PostgreSQL appends compile to INSERT ... ON CONFLICT DO NOTHING."""
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from app.models.company import CompanySnapshot

    values = {
        "id": "snap",
        "company_id": "c",
        "analysis_id": "aid",
        "report_id": "rid",
        "decision": "invest",
        "confidence": 0.8,
        "composite_score": 75.0,
        "readiness_score": 60.0,
        "dimension_scores": {},
        "created_at": NOW,
    }
    stmt = pg_insert(CompanySnapshot).values(**values).on_conflict_do_nothing(
        index_elements=[CompanySnapshot.analysis_id]
    )
    sql = str(stmt.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT" in sql
    assert "DO NOTHING" in sql
    assert "analysis_id" in sql


def test_sqlite_append_uses_insert_or_ignore():
    """SQLite appends compile to INSERT OR IGNORE."""
    from sqlalchemy.dialects import sqlite
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    from app.models.company import CompanySnapshot

    values = {
        "id": "snap",
        "company_id": "c",
        "analysis_id": "aid",
        "report_id": "rid",
        "decision": "invest",
        "confidence": 0.8,
        "composite_score": 75.0,
        "readiness_score": 60.0,
        "dimension_scores": {},
        "created_at": NOW,
    }
    stmt = sqlite_insert(CompanySnapshot).values(**values).on_conflict_do_nothing()
    sql = str(stmt.compile(dialect=sqlite.dialect()))
    assert "ON CONFLICT" in sql
    assert "DO NOTHING" in sql


# ── get_company ─────────────────────────────────────────────────────────


async def test_get_company_returns_record(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    found = await store.get_company(sqlite_session, comp.company_id, user_id="u1")
    assert found is not None
    assert found.company_id == comp.company_id


async def test_get_company_unknown_returns_none(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    assert await store.get_company(sqlite_session, "nope", user_id="u1") is None


async def test_get_company_scoped_to_other_user(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    assert await store.get_company(sqlite_session, comp.company_id, user_id="u2") is None


async def test_get_company_anonymous_cannot_see_user_owned(
    sqlite_session: AsyncSession,
):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    assert await store.get_company(sqlite_session, comp.company_id, user_id=None) is None


async def test_get_company_anonymous_scope_returns_null_owned(
    sqlite_session: AsyncSession,
):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id=None)
    found = await store.get_company(sqlite_session, comp.company_id, user_id=None)
    assert found is not None


# ── list_companies ──────────────────────────────────────────────────────


async def test_list_empty(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    page = await store.list_companies(sqlite_session, user_id="u1")
    assert page.companies == []
    assert page.total == 0


async def test_list_orders_newest_first(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    t1 = datetime(2025, 1, 1, tzinfo=UTC)
    t2 = datetime(2025, 2, 1, tzinfo=UTC)
    await store.upsert_company(
        sqlite_session, _identity(name="Old", website="old.com"),
        user_id="u1", seen_at=t1,
    )
    await store.upsert_company(
        sqlite_session, _identity(name="New", website="new.com"),
        user_id="u1", seen_at=t2,
    )
    page = await store.list_companies(sqlite_session, user_id="u1")
    assert [c.primary_name for c in page.companies] == ["New", "Old"]
    assert page.total == 2


async def test_list_respects_owner_scope(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    await store.upsert_company(
        sqlite_session, _identity(name="Mine", website="mine.com"), user_id="u1"
    )
    await store.upsert_company(
        sqlite_session, _identity(name="Theirs", website="theirs.com"), user_id="u2"
    )
    page = await store.list_companies(sqlite_session, user_id="u1")
    assert [c.primary_name for c in page.companies] == ["Mine"]
    assert page.total == 1


async def test_list_anonymous_only_null_owned(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    await store.upsert_company(
        sqlite_session, _identity(name="Anon", website="anon.com"), user_id=None
    )
    await store.upsert_company(
        sqlite_session, _identity(name="Mine", website="mine.com"), user_id="u1"
    )
    page = await store.list_companies(sqlite_session, user_id=None)
    assert [c.primary_name for c in page.companies] == ["Anon"]
    assert page.total == 1


async def test_list_pagination_offset_limit(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    for i in range(5):
        await store.upsert_company(
            sqlite_session,
            _identity(name=f"C{i}", website=f"c{i}.com"),
            user_id="u1",
            seen_at=datetime(2025, 1, 1 + i, tzinfo=UTC),
        )
    page = await store.list_companies(sqlite_session, user_id="u1", offset=1, limit=2)
    assert len(page.companies) == 2
    assert page.total == 5


# ── list_snapshots ──────────────────────────────────────────────────────


async def test_list_snapshots_newest_first(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    await store.append_snapshot(
        sqlite_session,
        _payload(comp.company_id, "aid-1", created_at=datetime(2025, 1, 1, tzinfo=UTC)),
    )
    await store.append_snapshot(
        sqlite_session,
        _payload(comp.company_id, "aid-2", created_at=datetime(2025, 2, 1, tzinfo=UTC)),
    )
    page = await store.list_snapshots(sqlite_session, comp.company_id, user_id="u1")
    assert [s.analysis_id for s in page.snapshots] == ["aid-2", "aid-1"]
    assert page.total == 2


async def test_list_snapshots_scoped_to_owner(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    await store.append_snapshot(sqlite_session, _payload(comp.company_id))
    page = await store.list_snapshots(sqlite_session, comp.company_id, user_id="u2")
    assert page.snapshots == []
    assert page.total == 0


async def test_list_snapshots_empty(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    page = await store.list_snapshots(sqlite_session, comp.company_id, user_id="u1")
    assert page.snapshots == []
    assert page.total == 0


async def test_list_snapshots_paginated(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    for i in range(4):
        await store.append_snapshot(
            sqlite_session,
            _payload(
                comp.company_id,
                f"aid-{i}",
                created_at=datetime(2025, 1, 1 + i, tzinfo=UTC),
            ),
        )
    page = await store.list_snapshots(sqlite_session, comp.company_id,
                                      user_id="u1", offset=1, limit=2)
    assert len(page.snapshots) == 2
    assert page.total == 4


# ── raw DB invariants ───────────────────────────────────────────────────


async def test_company_row_persists_columns(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    await sqlite_session.commit()
    row = await sqlite_session.get(Company, comp.company_id)
    assert row is not None
    assert row.canonical_domain == "acme.com"
    assert row.primary_name == "Acme Inc"


async def test_snapshot_row_persists_columns(sqlite_session: AsyncSession):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    result = await store.append_snapshot(sqlite_session, _payload(comp.company_id))
    await sqlite_session.commit()
    row = await sqlite_session.get(CompanySnapshot, result.snapshot.id)
    assert row is not None
    assert row.decision == "invest"
    assert row.dimension_scores == {"market": 80.0}


async def test_snapshot_unique_analysis_id_is_enforced_in_db(
    sqlite_session: AsyncSession,
):
    store = PostgresCompanyStore()
    comp = await store.upsert_company(sqlite_session, _identity(), user_id="u1")
    await store.append_snapshot(sqlite_session, _payload(comp.company_id, "aid-dup"))
    await sqlite_session.commit()
    stmt = select(CompanySnapshot).where(CompanySnapshot.analysis_id == "aid-dup")
    rows = (await sqlite_session.execute(stmt)).scalars().all()
    assert len(rows) == 1


# ── compute_snapshot_id unit tests ──────────────────────────────────────


def test_compute_snapshot_id_is_stable():
    assert compute_snapshot_id("c1", "a1") == compute_snapshot_id("c1", "a1")


def test_compute_snapshot_id_differs_by_company():
    assert compute_snapshot_id("c1", "a1") != compute_snapshot_id("c2", "a1")


def test_compute_snapshot_id_differs_by_analysis():
    assert compute_snapshot_id("c1", "a1") != compute_snapshot_id("c1", "a2")


def test_compute_snapshot_id_is_sha256_hex():
    import hashlib

    assert compute_snapshot_id("c", "a") == hashlib.sha256(
        b"c\x00a"
    ).hexdigest()


def test_compute_snapshot_id_is_64_chars():
    assert len(compute_snapshot_id("c", "a")) == 64


def test_compute_snapshot_id_unicode_safe():
    assert compute_snapshot_id("café", "分析") == compute_snapshot_id(
        "café", "分析"
    )
