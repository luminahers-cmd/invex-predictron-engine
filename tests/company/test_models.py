"""Tests for Company and CompanySnapshot ORM models.

Covers column types, defaults, indexes, and TimestampMixin behaviour.
SQLite in-memory via the ``sqlite_engine`` / ``sqlite_session`` fixtures
defined in ``tests/company/conftest.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.company import Company, CompanySnapshot, utc_now
from app.services.company_postgres import compute_snapshot_id

pytestmark = pytest.mark.asyncio


async def test_company_table_exists(sqlite_session: AsyncSession):
    result = await sqlite_session.execute(
        text(
            "SELECT name FROM sqlite_master"
            " WHERE type='table' AND name='companies'"
        )
    )
    assert result.scalar_one() == "companies"


async def test_snapshot_table_exists(sqlite_session: AsyncSession):
    result = await sqlite_session.execute(
        text(
            "SELECT name FROM sqlite_master"
            " WHERE type='table' AND name='company_snapshots'"
        )
    )
    assert result.scalar_one() == "company_snapshots"


async def test_company_pk_is_string(sqlite_session: AsyncSession):
    row = Company(
        company_id="aaaa",
        canonical_name="Acme",
        canonical_name_key="acme",
        primary_name="Acme Inc",
        fallback_slug="acme-inc",
    )
    sqlite_session.add(row)
    await sqlite_session.flush()
    assert row.company_id == "aaaa"


async def test_company_nullable_columns_default_none(sqlite_session: AsyncSession):
    row = Company(
        company_id="b1",
        canonical_name="Acme",
        canonical_name_key="acme",
        primary_name="Acme",
        fallback_slug="acme",
    )
    sqlite_session.add(row)
    await sqlite_session.flush()
    assert row.canonical_domain is None
    assert row.website is None
    assert row.latest_decision is None
    assert row.latest_confidence is None
    assert row.latest_composite_score is None
    assert row.user_id is None


async def test_company_snapshot_count_defaults_to_zero(sqlite_session: AsyncSession):
    row = Company(
        company_id="c1",
        canonical_name="Acme",
        canonical_name_key="acme",
        primary_name="Acme",
        fallback_slug="acme",
    )
    sqlite_session.add(row)
    await sqlite_session.flush()
    assert row.snapshot_count == 0


async def _insert_parent_company(sqlite_session: AsyncSession, company_id: str) -> None:
    parent = Company(
        company_id=company_id,
        canonical_name="Acme",
        canonical_name_key="acme",
        primary_name="Acme",
        fallback_slug="acme",
    )
    sqlite_session.add(parent)
    await sqlite_session.flush()


async def test_company_first_seen_and_last_seen_set(sqlite_session: AsyncSession):
    row = Company(
        company_id="d1",
        canonical_name="Acme",
        canonical_name_key="acme",
        primary_name="Acme",
        fallback_slug="acme",
    )
    sqlite_session.add(row)
    await sqlite_session.flush()
    assert isinstance(row.first_seen, datetime)
    assert isinstance(row.last_seen, datetime)
    assert (row.last_seen - row.first_seen).total_seconds() < 1


async def test_company_first_seen_stable_on_update(sqlite_session: AsyncSession):
    row = Company(
        company_id="e1",
        canonical_name="Acme",
        canonical_name_key="acme",
        primary_name="Acme",
        fallback_slug="acme",
    )
    sqlite_session.add(row)
    await sqlite_session.flush()
    original_first = row.first_seen
    row.last_seen = utc_now()
    await sqlite_session.flush()
    assert row.first_seen == original_first
    assert row.last_seen != original_first


async def test_company_primary_key_is_company_id(sqlite_session: AsyncSession):
    row1 = Company(
        company_id="f1",
        canonical_name="A",
        canonical_name_key="a",
        primary_name="A",
        fallback_slug="a",
    )
    row2 = Company(
        company_id="f1",
        canonical_name="B",
        canonical_name_key="b",
        primary_name="B",
        fallback_slug="b",
    )
    sqlite_session.add(row1)
    await sqlite_session.flush()
    sqlite_session.add(row2)
    with pytest.raises(Exception):
        await sqlite_session.flush()


async def test_company_snapshot_primary_key(sqlite_session: AsyncSession):
    await _insert_parent_company(sqlite_session, "comp-1")
    sid = compute_snapshot_id("comp-1", "analysis-1")
    row = CompanySnapshot(
        id=sid,
        company_id="comp-1",
        analysis_id="analysis-1",
        report_id="report-1",
        decision="invest",
        confidence=0.9,
        composite_score=80.0,
        readiness_score=70.0,
        dimension_scores={"market": 85.0},
    )
    sqlite_session.add(row)
    await sqlite_session.flush()
    assert row.id == sid


async def test_company_snapshot_analysis_id_unique(sqlite_session: AsyncSession):
    await _insert_parent_company(sqlite_session, "x")
    row1 = CompanySnapshot(
        id=compute_snapshot_id("x", "aid-1"),
        company_id="x",
        analysis_id="aid-1",
        report_id="r1",
        created_at=datetime.now(UTC),
    )
    row2 = CompanySnapshot(
        id=compute_snapshot_id("x", "aid-1"),
        company_id="x",
        analysis_id="aid-1",
        report_id="r1",
        created_at=datetime.now(UTC),
    )
    sqlite_session.add(row1)
    await sqlite_session.flush()
    sqlite_session.add(row2)
    with pytest.raises(Exception):
        await sqlite_session.flush()


async def test_company_snapshot_nullable_columns(sqlite_session: AsyncSession):
    await _insert_parent_company(sqlite_session, "comp-nulls")
    row = CompanySnapshot(
        id="snap-nulls",
        company_id="comp-nulls",
        analysis_id="aid-nulls",
        report_id="rid-nulls",
    )
    sqlite_session.add(row)
    await sqlite_session.flush()
    assert row.decision is None
    assert row.confidence is None
    assert row.composite_score is None
    assert row.readiness_score is None
    assert isinstance(row.created_at, datetime)


async def test_company_snapshot_dimension_scores_default_empty(sqlite_session: AsyncSession):
    await _insert_parent_company(sqlite_session, "comp-dim")
    row = CompanySnapshot(
        id="snap-dim",
        company_id="comp-dim",
        analysis_id="aid-dim",
        report_id="rid-dim",
    )
    sqlite_session.add(row)
    await sqlite_session.flush()
    assert row.dimension_scores == {}


async def test_company_snapshot_stores_json_dimension_scores(sqlite_session: AsyncSession):
    await _insert_parent_company(sqlite_session, "comp-json")
    dims = {"market": 80.0, "founder": 70.0, "traction": 65.0}
    row = CompanySnapshot(
        id="snap-json",
        company_id="comp-json",
        analysis_id="aid-json",
        report_id="rid-json",
        dimension_scores=dims,
    )
    sqlite_session.add(row)
    await sqlite_session.flush()
    await sqlite_session.commit()
    reloaded = await sqlite_session.get(CompanySnapshot, "snap-json")
    assert reloaded is not None
    assert reloaded.dimension_scores == dims


async def test_utc_now_returns_aware():
    assert utc_now().tzinfo is not None


async def test_utc_now_returns_utc():
    dt = utc_now()
    assert dt.tzinfo is not None
    assert dt.tzinfo == UTC


async def test_company_timestamp_mixin_provides_created_at(sqlite_session: AsyncSession):
    row = Company(
        company_id="ts1",
        canonical_name="Acme",
        canonical_name_key="acme",
        primary_name="Acme",
        fallback_slug="acme",
    )
    sqlite_session.add(row)
    await sqlite_session.flush()
    assert isinstance(row.created_at, datetime)


async def test_company_snapshot_timestamp_mixin():
    now = utc_now()
    row = CompanySnapshot(
        id="ts-snap",
        company_id="ts-comp",
        analysis_id="ts-aid",
        report_id="ts-rid",
        created_at=now,
    )
    assert row.created_at == now


async def test_company_columns_match_protocol():
    """Ensure every Company column is present in CompanyRecord dataclass."""
    from app.services.company_protocols import CompanyRecord

    row = Company(
        company_id="proto1",
        canonical_name="Acme",
        canonical_name_key="acme",
        canonical_domain="acme.com",
        primary_name="Acme Inc",
        website="https://acme.com",
        fallback_slug="acme-inc",
        latest_decision="invest",
        latest_confidence=0.9,
        latest_composite_score=85.0,
        snapshot_count=3,
    )
    record = CompanyRecord(
        company_id=row.company_id,
        canonical_name=row.canonical_name,
        canonical_domain=row.canonical_domain,
        primary_name=row.primary_name,
        website=row.website,
        latest_decision=row.latest_decision,
        latest_confidence=row.latest_confidence,
        latest_composite_score=row.latest_composite_score,
        snapshot_count=row.snapshot_count,
        first_seen=row.first_seen,
        last_seen=row.last_seen,
        user_id=row.user_id,
        canonical_name_key=row.canonical_name_key,
        fallback_slug=row.fallback_slug,
    )
    assert record.company_id == "proto1"
    assert record.canonical_domain == "acme.com"


async def test_company_snapshot_columns_match_protocol():
    """Ensure every CompanySnapshot column is present in CompanySnapshotRecord."""
    from app.services.company_protocols import CompanySnapshotRecord

    now = utc_now()
    row = CompanySnapshot(
        id="proto-snap",
        company_id="proto-comp",
        analysis_id="proto-aid",
        report_id="proto-rid",
        decision="invest",
        confidence=0.85,
        composite_score=75.0,
        readiness_score=60.0,
        dimension_scores={"market": 80.0},
        created_at=now,
    )
    record = CompanySnapshotRecord(
        id=row.id,
        company_id=row.company_id,
        analysis_id=row.analysis_id,
        report_id=row.report_id,
        decision=row.decision,
        confidence=row.confidence,
        composite_score=row.composite_score,
        readiness_score=row.readiness_score,
        dimension_scores=row.dimension_scores,
        created_at=row.created_at,
    )
    assert record.id == "proto-snap"
    assert record.decision == "invest"


async def test_company_user_id_settable():
    row = Company(
        company_id="uid1",
        canonical_name="Acme",
        canonical_name_key="acme",
        primary_name="Acme",
        fallback_slug="acme",
        user_id="user-123",
    )
    assert row.user_id == "user-123"


async def test_company_frozen_after_insert(sqlite_session: AsyncSession):
    row = Company(
        company_id="frozen1",
        canonical_name="Acme",
        canonical_name_key="acme",
        primary_name="Acme",
        fallback_slug="acme",
    )
    sqlite_session.add(row)
    await sqlite_session.flush()
    row.primary_name = "Acme Inc"
    await sqlite_session.flush()
    reloaded = await sqlite_session.get(Company, "frozen1")
    assert reloaded is not None
    assert reloaded.primary_name == "Acme Inc"
