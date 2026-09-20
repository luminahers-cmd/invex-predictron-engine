"""Tests for CompanyIngestService — resolve → upsert → snapshot append.

Exercises the service against the real PostgresCompanyStore on in-memory
SQLite, plus the pure build_snapshot_inputs mapping via report fixtures.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.companies import (
    CompanyIdentityResolver,
    CompanyIngestService,
    IngestResult,
)
from app.services.company_postgres import PostgresCompanyStore

pytestmark = pytest.mark.asyncio


async def test_ingest_creates_company_and_snapshot(sqlite_session: AsyncSession):
    svc = CompanyIngestService()
    result = await svc.ingest_analysis(
        sqlite_session,
        startup_name="Acme Inc",
        website="https://acme.com",
        analysis_id="a1",
        report_id="r1",
        decision="invest",
        confidence=0.9,
        composite_score=80.0,
        readiness_score=70.0,
        user_id="u1",
    )
    assert isinstance(result, IngestResult)
    assert result.created_snapshot is True
    assert result.company_id
    assert result.snapshot_id


async def test_ingest_is_idempotent_for_duplicate_analysis(
    sqlite_session: AsyncSession,
):
    svc = CompanyIngestService()
    kwargs = dict(
        startup_name="Acme Inc",
        website="https://acme.com",
        analysis_id="dup-1",
        report_id="r1",
        user_id="u1",
    )
    first = await svc.ingest_analysis(sqlite_session, **kwargs)
    second = await svc.ingest_analysis(sqlite_session, **kwargs)
    assert first.created_snapshot is True
    assert second.created_snapshot is False
    assert second.snapshot_id == first.snapshot_id
    assert second.company_id == first.company_id


async def test_ingest_deterministic_company_id(sqlite_session: AsyncSession):
    svc = CompanyIngestService()
    a = await svc.ingest_analysis(
        sqlite_session,
        startup_name="Acme Inc",
        website="https://acme.com",
        analysis_id="a1",
        report_id="r1",
        user_id="u1",
    )
    b = await svc.ingest_analysis(
        sqlite_session,
        startup_name="Acme Incorporated",
        website="https://acme.com",
        analysis_id="a2",
        report_id="r2",
        user_id="u1",
    )
    assert a.company_id == b.company_id


async def test_ingest_merges_history_for_same_company(
    sqlite_session: AsyncSession,
):
    svc = CompanyIngestService()
    await svc.ingest_analysis(
        sqlite_session,
        startup_name="Acme Inc",
        website="https://acme.com",
        analysis_id="a1",
        report_id="r1",
        user_id="u1",
    )
    await svc.ingest_analysis(
        sqlite_session,
        startup_name="Acme Inc",
        website="https://acme.com",
        analysis_id="a2",
        report_id="r2",
        decision="pass",
        confidence=0.4,
        composite_score=40.0,
        user_id="u1",
    )
    store = PostgresCompanyStore()
    comp = await store.list_companies(sqlite_session, user_id="u1")
    assert comp.total == 1
    page = await store.list_snapshots(
        sqlite_session, comp.companies[0].company_id, user_id="u1"
    )
    assert page.total == 2


async def test_ingest_latest_metrics_reflect_last_analysis(
    sqlite_session: AsyncSession,
):
    svc = CompanyIngestService()
    first = await svc.ingest_analysis(
        sqlite_session,
        startup_name="Acme Inc",
        website="https://acme.com",
        analysis_id="a1",
        report_id="r1",
        decision="invest",
        confidence=0.9,
        composite_score=80.0,
        user_id="u1",
    )
    await svc.ingest_analysis(
        sqlite_session,
        startup_name="Acme Inc",
        website="https://acme.com",
        analysis_id="a2",
        report_id="r2",
        decision="pass",
        confidence=0.2,
        composite_score=30.0,
        user_id="u1",
    )
    store = PostgresCompanyStore()
    comp = await store.get_company(
        sqlite_session, first.company_id, user_id="u1"
    )
    assert comp is not None
    assert comp.latest_decision == "pass"
    assert comp.latest_composite_score == 30.0


async def test_ingest_without_website_uses_name_identity(
    sqlite_session: AsyncSession,
):
    svc = CompanyIngestService()
    a = await svc.ingest_analysis(
        sqlite_session,
        startup_name="Acme Inc",
        website=None,
        analysis_id="a1",
        report_id="r1",
        user_id="u1",
    )
    b = await svc.ingest_analysis(
        sqlite_session,
        startup_name="Acme Inc",
        website=None,
        analysis_id="a2",
        report_id="r2",
        user_id="u1",
    )
    assert a.company_id == b.company_id


async def test_ingest_different_names_different_companies(
    sqlite_session: AsyncSession,
):
    svc = CompanyIngestService()
    a = await svc.ingest_analysis(
        sqlite_session,
        startup_name="Acme Inc",
        website=None,
        analysis_id="a1",
        report_id="r1",
        user_id="u1",
    )
    b = await svc.ingest_analysis(
        sqlite_session,
        startup_name="Betty Inc",
        website=None,
        analysis_id="a2",
        report_id="r2",
        user_id="u1",
    )
    assert a.company_id != b.company_id


async def test_ingest_owns_company_for_first_user(sqlite_session: AsyncSession):
    """Company identity is global; ownership is stable on first insert.

    A second user re-ingesting the same identity adopts the existing row
    (same deterministic company_id) rather than creating a duplicate, and the
    row remains visible only to its owner's namespace.
    """
    svc = CompanyIngestService()
    u1 = await svc.ingest_analysis(
        sqlite_session,
        startup_name="Acme Inc",
        website="https://acme.com",
        analysis_id="aid-u1",
        report_id="rid-u1",
        user_id="u1",
    )
    u2 = await svc.ingest_analysis(
        sqlite_session,
        startup_name="Acme Inc",
        website="https://acme.com",
        analysis_id="aid-u2",
        report_id="rid-u2",
        user_id="u2",
    )
    assert u2.company_id == u1.company_id
    store = PostgresCompanyStore()
    page_u1 = await store.list_companies(sqlite_session, user_id="u1")
    page_u2 = await store.list_companies(sqlite_session, user_id="u2")
    assert page_u1.total == 1
    assert page_u2.total == 0
    snapshots = await store.list_snapshots(
        sqlite_session, u1.company_id, user_id="u1"
    )
    assert snapshots.total == 2


async def test_ingest_uses_created_at(sqlite_session: AsyncSession):
    svc = CompanyIngestService()
    ts = datetime(2025, 3, 1, tzinfo=UTC)
    result = await svc.ingest_analysis(
        sqlite_session,
        startup_name="Acme Inc",
        website="https://acme.com",
        analysis_id="a-ts",
        report_id="r-ts",
        created_at=ts,
        user_id="u1",
    )
    store = PostgresCompanyStore()
    page = await store.list_snapshots(
        sqlite_session, result.company_id, user_id="u1"
    )
    assert page.snapshots[0].created_at == ts.replace(tzinfo=None)


async def test_ingest_no_decision_fields_ok(sqlite_session: AsyncSession):
    svc = CompanyIngestService()
    result = await svc.ingest_analysis(
        sqlite_session,
        startup_name="Acme Inc",
        website="https://acme.com",
        analysis_id="a-null",
        report_id="r-null",
        user_id="u1",
    )
    assert result.created_snapshot is True


async def test_ingest_empty_dimensions_stored_as_empty(
    sqlite_session: AsyncSession,
):
    svc = CompanyIngestService()
    result = await svc.ingest_analysis(
        sqlite_session,
        startup_name="Acme Inc",
        website="https://acme.com",
        analysis_id="a-dim",
        report_id="r-dim",
        user_id="u1",
    )
    store = PostgresCompanyStore()
    page = await store.list_snapshots(
        sqlite_session, result.company_id, user_id="u1"
    )
    assert page.snapshots[0].dimension_scores == {}


# ── ingest_after_persist ────────────────────────────────────────────────


async def test_ingest_after_persist_resolves_report_id(sqlite_session: AsyncSession):
    from app.models import AnalysisReport, AnalysisRequest

    req = AnalysisRequest(
        id="req-1",
        user_id="u1",
        startup_name="Acme Inc",
        website="https://acme.com",
        description="d",
    )
    rep = AnalysisReport(
        id="rep-1",
        request_id="req-1",
        startup_name="Acme Inc",
        venture_score=70.0,
        market_score=80.0,
        founder_score=60.0,
        traction_score=50.0,
        recommendations=[],
        confidence=0.8,
        full_report={},
    )
    sqlite_session.add_all([req, rep])
    await sqlite_session.flush()

    from app.services.companies import AnalysisSnapshotInputs

    svc = CompanyIngestService()
    inputs = AnalysisSnapshotInputs(
        startup_name="Acme Inc",
        website="https://acme.com",
        decision="invest",
        confidence=0.8,
        composite_score=75.0,
        readiness_score=60.0,
    )
    result = await svc.ingest_after_persist(
        sqlite_session, analysis=req, inputs=inputs, user_id="u1"
    )
    assert result is not None
    assert result.company_id
    assert result.created_snapshot is True


async def test_ingest_after_persist_missing_report_returns_none(
    sqlite_session: AsyncSession,
):
    from app.models import AnalysisRequest
    from app.services.companies import AnalysisSnapshotInputs

    req = AnalysisRequest(
        id="req-ghost",
        user_id="u1",
        startup_name="Acme Inc",
        website="https://acme.com",
        description="d",
    )
    sqlite_session.add(req)
    await sqlite_session.flush()

    svc = CompanyIngestService()
    inputs = AnalysisSnapshotInputs(
        startup_name="Acme Inc",
        website="https://acme.com",
        decision=None,
        confidence=None,
        composite_score=None,
        readiness_score=None,
    )
    result = await svc.ingest_after_persist(
        sqlite_session, analysis=req, inputs=inputs, user_id="u1"
    )
    assert result is None


async def test_ingest_after_persist_idempotent(sqlite_session: AsyncSession):
    from app.models import AnalysisReport, AnalysisRequest
    from app.services.companies import AnalysisSnapshotInputs

    req = AnalysisRequest(
        id="req-2",
        user_id="u1",
        startup_name="Acme Inc",
        website="https://acme.com",
        description="d",
    )
    rep = AnalysisReport(
        id="rep-2",
        request_id="req-2",
        startup_name="Acme Inc",
        venture_score=70.0,
        market_score=80.0,
        founder_score=60.0,
        traction_score=50.0,
        recommendations=[],
        confidence=0.8,
        full_report={},
    )
    sqlite_session.add_all([req, rep])
    await sqlite_session.flush()

    svc = CompanyIngestService()
    inputs = AnalysisSnapshotInputs(
        startup_name="Acme Inc",
        website="https://acme.com",
        decision="invest",
        confidence=0.8,
        composite_score=75.0,
        readiness_score=60.0,
    )
    first = await svc.ingest_after_persist(
        sqlite_session, analysis=req, inputs=inputs, user_id="u1"
    )
    second = await svc.ingest_after_persist(
        sqlite_session, analysis=req, inputs=inputs, user_id="u1"
    )
    assert first is not None
    assert second is not None
    assert first.snapshot_id == second.snapshot_id
    assert first.created_snapshot is True
    assert second.created_snapshot is False


async def test_ingest_after_persist_with_custom_store_and_resolver(
    sqlite_session: AsyncSession,
):
    from app.models import AnalysisReport, AnalysisRequest
    from app.services.companies import AnalysisSnapshotInputs

    req = AnalysisRequest(
        id="req-3",
        user_id="u1",
        startup_name="Acme Inc",
        website="https://acme.com",
        description="d",
    )
    rep = AnalysisReport(
        id="rep-3",
        request_id="req-3",
        startup_name="Acme Inc",
        venture_score=70.0,
        market_score=80.0,
        founder_score=60.0,
        traction_score=50.0,
        recommendations=[],
        confidence=0.8,
        full_report={},
    )
    sqlite_session.add_all([req, rep])
    await sqlite_session.flush()

    from tests.company.conftest import MemoryCompanyStore

    svc = CompanyIngestService(
        store=MemoryCompanyStore(),
        resolver=CompanyIdentityResolver(),
    )
    inputs = AnalysisSnapshotInputs(
        startup_name="Acme Inc",
        website="https://acme.com",
        decision="invest",
        confidence=0.8,
        composite_score=75.0,
        readiness_score=60.0,
    )
    result = await svc.ingest_after_persist(
        sqlite_session, analysis=req, inputs=inputs, user_id="u1"
    )
    assert result is not None
    assert result.created_snapshot is True


async def test_service_defaults_use_postgres_store():
    from tests.company.conftest import MemoryCompanyStore

    svc = CompanyIngestService()
    assert isinstance(svc._store, PostgresCompanyStore)
    assert not isinstance(svc._store, MemoryCompanyStore)


async def test_service_with_injected_memory_store():
    from tests.company.conftest import MemoryCompanyStore

    svc = CompanyIngestService(store=MemoryCompanyStore())
    assert isinstance(svc._store, MemoryCompanyStore)
