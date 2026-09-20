"""Tests for the company registry backfill (roadmap #6).

Covers the pure input mapping, the application-side async service
(``CompanyBackfillService``), the sync migration path (``backfill_sync``),
and the ``0006`` migration wiring + tracking-table lifecycle.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.models import Company, CompanySnapshot
from app.models.analysis import AnalysisReport, AnalysisRequest
from app.services.companies import CompanyIdentityResolver
from app.services.company_backfill import (
    CompanyBackfillService,
    backfill_sync,
    build_backfill_inputs,
)

NOW = datetime(2025, 5, 1, 12, 0, tzinfo=UTC)
MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "0006_company_registry_backfill.py"
)

_FULL_REPORT = {
    "investment_decision": {"category": "invest", "composite_score": 76.0},
    "overall_score": 74.0,
}


class _Request:
    startup_name = "Acme Corp"
    website = "https://acme.example.com"
    created_at = NOW
    user_id = "u1"


def _report(**overrides):
    base = {
        "id": "report-1",
        "request_id": "req-1",
        "startup_name": "Acme Corp",
        "venture_score": 70.0,
        "market_score": 60.0,
        "founder_score": 80.0,
        "traction_score": 50.0,
        "recommendations": [],
        "confidence": 0.8,
        "engine_version": "0.12.1",
        "processing_time_ms": 10.0,
        "full_report": dict(_FULL_REPORT),
        "created_at": NOW,
    }
    base.update(overrides)
    return base


def _identity():
    return CompanyIdentityResolver().resolve("Acme Corp", "https://acme.example.com")


# ---------------------------------------------------------------------------
# Pure input mapping
# ---------------------------------------------------------------------------


def test_build_backfill_inputs_full_report_mapping():
    inputs = build_backfill_inputs(AnalysisReport(**_report()), _Request())
    assert inputs.startup_name == "Acme Corp"
    assert inputs.website == "https://acme.example.com"
    assert inputs.decision == "invest"
    assert inputs.composite_score == 76.0
    assert inputs.confidence == 0.8
    assert inputs.readiness_score is None
    assert inputs.dimension_scores == {
        "venture": 70.0,
        "market": 60.0,
        "founder": 80.0,
        "traction": 50.0,
    }
    assert inputs.created_at == NOW


def test_build_backfill_inputs_falls_back_to_max_score():
    inputs = build_backfill_inputs(AnalysisReport(**_report(full_report={})), _Request())
    assert inputs.decision is None
    assert inputs.composite_score == 80.0


def test_build_backfill_inputs_no_website():
    request = _Request()
    request.website = ""
    inputs = build_backfill_inputs(AnalysisReport(**_report()), request)
    assert inputs.website is None


# ---------------------------------------------------------------------------
# Async application service
# ---------------------------------------------------------------------------


def _seed_report(session, *, report_id="report-1", request_id="req-1"):
    session.add(
        AnalysisRequest(
            id=request_id,
            user_id="u1",
            startup_name="Acme Corp",
            website="https://acme.example.com",
            description="Backfill test description",
            created_at=NOW,
        )
    )
    session.add(AnalysisReport(**_report(id=report_id, request_id=request_id)))


async def test_backfill_all_creates_company_and_snapshot(sqlite_session):
    async with sqlite_session.begin():
        _seed_report(sqlite_session)

    summary = await CompanyBackfillService().backfill_all(sqlite_session)

    assert summary.reports_scanned == 1
    assert summary.companies_created == 1
    assert summary.companies_adopted == 0
    assert summary.snapshots_created == 1
    assert summary.snapshots_skipped == 0
    assert summary.created_company_ids == [_identity().company_id]

    company = (await sqlite_session.execute(sa.select(Company))).scalars().one()
    assert company.company_id == _identity().company_id
    assert company.canonical_domain == _identity().canonical_domain
    assert company.snapshot_count == 1
    assert company.latest_composite_score == 76.0
    assert company.latest_decision == "invest"

    snapshot = (
        (await sqlite_session.execute(sa.select(CompanySnapshot))).scalars().one()
    )
    assert snapshot.analysis_id == "report-1"
    assert snapshot.report_id == "report-1"
    assert snapshot.composite_score == 76.0
    assert snapshot.dimension_scores["founder"] == 80.0


async def test_backfill_all_is_idempotent(sqlite_session):
    async with sqlite_session.begin():
        _seed_report(sqlite_session)

    first = await CompanyBackfillService().backfill_all(sqlite_session)
    assert first.snapshots_created == 1

    second = await CompanyBackfillService().backfill_all(sqlite_session)
    assert second.snapshots_created == 0
    assert second.snapshots_skipped == 1
    assert second.companies_adopted == 1
    assert second.companies_created == 0
    assert second.created_company_ids == []

    company = (await sqlite_session.execute(sa.select(Company))).scalars().one()
    assert company.snapshot_count == 1
    snapshots = (await sqlite_session.execute(sa.select(CompanySnapshot))).scalars().all()
    assert len(snapshots) == 1


async def test_backfill_all_two_reports_same_company(sqlite_session):
    async with sqlite_session.begin():
        _seed_report(sqlite_session, report_id="report-1", request_id="req-1")
        _seed_report(sqlite_session, report_id="report-2", request_id="req-2")

    summary = await CompanyBackfillService().backfill_all(sqlite_session)

    assert summary.companies_created == 1
    assert summary.snapshots_created == 2
    company = (await sqlite_session.execute(sa.select(Company))).scalars().one()
    assert company.snapshot_count == 2
    snapshots = (await sqlite_session.execute(sa.select(CompanySnapshot))).scalars().all()
    assert len(snapshots) == 2


# ---------------------------------------------------------------------------
# Sync migration path
# ---------------------------------------------------------------------------


def _sync_engine(tmp_path):
    from sqlalchemy import create_engine

    from app.db.session import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'backfill.db'}")
    Base.metadata.create_all(engine)
    return engine


def _seed_sync(engine):
    with Session(engine, expire_on_commit=False) as session:
        _seed_report(session)
        session.commit()


def test_backfill_sync_creates_rows_and_tracks_company(tmp_path):
    engine = _sync_engine(tmp_path)
    _seed_sync(engine)

    with engine.begin() as conn:
        summary = backfill_sync(conn)
        assert summary.reports_scanned == 1
        assert summary.companies_created == 1
        assert summary.snapshots_created == 1
        assert summary.created_company_ids == [_identity().company_id]

    with Session(engine) as session:
        company = session.execute(sa.select(Company)).scalars().one()
        assert company.company_id == _identity().company_id
        assert company.snapshot_count == 1
        snapshot = session.execute(sa.select(CompanySnapshot)).scalars().one()
        assert snapshot.analysis_id == "report-1"


def test_backfill_sync_is_idempotent(tmp_path):
    engine = _sync_engine(tmp_path)
    _seed_sync(engine)

    with engine.begin() as conn:
        first = backfill_sync(conn)
        assert first.snapshots_created == 1
    with engine.begin() as conn:
        second = backfill_sync(conn)
        assert second.snapshots_created == 0
        assert second.snapshots_skipped == 1
        assert second.companies_created == 0
        assert second.companies_adopted == 1

    with Session(engine) as session:
        company = session.execute(sa.select(Company)).scalars().one()
        assert company.snapshot_count == 1
        assert len(session.execute(sa.select(CompanySnapshot)).scalars().all()) == 1


# ---------------------------------------------------------------------------
# Migration module wiring + end-to-end upgrade/downgrade
# ---------------------------------------------------------------------------


def _load_migration():
    spec = importlib.util.spec_from_file_location("migration_0006_test", MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _tracking_table():
    return sa.Table(
        "company_registry_backfill",
        sa.MetaData(),
        sa.Column("company_id", sa.String(64), primary_key=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True)),
    )


def test_migration_revision_wiring():
    module = _load_migration()
    assert module.revision == "0006"
    assert module.down_revision == "0005"
    assert module.branch_labels is None
    assert module.depends_on is None


def test_migration_upgrade_tracks_and_backfills(tmp_path):
    import alembic.op as op_module

    module = _load_migration()
    engine = _sync_engine(tmp_path)
    _seed_sync(engine)

    with engine.begin() as conn:
        tracking = _tracking_table()
        tracking.create(conn)

        saved_bind = op_module.get_bind
        saved_create = op_module.create_table
        calls: list[str] = []
        op_module.get_bind = lambda: conn
        op_module.create_table = lambda name, *cols, **kw: calls.append(name)
        try:
            module.upgrade()
        finally:
            op_module.get_bind = saved_bind
            op_module.create_table = saved_create

        assert module._TRACKING_TABLE in calls
        created_ids = conn.execute(sa.select(tracking.c.company_id)).scalars().all()
        assert created_ids == [_identity().company_id]

    with Session(engine) as session:
        company = session.execute(sa.select(Company)).scalars().one()
        assert company.snapshot_count == 1


def test_migration_downgrade_removes_backfilled_rows(tmp_path):
    import alembic.op as op_module

    module = _load_migration()
    engine = _sync_engine(tmp_path)
    _seed_sync(engine)

    with engine.begin() as conn:
        _tracking_table().create(conn)
        saved_bind = op_module.get_bind
        saved_create = op_module.create_table
        op_module.get_bind = lambda: conn
        op_module.create_table = lambda name, *cols, **kw: None
        try:
            module.upgrade()
        finally:
            op_module.get_bind = saved_bind
            op_module.create_table = saved_create

    with Session(engine) as session:
        assert len(session.execute(sa.select(Company)).scalars().all()) == 1

    with engine.begin() as conn:
        saved_bind = op_module.get_bind
        saved_drop = op_module.drop_table
        dropped: list[str] = []
        op_module.get_bind = lambda: conn
        op_module.drop_table = lambda name, **kw: dropped.append(name)
        try:
            module.downgrade()
        finally:
            op_module.get_bind = saved_bind
            op_module.drop_table = saved_drop

        assert module._TRACKING_TABLE in dropped

    with Session(engine) as session:
        assert len(session.execute(sa.select(Company)).scalars().all()) == 0
        assert len(session.execute(sa.select(CompanySnapshot)).scalars().all()) == 0
