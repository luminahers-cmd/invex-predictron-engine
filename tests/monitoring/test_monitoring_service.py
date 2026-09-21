"""Service tests for the Phase 6 monitoring service.

Covers the operator-facing ``repository_snapshot`` (scope_all — counts every
company regardless of ownership) and the namespace scoping of the live
projections, which the HTTP suite does not exercise directly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.services.companies import CompanyIdentityResolver
from app.services.company_postgres import PostgresCompanyStore
from app.services.company_protocols import CompanySnapshotPayload
from app.services.monitoring import MonitoringService


async def _seed_company(session, *, name: str, user_id: str | None = None) -> str:
    identity = CompanyIdentityResolver().resolve(name, None)
    record = await PostgresCompanyStore().upsert_company(session, identity, user_id=user_id)
    return record.company_id


async def _seed_snapshot(session, *, company_id: str):
    from uuid import uuid4

    payload = CompanySnapshotPayload(
        company_id=company_id,
        analysis_id=f"analysis-{uuid4().hex[:24]}",
        report_id=f"report-{uuid4().hex[:24]}",
        decision="invest",
        confidence=0.8,
        composite_score=70.0,
        readiness_score=60.0,
        dimension_scores={"market": 80.0},
        created_at=datetime.now(UTC) - timedelta(days=30),
    )
    append = await PostgresCompanyStore().append_snapshot(session, payload)
    return append.snapshot


async def _register(
    session, *, company_id: str, snapshot_id: str, user_id: str | None = None
) -> str:
    from app.services.forecasts import ForecastService

    result = await ForecastService().register(
        session, company_id=company_id, snapshot_id=snapshot_id, user_id=user_id
    )
    assert result is not None
    return result.forecast.id


async def _seed_forecast(
    session, *, company_id: str, user_id: str | None = None
) -> str:
    snapshot = await _seed_snapshot(session, company_id=company_id)
    return await _register(
        session, company_id=company_id, snapshot_id=snapshot.id, user_id=user_id
    )


@pytest.mark.asyncio
async def test_repository_snapshot_ignores_namespace(sqlite_session) -> None:
    owned = await _seed_company(sqlite_session, name="cmp-owned", user_id="u1")
    await _seed_forecast(sqlite_session, company_id=owned, user_id="u1")

    public = await _seed_company(sqlite_session, name="cmp-public")
    await _seed_forecast(sqlite_session, company_id=public)

    service = MonitoringService()
    before = await service.summary(sqlite_session, user_id="u1")
    assert before.counts["forecasts"] == 1

    repo = await service.repository_snapshot(sqlite_session)
    assert repo.scope == "repository"
    assert repo.counts["forecasts"] == 2
    assert repo.counts["companies"] == 2


@pytest.mark.asyncio
async def test_live_summary_scopes_to_user(sqlite_session) -> None:
    u1 = await _seed_company(sqlite_session, name="cmp-u1", user_id="u1")
    await _seed_forecast(sqlite_session, company_id=u1, user_id="u1")

    other = await _seed_company(sqlite_session, name="cmp-other", user_id="other")
    await _seed_forecast(sqlite_session, company_id=other, user_id="other")

    service = MonitoringService()
    scoped = await service.summary(sqlite_session, user_id="u1")
    other_view = await service.summary(sqlite_session, user_id="other")
    anon = await service.summary(sqlite_session, user_id=None)
    assert scoped.counts["forecasts"] == 1
    assert other_view.counts["forecasts"] == 1
    assert anon.counts["forecasts"] == 0
    assert scoped.scope == "user:u1"
    assert anon.scope == "anonymous"


@pytest.mark.asyncio
async def test_repository_snapshot_then_history_trends(
    sqlite_session, tmp_path
) -> None:
    from predictron_engine.monitoring.history import MonitorHistory
    from predictron_engine.monitoring.trends import compute_trend

    company = await _seed_company(sqlite_session, name="cmp-1")
    await _seed_forecast(sqlite_session, company_id=company)

    service = MonitoringService(history=MonitorHistory(tmp_path))
    first = await service.repository_snapshot(sqlite_session)
    service._history.record_snapshot(first)

    second = await service.repository_snapshot(
        sqlite_session, as_of=datetime.now(UTC) + timedelta(days=1)
    )
    service._history.record_snapshot(second)

    loaded = service._history.snapshots("repository", "daily")
    assert len(loaded) == 2
    assert loaded[0].snapshot_id != loaded[1].snapshot_id
    trend = compute_trend(loaded, "confidence.mean")
    assert trend is not None
