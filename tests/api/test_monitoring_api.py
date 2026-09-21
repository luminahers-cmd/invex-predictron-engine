"""API tests for the Continuous Intelligence & Drift Detection (Phase 6).

Mirrors the forecasts suite: real routers against in-memory SQLite with
``get_db`` overridden.  Companies/snapshots/outcomes are seeded through the
real stores/services so the monitor endpoints exercise their exact query
paths.  History-dependent endpoints (``trends``/``drift``) run against an
isolated snapshot directory via ``MONITOR_HISTORY_DIR``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from app.db.session import get_db
from app.main import app


@pytest.fixture
def override_get_db(sqlite_session):
    """Point every router's ``get_db`` dependency at the test session."""

    async def _override():
        yield sqlite_session

    app.dependency_overrides[get_db] = _override
    yield sqlite_session
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def isolate_history(monkeypatch, tmp_path):
    """Bind ``MONITOR_HISTORY_DIR`` to a fresh temp dir for this test."""
    from app.core.config import get_settings

    monkeypatch.setenv("MONITOR_HISTORY_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def _headers(user_id: str) -> dict[str, str]:
    from app.auth.jwt import create_access_token

    return {"Authorization": f"Bearer {create_access_token(subject=user_id)}"}


async def _seed_company(session, *, name: str, user_id: str | None = None) -> str:
    from app.services.companies import CompanyIdentityResolver
    from app.services.company_postgres import PostgresCompanyStore

    identity = CompanyIdentityResolver().resolve(name, None)
    record = await PostgresCompanyStore().upsert_company(session, identity, user_id=user_id)
    return record.company_id


async def _seed_snapshot(session, *, company_id: str, created_at: datetime):
    from uuid import uuid4

    from app.services.company_postgres import PostgresCompanyStore
    from app.services.company_protocols import CompanySnapshotPayload

    payload = CompanySnapshotPayload(
        company_id=company_id,
        analysis_id=f"analysis-{uuid4().hex[:24]}",
        report_id=f"report-{uuid4().hex[:24]}",
        decision="invest",
        confidence=0.8,
        composite_score=70.0,
        readiness_score=60.0,
        dimension_scores={"market": 80.0},
        created_at=created_at,
    )
    append = await PostgresCompanyStore().append_snapshot(session, payload)
    return append.snapshot


async def _register(session, client, *, company_id: str, snapshot_id: str,
                    horizon_days: int | None = None, user_id: str | None = None) -> str:
    payload: dict = {"company_id": company_id, "snapshot_id": snapshot_id}
    if horizon_days is not None:
        payload["horizon_days"] = horizon_days
    kwargs = {}
    if user_id is not None:
        kwargs["headers"] = _headers(user_id)
    resp = await client.post("/api/v1/forecasts", json=payload, **kwargs)
    assert resp.status_code == 201, resp.text
    return resp.json()["forecast"]["id"]


async def _seed_outcome(
    session,
    *,
    company_id: str,
    occurred_at: datetime | None = None,
    user_id: str | None = None,
):
    from app.schemas.outcome import OutcomeCreateRequest
    from app.services.company_outcomes import CompanyOutcomeService
    from predictron_engine.dataset.outcomes import StartupOutcome

    request = OutcomeCreateRequest(
        outcome=StartupOutcome.model_validate(
            {
                "status": "fully_verified",
                "acquisition": "Acquirer Corp",
                "acquisition_price_usd": 50_000_000.0,
                "exit_type": "acquisition",
            }
        ),
        occurred_at=occurred_at or (datetime.now(UTC) - timedelta(days=10)),
        source="manual",
        notes="seeded outcome",
    )
    return await CompanyOutcomeService().record(
        session, company_id, request, user_id=user_id
    )


class TestMonitorSummaryApi:
    @pytest.mark.asyncio
    async def test_summary_empty_db(self, client, override_get_db) -> None:
        resp = await client.get("/api/v1/monitor/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["counts"]["forecasts"] == 0
        assert body["scope"] == "anonymous"

    @pytest.mark.asyncio
    async def test_summary_counts_and_accuracy(
        self, client, override_get_db
    ) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        snapshot = await _seed_snapshot(
            override_get_db, company_id=cid, created_at=datetime.now(UTC) - timedelta(days=30)
        )
        forecast_id = await _register(
            override_get_db, client, company_id=cid, snapshot_id=snapshot.id
        )
        await _seed_outcome(override_get_db, company_id=cid)
        reconciled = await client.post(f"/api/v1/forecasts/{forecast_id}/reconcile")
        assert reconciled.status_code == 200, reconciled.text

        resp = await client.get("/api/v1/monitor/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["counts"]["forecasts"] == 1
        assert body["counts"]["evaluations"] == 1
        assert body["counts"]["companies"] == 1
        assert body["metrics"]["metrics.accuracy"] == 1.0
        assert body["health"]["resolved"] == 1
        assert body["distributions"]["evaluation_verdict"]["counts"]["correct"] == 1

    @pytest.mark.asyncio
    async def test_summary_namespace_scoping(self, client, override_get_db) -> None:
        u1 = await _seed_company(override_get_db, name="cmp-u1", user_id="u1")
        snap1 = await _seed_snapshot(
            override_get_db, company_id=u1, created_at=datetime.now(UTC) - timedelta(days=10)
        )
        await _register(
            override_get_db, client, company_id=u1, snapshot_id=snap1.id, user_id="u1"
        )

        anon = await client.get("/api/v1/monitor/summary")
        scoped = await client.get("/api/v1/monitor/summary", headers=_headers("u1"))
        assert anon.json()["counts"]["forecasts"] == 0
        assert scoped.json()["counts"]["forecasts"] == 1
        assert scoped.json()["scope"] == "user:u1"

    @pytest.mark.asyncio
    async def test_rollup_same_shape(self, client, override_get_db) -> None:
        resp = await client.get("/api/v1/monitor/rollup")
        assert resp.status_code == 200
        assert resp.json()["counts"]["forecasts"] == 0


class TestMonitorHealthApi:
    @pytest.mark.asyncio
    async def test_health_active(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        snapshot = await _seed_snapshot(
            override_get_db, company_id=cid, created_at=datetime.now(UTC) - timedelta(days=30)
        )
        await _register(override_get_db, client, company_id=cid, snapshot_id=snapshot.id)

        resp = await client.get("/api/v1/monitor/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["distribution"] == {"active": 1}
        assert body["entries"][0]["health"] == "active"
        assert body["entries"][0]["evaluation_status"] == "pending"

    @pytest.mark.asyncio
    async def test_health_stale_and_overdue_filters(
        self, client, override_get_db
    ) -> None:
        now = datetime.now(UTC)
        overdue_cid = await _seed_company(override_get_db, name="cmp-overdue")
        overdue_snap = await _seed_snapshot(
            override_get_db, company_id=overdue_cid, created_at=now - timedelta(days=400)
        )
        await _register(override_get_db, client, company_id=overdue_cid,
                        snapshot_id=overdue_snap.id, horizon_days=90)

        stale_cid = await _seed_company(override_get_db, name="cmp-stale")
        stale_snap = await _seed_snapshot(
            override_get_db, company_id=stale_cid, created_at=now - timedelta(days=400)
        )
        await _register(override_get_db, client, company_id=stale_cid,
                        snapshot_id=stale_snap.id, horizon_days=730)

        overdue = await client.get("/api/v1/monitor/overdue")
        stale = await client.get("/api/v1/monitor/stale")
        assert overdue.status_code == 200
        assert stale.status_code == 200
        assert [e["health"] for e in overdue.json()["entries"]] == ["overdue"]
        assert [e["health"] for e in stale.json()["entries"]] == ["stale"]
        assert overdue.json()["distribution"] == {"overdue": 1, "stale": 1}
        assert stale.json()["distribution"] == {"overdue": 1, "stale": 1}


class TestMonitorReanalysisApi:
    @pytest.mark.asyncio
    async def test_reanalysis_evidence_changed_after_new_outcome(
        self, client, override_get_db
    ) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        snapshot = await _seed_snapshot(
            override_get_db, company_id=cid, created_at=datetime.now(UTC) - timedelta(days=30)
        )
        forecast_id = await _register(
            override_get_db, client, company_id=cid, snapshot_id=snapshot.id
        )

        before = await client.get("/api/v1/monitor/reanalysis")
        assert before.json()["recommendations"] == []

        await _seed_outcome(override_get_db, company_id=cid)
        reconciled = await client.post(f"/api/v1/forecasts/{forecast_id}/reconcile")
        assert reconciled.status_code == 200, reconciled.text

        concluded = await client.get("/api/v1/monitor/reanalysis")
        assert concluded.json()["recommendations"] == []

        await _seed_outcome(
            override_get_db,
            company_id=cid,
            occurred_at=datetime.now(UTC) + timedelta(days=2),
        )

        after = await client.get("/api/v1/monitor/reanalysis")
        assert after.status_code == 200
        body = after.json()
        assert len(body["recommendations"]) == 1
        assert "evidence_changed" in body["recommendations"][0]["reasons"]


class TestMonitorTrendsApi:
    @pytest.mark.asyncio
    async def test_trends_empty_history(self, client, isolate_history) -> None:
        resp = await client.get("/api/v1/monitor/trends")
        assert resp.status_code == 200
        assert resp.json()["trends"] == []

    @pytest.mark.asyncio
    async def test_trends_over_recorded_snapshots(
        self, client, isolate_history
    ) -> None:
        from predictron_engine.monitoring.history import MonitorHistory
        from tests.monitoring.test_monitoring_history import snapshot as build_test_snap

        history = MonitorHistory(isolate_history)
        history.record_snapshot(build_test_snap(date(2026, 9, 21), 0.5, "trend-21"))
        history.record_snapshot(build_test_snap(date(2026, 9, 22), 0.9, "trend-22"))

        resp = await client.get("/api/v1/monitor/trends", params={"period": "daily"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["as_of"] == "2026-09-22"
        accuracy = next(t for t in body["trends"] if t["metric"] == "metrics.accuracy")
        assert accuracy["direction"] == "up"
        assert [p["value"] for p in accuracy["series"]] == [0.5, 0.9]


class TestMonitorDriftApi:
    @pytest.mark.asyncio
    async def test_drift_404_when_no_history(self, client, isolate_history) -> None:
        resp = await client.get("/api/v1/monitor/drift")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_drift_default_latest_two(
        self, client, isolate_history
    ) -> None:
        from predictron_engine.monitoring.history import MonitorHistory
        from tests.monitoring.test_monitoring_history import snapshot as build_test_snap

        history = MonitorHistory(isolate_history)
        history.record_snapshot(build_test_snap(date(2026, 9, 21), 0.5, "drift-21"))
        history.record_snapshot(build_test_snap(date(2026, 9, 22), 0.9, "drift-22"))

        resp = await client.get("/api/v1/monitor/drift")
        assert resp.status_code == 200
        body = resp.json()
        assert body["baseline_id"] == "drift-21"
        assert body["comparison_id"] == "drift-22"
        assert len(body["signals"]) >= 3

    @pytest.mark.asyncio
    async def test_drift_explicit_ids(self, client, isolate_history) -> None:
        from predictron_engine.monitoring.history import MonitorHistory
        from tests.monitoring.test_monitoring_history import snapshot as build_test_snap

        history = MonitorHistory(isolate_history)
        history.record_snapshot(build_test_snap(date(2026, 9, 21), 0.5, "drift-21"))
        history.record_snapshot(build_test_snap(date(2026, 9, 22), 0.9, "drift-22"))
        history.record_snapshot(build_test_snap(date(2026, 9, 23), 0.6, "drift-23"))

        resp = await client.get(
            "/api/v1/monitor/drift",
            params={"before_id": "drift-21", "after_id": "drift-23"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["baseline_id"] == "drift-21"
        assert body["comparison_id"] == "drift-23"

    @pytest.mark.asyncio
    async def test_drift_unknown_id_404(self, client, isolate_history) -> None:
        resp = await client.get(
            "/api/v1/monitor/drift", params={"before_id": "missing"}
        )
        assert resp.status_code == 404
