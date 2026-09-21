"""API tests for the Live Prediction Ledger (CIH Phase 5).

Runs the real routers against an in-memory SQLite database: ``get_db`` is
overridden to yield the fixture session. Companies/snapshots are seeded
through the real store and outcomes through the real outcome service, so the
forecast endpoints exercise their exact query paths.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.db.session import get_db
from app.main import app

SNAPSHOT_AT = datetime.now(UTC) - timedelta(days=30)
OUTCOME_AT = datetime.now(UTC) - timedelta(days=10)


@pytest.fixture
def override_get_db(sqlite_session):
    """Point every router's ``get_db`` dependency at the test session."""

    async def _override():
        yield sqlite_session

    app.dependency_overrides[get_db] = _override
    yield sqlite_session
    app.dependency_overrides.pop(get_db, None)


def _headers(user_id: str) -> dict[str, str]:
    from app.auth.jwt import create_access_token

    return {"Authorization": f"Bearer {create_access_token(subject=user_id)}"}


async def _seed_company(session, *, name: str, user_id: str | None = None) -> str:
    from app.services.companies import CompanyIdentityResolver
    from app.services.company_postgres import PostgresCompanyStore

    identity = CompanyIdentityResolver().resolve(name, None)
    record = await PostgresCompanyStore().upsert_company(session, identity, user_id=user_id)
    return record.company_id


async def _seed_snapshot(session, *, company_id: str):
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
        created_at=SNAPSHOT_AT,
    )
    append = await PostgresCompanyStore().append_snapshot(session, payload)
    return append.snapshot


def _register_payload(company_id: str, snapshot_id: str, horizon_days: int | None = None) -> dict:
    payload: dict = {"company_id": company_id, "snapshot_id": snapshot_id}
    if horizon_days is not None:
        payload["horizon_days"] = horizon_days
    return payload


async def _seed_outcome(session, *, company_id: str, user_id: str | None = None):
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
        occurred_at=OUTCOME_AT,
        source="manual",
        notes="seeded outcome",
    )
    return await CompanyOutcomeService().record(
        session, company_id, request, user_id=user_id
    )


class TestRegisterForecastApi:
    @pytest.mark.asyncio
    async def test_register_creates_forecast(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        snapshot = await _seed_snapshot(override_get_db, company_id=cid)

        resp = await client.post(
            "/api/v1/forecasts", json=_register_payload(cid, snapshot.id)
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["created"] is True
        forecast = body["forecast"]
        assert len(forecast["id"]) == 64
        assert forecast["company_id"] == cid
        assert forecast["snapshot_id"] == snapshot.id
        assert forecast["horizon_days"] == 365
        assert forecast["status"] == "active"
        assert forecast["engine_version"] == "unknown"
        assert forecast["schema_version"] == "1.0"
        assert forecast["decision"] == "invest"
        assert forecast["outcome_id"] is None

    @pytest.mark.asyncio
    async def test_register_is_idempotent(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        snapshot = await _seed_snapshot(override_get_db, company_id=cid)
        url = "/api/v1/forecasts"
        payload = _register_payload(cid, snapshot.id, horizon_days=30)

        first = await client.post(url, json=payload)
        second = await client.post(url, json=payload)
        assert first.status_code == 201
        assert second.status_code == 201
        assert first.json()["created"] is True
        assert second.json()["created"] is False
        assert second.json()["forecast"]["id"] == first.json()["forecast"]["id"]
        assert second.json()["forecast"]["horizon_days"] == 30

    @pytest.mark.asyncio
    async def test_unknown_company_404(self, client, override_get_db) -> None:
        resp = await client.post(
            "/api/v1/forecasts",
            json={"company_id": "nope", "snapshot_id": "s"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_out_of_scope_company_404(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-other", user_id="other")
        snapshot = await _seed_snapshot(override_get_db, company_id=cid)
        resp = await client.post(
            "/api/v1/forecasts",
            json=_register_payload(cid, snapshot.id),
            headers=_headers("u1"),
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_payload_422(self, client, override_get_db) -> None:
        resp = await client.post("/api/v1/forecasts", json={"snapshot_id": "s"})
        assert resp.status_code == 422


class TestListAndDetailApi:
    @pytest.mark.asyncio
    async def test_list_forecasts(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        snapshot = await _seed_snapshot(override_get_db, company_id=cid)
        await client.post(
            "/api/v1/forecasts", json=_register_payload(cid, snapshot.id)
        )

        resp = await client.get("/api/v1/forecasts")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["forecasts"][0]["company_id"] == cid

    @pytest.mark.asyncio
    async def test_list_filters_by_status(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        snapshot = await _seed_snapshot(override_get_db, company_id=cid)
        await client.post(
            "/api/v1/forecasts", json=_register_payload(cid, snapshot.id)
        )
        active = await client.get("/api/v1/forecasts", params={"status": "active"})
        resolved = await client.get("/api/v1/forecasts", params={"status": "resolved"})
        assert active.status_code == 200
        assert active.json()["total"] == 1
        assert resolved.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_list_scoped_to_namespace(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1", user_id="u1")
        snapshot = await _seed_snapshot(override_get_db, company_id=cid)
        await client.post(
            "/api/v1/forecasts",
            json=_register_payload(cid, snapshot.id),
            headers=_headers("u1"),
        )
        other = await _seed_company(override_get_db, name="cmp-2", user_id="other")
        other_snapshot = await _seed_snapshot(override_get_db, company_id=other)
        await client.post(
            "/api/v1/forecasts",
            json=_register_payload(other, other_snapshot.id),
            headers=_headers("other"),
        )

        resp = await client.get("/api/v1/forecasts", headers=_headers("u1"))
        assert resp.json()["total"] == 1
        assert resp.json()["forecasts"][0]["company_id"] == cid

    @pytest.mark.asyncio
    async def test_detail_returns_prediction_and_events(
        self, client, override_get_db
    ) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        snapshot = await _seed_snapshot(override_get_db, company_id=cid)
        registered = await client.post(
            "/api/v1/forecasts", json=_register_payload(cid, snapshot.id)
        )
        forecast_id = registered.json()["forecast"]["id"]

        resp = await client.get(f"/api/v1/forecasts/{forecast_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == forecast_id
        assert body["prediction"]["decision"] == "invest"
        assert body["prediction"]["composite_score"] == 70.0
        assert [e["event_type"] for e in body["events"]] == ["registered"]

    @pytest.mark.asyncio
    async def test_detail_unknown_404(self, client, override_get_db) -> None:
        resp = await client.get("/api/v1/forecasts/no-such-id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_detail_out_of_scope_404(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-other", user_id="other")
        snapshot = await _seed_snapshot(override_get_db, company_id=cid)
        registered = await client.post(
            "/api/v1/forecasts",
            json=_register_payload(cid, snapshot.id),
            headers=_headers("other"),
        )
        forecast_id = registered.json()["forecast"]["id"]
        resp = await client.get(
            f"/api/v1/forecasts/{forecast_id}", headers=_headers("u1")
        )
        assert resp.status_code == 404


class TestReconcileApi:
    @pytest.mark.asyncio
    async def test_reconcile_one_forecast(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        snapshot = await _seed_snapshot(override_get_db, company_id=cid)
        registered = await client.post(
            "/api/v1/forecasts", json=_register_payload(cid, snapshot.id)
        )
        forecast_id = registered.json()["forecast"]["id"]
        await _seed_outcome(override_get_db, company_id=cid)

        resp = await client.post(f"/api/v1/forecasts/{forecast_id}/reconcile")
        assert resp.status_code == 200
        body = resp.json()
        assert body["reconciled"] is True
        assert body["outcome_id"] is not None
        assert body["outcome_verdict"] == "success"
        assert body["forecast"]["status"] == "resolved"

    @pytest.mark.asyncio
    async def test_reconcile_unknown_forecast_404(self, client, override_get_db) -> None:
        resp = await client.post("/api/v1/forecasts/no-such-id/reconcile")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_batch_reconcile(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        snapshot = await _seed_snapshot(override_get_db, company_id=cid)
        await client.post("/api/v1/forecasts", json=_register_payload(cid, snapshot.id))
        await _seed_outcome(override_get_db, company_id=cid)

        resp = await client.post("/api/v1/forecasts/reconcile")
        assert resp.status_code == 200
        body = resp.json()
        assert body["reconciled_count"] == 1
        assert body["forecasts"][0]["status"] == "resolved"

    @pytest.mark.asyncio
    async def test_summary_route_not_shadowed_by_id(
        self, client, override_get_db
    ) -> None:
        resp = await client.get("/api/v1/forecasts/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["by_status"] == {"active": 0, "due": 0, "resolved": 0}

        batch = await client.post("/api/v1/forecasts/reconcile")
        assert batch.status_code == 200
        assert batch.json()["reconciled_count"] == 0


class TestSummaryApi:
    @pytest.mark.asyncio
    async def test_summary_counts(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        snapshot = await _seed_snapshot(override_get_db, company_id=cid)
        await client.post(
            "/api/v1/forecasts", json=_register_payload(cid, snapshot.id)
        )
        await _seed_outcome(override_get_db, company_id=cid)

        resp = await client.get("/api/v1/forecasts/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["active"] == 1
        assert body["due"] == 0
        assert body["resolved"] == 0
        assert body["outcome_linked"] == 0
        assert len(body["recent"]) == 1
