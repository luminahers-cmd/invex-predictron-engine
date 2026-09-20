"""API tests for company/global evaluation views (CIH Phase 4).

Runs the real routers against an in-memory SQLite database via a ``get_db``
override: outrich of a company (outcome recording, performance) and the
cross-company calibration/summary views.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.auth.jwt import create_access_token
from app.db.session import get_db
from app.main import app

SNAPSHOT_AT = datetime(2025, 1, 1, tzinfo=UTC)
OUTCOME_AT = datetime(2025, 6, 1, tzinfo=UTC)


def _headers(user_id: str | None = None) -> dict:
    if user_id is None:
        return {}
    return {"Authorization": f"Bearer {create_access_token(subject=user_id)}"}


@pytest.fixture
def override_get_db(sqlite_session):
    async def _override():
        yield sqlite_session

    app.dependency_overrides[get_db] = _override
    yield sqlite_session
    app.dependency_overrides.pop(get_db, None)


async def _seed_company(session, *, name: str, user_id: str | None = "u1") -> str:
    from app.services.companies import CompanyIdentityResolver
    from app.services.company_postgres import PostgresCompanyStore

    identity = CompanyIdentityResolver().resolve(name, None)
    record = await PostgresCompanyStore().upsert_company(session, identity, user_id=user_id)
    return record.company_id


async def _seed_invest_snapshot(session, *, company_id: str):
    from app.services.company_postgres import PostgresCompanyStore
    from app.services.company_protocols import CompanySnapshotPayload

    payload = CompanySnapshotPayload(
        company_id=company_id,
        analysis_id="analysis-a",
        report_id="report-a",
        decision="invest",
        confidence=0.8,
        composite_score=70.0,
        readiness_score=60.0,
        dimension_scores={"market": 80.0},
        created_at=SNAPSHOT_AT,
    )
    return await PostgresCompanyStore().append_snapshot(session, payload)


async def _post_success_outcome(client, *, company_id: str, user_id: str = "u1"):
    return await client.post(
        f"/api/v1/companies/{company_id}/outcomes",
        headers=_headers(user_id),
        json={
            "outcome": {
                "status": "fully_verified",
                "acquisition": "Acquirer Corp",
                "acquisition_price_usd": 50_000_000.0,
                "exit_type": "acquisition",
            },
            "occurred_at": OUTCOME_AT.isoformat(),
            "source": "manual",
        },
    )


class TestCompanyPerformance:
    @pytest.mark.asyncio
    async def test_performance_empty_before_outcomes(
        self, client, override_get_db
    ) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        await _seed_invest_snapshot(override_get_db, company_id=cid)
        resp = await client.get(
            f"/api/v1/companies/{cid}/performance", headers=_headers("u1")
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["outcome_count"] == 0
        assert body["evaluation_count"] == 0
        assert body["history"] == []

    @pytest.mark.asyncio
    async def test_performance_after_outcome(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        await _seed_invest_snapshot(override_get_db, company_id=cid)
        posted = await _post_success_outcome(client, company_id=cid)
        assert posted.status_code == 201
        assert posted.json()["evaluations_created"] >= 1

        resp = await client.get(
            f"/api/v1/companies/{cid}/performance", headers=_headers("u1")
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["outcome_count"] == 1
        assert body["evaluation_count"] == 1
        assert body["metrics"]["scoreable"] >= 1
        assert body["calibration"]["total_samples"] >= 1
        entry = body["history"][0]
        assert entry["verdict"] == "correct"
        assert entry["decision_match"] is True
        assert body["alignment"]["strong_match"] == 1

    @pytest.mark.asyncio
    async def test_performance_missing_company_404(self, client, override_get_db) -> None:
        resp = await client.get(
            "/api/v1/companies/nope/performance", headers=_headers("u1")
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_performance_scope_other_user_404(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-other", user_id="other")
        resp = await client.get(
            f"/api/v1/companies/{cid}/performance", headers=_headers("u1")
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_performance_anonymous_scoped(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-open", user_id=None)
        resp = await client.get(f"/api/v1/companies/{cid}/performance")
        assert resp.status_code == 200


class TestGlobalViews:
    @pytest.mark.asyncio
    async def test_summary_scopes_to_authenticated_user(
        self, client, override_get_db
    ) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        await _seed_invest_snapshot(override_get_db, company_id=cid)
        await _post_success_outcome(client, company_id=cid)

        resp = await client.get("/api/v1/evaluation/summary", headers=_headers("u1"))
        assert resp.status_code == 200
        body = resp.json()
        assert body["company_count"] == 1
        assert body["outcome_count"] == 1
        assert body["evaluation_count"] == 1
        assert body["by_company"][0]["company_id"] == cid

        resp_other = await client.get(
            "/api/v1/evaluation/summary", headers=_headers("u2")
        )
        assert resp_other.status_code == 200
        assert resp_other.json()["company_count"] == 0

    @pytest.mark.asyncio
    async def test_calibration_aggregates_scoreable_outcomes(
        self, client, override_get_db
    ) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        await _seed_invest_snapshot(override_get_db, company_id=cid)
        await _post_success_outcome(client, company_id=cid)

        resp = await client.get("/api/v1/evaluation/calibration", headers=_headers("u1"))
        assert resp.status_code == 200
        body = resp.json()
        assert body["company_count"] == 1
        assert body["evaluation_count"] == 1
        assert body["calibration"]["total_samples"] >= 1
