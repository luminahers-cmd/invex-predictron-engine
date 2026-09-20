"""API tests for company outcome recording/listing (CIH Phase 4).

Runs the real routers against an in-memory SQLite database: ``get_db`` is
overridden to yield the fixture session, and companies/snapshots are seeded
through the real store so the outcome and evaluation services exercise their
exact query paths.
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
    """Point every router's ``get_db`` dependency at the test session."""

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


async def _seed_snapshot(session, *, company_id: str, created_at: datetime = SNAPSHOT_AT):
    from app.services.company_postgres import PostgresCompanyStore
    from app.services.company_protocols import CompanySnapshotPayload

    payload = CompanySnapshotPayload(
        company_id=company_id,
        analysis_id=f"analysis-{created_at.date()}",
        report_id=f"report-{created_at.date()}",
        decision="invest",
        confidence=0.8,
        composite_score=70.0,
        readiness_score=60.0,
        dimension_scores={"market": 80.0},
        created_at=created_at,
    )
    return await PostgresCompanyStore().append_snapshot(session, payload)


def _success_outcome_payload() -> dict:
    return {
        "status": "fully_verified",
        "acquisition": "Acquirer Corp",
        "acquisition_price_usd": 50_000_000.0,
        "exit_type": "acquisition",
    }


class TestRecordOutcome:
    @pytest.mark.asyncio
    async def test_record_requires_auth(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        resp = await client.post(f"/api/v1/companies/{cid}/outcomes", json={})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_record_creates_outcome_and_evaluations(
        self, client, override_get_db
    ) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        await _seed_snapshot(override_get_db, company_id=cid)
        resp = await client.post(
            f"/api/v1/companies/{cid}/outcomes",
            headers=_headers("u1"),
            json={
                "outcome": _success_outcome_payload(),
                "occurred_at": OUTCOME_AT.isoformat(),
                "source": "manual",
                "notes": "verified via filing",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["created"] is True
        assert body["evaluations_created"] >= 1
        assert len(body["outcome"]["id"]) == 64
        assert body["outcome"]["verdict"] == "success"
        assert body["outcome"]["company_id"] == cid

    @pytest.mark.asyncio
    async def test_duplicate_submission_is_idempotent(
        self, client, override_get_db
    ) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        payload = {
            "outcome": _success_outcome_payload(),
            "occurred_at": OUTCOME_AT.isoformat(),
            "source": "manual",
        }
        headers = _headers("u1")
        url = f"/api/v1/companies/{cid}/outcomes"
        first = await client.post(url, headers=headers, json=payload)
        second = await client.post(url, headers=headers, json=payload)
        assert first.status_code == 201
        assert second.status_code == 201
        assert first.json()["created"] is True
        assert second.json()["created"] is False
        assert second.json()["outcome"]["id"] == first.json()["outcome"]["id"]

    @pytest.mark.asyncio
    async def test_unknown_company_404(self, client, override_get_db) -> None:
        resp = await client.post(
            "/api/v1/companies/nope/outcomes",
            headers=_headers("u1"),
            json={"outcome": _success_outcome_payload()},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_out_of_scope_company_404(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-other", user_id="other")
        resp = await client.post(
            f"/api/v1/companies/{cid}/outcomes",
            headers=_headers("u1"),
            json={"outcome": _success_outcome_payload()},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_payload_422(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        resp = await client.post(
            f"/api/v1/companies/{cid}/outcomes",
            headers=_headers("u1"),
            json={"snapshot_id": "x"},
        )
        assert resp.status_code == 422


class TestListOutcomes:
    @pytest.mark.asyncio
    async def test_list_outcomes_for_company(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-1")
        await _seed_snapshot(override_get_db, company_id=cid)
        payload = {
            "outcome": _success_outcome_payload(),
            "occurred_at": OUTCOME_AT.isoformat(),
            "source": "manual",
        }
        for _ in range(2):
            resp = await client.post(
                f"/api/v1/companies/{cid}/outcomes",
                headers=_headers("u1"),
                json=payload,
            )
            assert resp.status_code == 201

        resp = await client.get(
            f"/api/v1/companies/{cid}/outcomes", headers=_headers("u1")
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["company_id"] == cid
        assert body["total"] == 1  # duplicate submission collapses to one row

    @pytest.mark.asyncio
    async def test_list_missing_company_404(self, client, override_get_db) -> None:
        resp = await client.get("/api/v1/companies/nope/outcomes", headers=_headers("u1"))
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_scope_other_user_404(self, client, override_get_db) -> None:
        cid = await _seed_company(override_get_db, name="cmp-other", user_id="other")
        resp = await client.get(
            f"/api/v1/companies/{cid}/outcomes", headers=_headers("u1")
        )
        assert resp.status_code == 404
