"""API tests for GET /api/v1/companies endpoints.

The router instantiates ``app.api.companies.PostgresCompanyStore`` directly;
tests monkeypatch that symbol to a MemoryCompanyStore (a protocol test
double) so no database is required, and use real JWT auth through the same
dependency chain the rest of the API suite exercises.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.auth.jwt import create_access_token
from tests.company.conftest import MemoryCompanyStore


def _headers(user_id: str | None = None) -> dict:
    if user_id is None:
        return {}
    return {"Authorization": f"Bearer {create_access_token(subject=user_id)}"}


@pytest.fixture
def fake_store(monkeypatch):
    """Point the companies router at an in-memory store."""
    store = MemoryCompanyStore()
    monkeypatch.setattr(
        "app.api.companies.PostgresCompanyStore", lambda: store
    )
    return store


async def _seed_company(store, *, company_id, user_id=None, **kwargs):
    from app.services.companies import CompanyIdentityResolver

    resolved = CompanyIdentityResolver().resolve(company_id, None)
    r = await store.upsert_company(None, resolved, user_id=user_id, **kwargs)
    store._companies.pop(resolved.company_id, None)
    r.company_id = company_id
    store._companies[company_id] = r
    return r


async def _seed_snapshot(store, *, company_id, analysis_id, created_at):
    from app.services.company_protocols import CompanySnapshotPayload

    payload = CompanySnapshotPayload(
        company_id=company_id,
        analysis_id=analysis_id,
        report_id=f"report-{analysis_id}",
        decision="invest",
        confidence=0.8,
        composite_score=70.0,
        readiness_score=60.0,
        dimension_scores={},
        created_at=created_at,
    )
    return await store.append_snapshot(None, payload)


# ── list companies: auth gating ────────────────────────────────────────


class TestListCompanies:
    @pytest.mark.asyncio
    async def test_list_requires_auth(self, client, fake_store):
        resp = await client.get("/api/v1/companies")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_list_invalid_token(self, client, fake_store):
        resp = await client.get(
            "/api/v1/companies",
            headers={"Authorization": "Bearer bogus"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_list_empty_for_new_user(self, client, fake_store):
        resp = await client.get(
            "/api/v1/companies", headers=_headers("u1")
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["companies"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_scopes_to_authenticated_user(self, client, fake_store):
        await _seed_company(fake_store, company_id="company-1", user_id="u1")
        await _seed_company(fake_store, company_id="company-2", user_id="u2")

        resp = await client.get("/api/v1/companies", headers=_headers("u1"))
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["companies"][0]["company_id"] == "company-1"

    @pytest.mark.asyncio
    async def test_list_returns_company_fields(self, client, fake_store):
        from app.services.companies import CompanyIdentityResolver

        resolved = CompanyIdentityResolver().resolve("Acme Inc", "https://acme.com")
        r = await fake_store.upsert_company(
            None, resolved, user_id="u1",
            latest_decision="invest",
            latest_confidence=0.85,
            latest_composite_score=72.0,
        )
        r.company_id = "abc"
        r.canonical_domain = "acme.com"
        r.website = "https://acme.com"
        fake_store._companies["abc"] = r

        resp = await client.get("/api/v1/companies", headers=_headers("u1"))
        assert resp.status_code == 200
        row = resp.json()["companies"][0]
        assert row["company_id"] == "abc"
        assert row["canonical_domain"] == "acme.com"
        assert row["latest_decision"] == "invest"
        assert row["latest_confidence"] == 0.85
        assert row["snapshot_count"] == 0

    @pytest.mark.asyncio
    async def test_list_pagination_validates_offset(self, client, fake_store):
        resp = await client.get(
            "/api/v1/companies",
            headers=_headers("u1"),
            params={"offset": -1},
        )
        assert resp.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_list_pagination_validates_limit_upper(self, client, fake_store):
        resp = await client.get(
            "/api/v1/companies",
            headers=_headers("u1"),
            params={"limit": 101},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_list_pagination_validates_limit_lower(self, client, fake_store):
        resp = await client.get(
            "/api/v1/companies",
            headers=_headers("u1"),
            params={"limit": 0},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_list_snapshot_count_surfaced(self, client, fake_store):
        await _seed_company(fake_store, company_id="with-snap", user_id="u1")
        await _seed_snapshot(
            fake_store,
            company_id="with-snap",
            analysis_id="a1",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        fake_store._companies["with-snap"].snapshot_count = 1

        resp = await client.get("/api/v1/companies", headers=_headers("u1"))
        assert resp.status_code == 200
        row = resp.json()["companies"][0]
        assert row["snapshot_count"] == 1


# ── get company ────────────────────────────────────────────────────────


class TestGetCompany:
    @pytest.mark.asyncio
    async def test_missing_company_404_authenticated(self, client, fake_store):
        resp = await client.get("/api/v1/companies/nope", headers=_headers("u1"))
        assert resp.status_code == 404
        assert "nope" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_missing_company_404_anonymous(self, client, fake_store):
        resp = await client.get("/api/v1/companies/nope")
        assert resp.status_code == 404
        assert "nope" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_returns_company_detail(self, client, fake_store):
        from app.services.companies import CompanyIdentityResolver

        resolved = CompanyIdentityResolver().resolve("Acme Inc", "https://acme.com")
        r = await fake_store.upsert_company(
            None, resolved, user_id="u1",
            latest_decision="invest",
            latest_confidence=0.85,
            latest_composite_score=72.0,
        )
        r.company_id = "detail-1"
        fake_store._companies["detail-1"] = r

        resp = await client.get(
            "/api/v1/companies/detail-1", headers=_headers("u1")
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["company_id"] == "detail-1"
        assert body["latest_decision"] == "invest"
        assert body["latest_snapshot"] is None

    @pytest.mark.asyncio
    async def test_get_other_users_company_404(self, client, fake_store):
        await _seed_company(fake_store, company_id="mine-1", user_id="u1")

        resp = await client.get(
            "/api/v1/companies/mine-1", headers=_headers("u2")
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_anonymous_cannot_see_user_owned(self, client, fake_store):
        await _seed_company(fake_store, company_id="mine-2", user_id="u1")

        resp = await client.get("/api/v1/companies/mine-2")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_anonymous_can_see_null_owned(self, client, fake_store):
        await _seed_company(fake_store, company_id="open-1", user_id=None)

        resp = await client.get("/api/v1/companies/open-1")
        assert resp.status_code == 200
        assert resp.json()["company_id"] == "open-1"


# ── company history ────────────────────────────────────────────────────


class TestCompanyHistory:
    @pytest.mark.asyncio
    async def test_missing_company_404(self, client, fake_store):
        resp = await client.get(
            "/api/v1/companies/nope/history", headers=_headers("u1")
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_history_empty(self, client, fake_store):
        await _seed_company(fake_store, company_id="hist-empty", user_id="u1")

        resp = await client.get(
            "/api/v1/companies/hist-empty/history", headers=_headers("u1")
        )
        assert resp.status_code == 200
        assert resp.json()["snapshots"] == []
        assert resp.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_history_returns_snapshots(self, client, fake_store):
        await _seed_company(fake_store, company_id="hist-1", user_id="u1")
        await _seed_snapshot(
            fake_store,
            company_id="hist-1",
            analysis_id="aid-0",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        await _seed_snapshot(
            fake_store,
            company_id="hist-1",
            analysis_id="aid-1",
            created_at=datetime(2025, 2, 1, tzinfo=UTC),
        )

        resp = await client.get(
            "/api/v1/companies/hist-1/history", headers=_headers("u1")
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["company_id"] == "hist-1"
        assert body["total"] == 2
        assert body["snapshots"][0]["analysis_id"] == "aid-1"
        assert body["snapshots"][0]["decision"] == "invest"

    @pytest.mark.asyncio
    async def test_history_pagination_validated(self, client, fake_store):
        resp = await client.get(
            "/api/v1/companies/hist-1/history",
            headers=_headers("u1"),
            params={"limit": 200},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_history_scope_other_user_404(self, client, fake_store):
        await _seed_company(fake_store, company_id="hist-scope", user_id="u1")

        resp = await client.get(
            "/api/v1/companies/hist-scope/history", headers=_headers("u2")
        )
        assert resp.status_code == 404


# ── OpenAPI wiring ─────────────────────────────────────────────────────


class TestOpenAPIWiring:
    @pytest.mark.asyncio
    async def test_companies_tagged_in_openapi(self, client):
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json()["paths"]
        assert "/api/v1/companies" in paths
        assert "/api/v1/companies/{company_id}" in paths
        assert "/api/v1/companies/{company_id}/history" in paths

    @pytest.mark.asyncio
    async def test_companies_operations_are_get(self, client):
        resp = await client.get("/openapi.json")
        ops = resp.json()["paths"]["/api/v1/companies"]
        assert set(ops.keys()) == {"get"}

    @pytest.mark.asyncio
    async def test_list_companies_response_model(self, client):
        resp = await client.get("/openapi.json")
        op = resp.json()["paths"]["/api/v1/companies"]["get"]
        assert "200" in op["responses"]

    @pytest.mark.asyncio
    async def test_get_company_404_in_openapi(self, client):
        resp = await client.get("/openapi.json")
        op = resp.json()["paths"]["/api/v1/companies/{company_id}"]["get"]
        assert "404" in op["responses"]

    @pytest.mark.asyncio
    async def test_company_schemas_defined_in_openapi(self, client):
        resp = await client.get("/openapi.json")
        schemas = resp.json()["components"]["schemas"]
        for name in (
            "CompanyListResponse",
            "CompanyDetailResponse",
            "CompanyHistoryResponse",
            "CompanySummaryResponse",
            "CompanySnapshotResponse",
        ):
            assert name in schemas
