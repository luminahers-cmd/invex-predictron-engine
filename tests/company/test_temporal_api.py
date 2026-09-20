"""API tests for the Phase 3 temporal endpoints.

The companies router instantiates ``app.api.companies.PostgresCompanyStore``,
``CompanyHistoryService`` and ``CompanyDeltaService`` directly; tests replace
all three with in-memory, dataset-free implementations so no database or
offline dataset is required. JWT auth flows through the same dependency chain
as the rest of the API suite.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.auth.jwt import create_access_token
from app.services.company_deltas import CompanyDeltaService
from app.services.company_history import CompanyHistoryService
from tests.company.conftest import MemoryCompanyStore


def _headers(user_id: str | None = None) -> dict:
    if user_id is None:
        return {}
    return {"Authorization": f"Bearer {create_access_token(subject=user_id)}"}


def _make_report_resolver(store):
    """Resolve report payloads from an in-memory map (no database)."""

    async def _resolve(session, report_id):
        del session
        return store._reports.get(report_id)

    return _resolve


@pytest.fixture
def temporal_fake_store(monkeypatch):
    """Point the temporal endpoints at an in-memory, dataset-free stack."""
    store = MemoryCompanyStore()
    store._reports: dict[str, dict] = {}
    history = CompanyHistoryService(
        store=store, report_resolver=_make_report_resolver(store)
    )
    monkeypatch.setattr("app.api.companies.PostgresCompanyStore", lambda: store)
    monkeypatch.setattr(
        "app.api.companies.CompanyHistoryService",
        lambda *args, **kwargs: history,
    )
    monkeypatch.setattr(
        "app.api.companies.CompanyDeltaService",
        lambda *args, **kwargs: CompanyDeltaService(history=history),
    )
    return store


async def _seed_company(store, *, company_id, user_id=None):
    from app.services.companies import CompanyIdentityResolver

    resolved = CompanyIdentityResolver().resolve(company_id, None)
    record = await store.upsert_company(None, resolved, user_id=user_id)
    store._companies.pop(resolved.company_id, None)
    record.company_id = company_id
    store._companies[company_id] = record
    return record


async def _seed_snapshot(
    store,
    *,
    company_id,
    snap_id,
    created_at,
    decision="invest",
    confidence=0.8,
    composite=70.0,
    readiness=60.0,
):
    from app.services.company_protocols import CompanySnapshotPayload

    payload = CompanySnapshotPayload(
        company_id=company_id,
        analysis_id=f"analysis-{snap_id}",
        report_id=f"report-{snap_id}",
        decision=decision,
        confidence=confidence,
        composite_score=composite,
        readiness_score=readiness,
        dimension_scores={"market_opportunity": 75.0},
        created_at=created_at,
    )
    snapshot = await store.append_snapshot(None, payload)
    company = store._companies[company_id]
    company.snapshot_count += 1
    company.last_seen = max(company.last_seen, created_at)
    return snapshot


# ── timeline ───────────────────────────────────────────────────────────


class TestTimelineEndpoint:
    async def test_missing_company_404(self, client, temporal_fake_store):
        resp = await client.get(
            "/api/v1/companies/nope/timeline", headers=_headers("u1")
        )
        assert resp.status_code == 404

    async def test_empty_timeline(self, client, temporal_fake_store):
        await _seed_company(temporal_fake_store, company_id="tl-empty", user_id="u1")

        resp = await client.get(
            "/api/v1/companies/tl-empty/timeline", headers=_headers("u1")
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["company_id"] == "tl-empty"
        assert body["total"] == 0
        assert body["entries"] == []
        assert body["trend"]["snapshot_count"] == 0

    async def test_timeline_returns_enriched_entries_newest_first(
        self, client, temporal_fake_store
    ):
        await _seed_company(temporal_fake_store, company_id="tl-1", user_id="u1")
        earlier = await _seed_snapshot(
            temporal_fake_store,
            company_id="tl-1",
            snap_id="s0",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            confidence=0.7,
        )
        later = await _seed_snapshot(
            temporal_fake_store,
            company_id="tl-1",
            snap_id="s1",
            created_at=datetime(2025, 2, 1, tzinfo=UTC),
            confidence=0.85,
        )

        resp = await client.get(
            "/api/v1/companies/tl-1/timeline", headers=_headers("u1")
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert [entry["snapshot_id"] for entry in body["entries"]] == [
            later.snapshot.id,
            earlier.snapshot.id,
        ]
        entry = body["entries"][0]
        assert entry["decision"] == "invest"
        assert entry["confidence"] == 0.85
        assert entry["composite_score"] == 70.0
        assert entry["dimension_scores"]["market_opportunity"] == 75.0
        assert entry["benchmark"] is None
        assert entry["evidence_count"] is None
        assert entry["decision_explanation"] == []
        assert sorted(entry["key_facts"]) == [
            "composite_score=70",
            "confidence=0.85",
            "decision=invest",
            "dimension.market_opportunity=75",
            "readiness_score=60",
        ]

    async def test_timeline_enriches_from_seeded_report(
        self, client, temporal_fake_store
    ):
        await _seed_company(temporal_fake_store, company_id="tl-r", user_id="u1")
        await _seed_snapshot(
            temporal_fake_store,
            company_id="tl-r",
            snap_id="s0",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        temporal_fake_store._reports["report-s0"] = {
            "evidence": [{"statement": "e1"}],
            "investment_decision": {
                "rationale": {"key_evidence_summary": "Evidence summary A"}
            },
            "investment_readiness": {
                "readiness_level": "strong",
                "key_strengths": ["Great team"],
            },
            "recommendations": [{"action": "Hire CTO"}],
        }

        resp = await client.get(
            "/api/v1/companies/tl-r/timeline", headers=_headers("u1")
        )
        assert resp.status_code == 200
        entry = resp.json()["entries"][0]
        assert entry["evidence_count"] == 1
        assert entry["decision_explanation"] == ["Evidence summary A"]
        assert "recommendation=Hire CTO" in entry["key_facts"]
        assert "readiness_level=strong" in entry["key_facts"]
        assert "key_strength=Great team" in entry["key_facts"]

    async def test_timeline_pagination_validated(self, client, temporal_fake_store):
        await _seed_company(temporal_fake_store, company_id="tl-p", user_id="u1")
        resp = await client.get(
            "/api/v1/companies/tl-p/timeline",
            headers=_headers("u1"),
            params={"limit": 101},
        )
        assert resp.status_code == 422
        resp = await client.get(
            "/api/v1/companies/tl-p/timeline",
            headers=_headers("u1"),
            params={"limit": 0},
        )
        assert resp.status_code == 422
        resp = await client.get(
            "/api/v1/companies/tl-p/timeline",
            headers=_headers("u1"),
            params={"offset": -1},
        )
        assert resp.status_code == 422

    async def test_timeline_scope_other_user_404(self, client, temporal_fake_store):
        await _seed_company(temporal_fake_store, company_id="tl-scope", user_id="u1")

        resp = await client.get(
            "/api/v1/companies/tl-scope/timeline", headers=_headers("u2")
        )
        assert resp.status_code == 404

    async def test_anonymous_sees_public_timeline(self, client, temporal_fake_store):
        await _seed_company(temporal_fake_store, company_id="tl-open", user_id=None)
        await _seed_snapshot(
            temporal_fake_store,
            company_id="tl-open",
            snap_id="s0",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        )

        resp = await client.get("/api/v1/companies/tl-open/timeline")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1


# ── latest history ─────────────────────────────────────────────────────


class TestLatestEndpoint:
    async def test_missing_company_404(self, client, temporal_fake_store):
        resp = await client.get(
            "/api/v1/companies/nope/history/latest", headers=_headers("u1")
        )
        assert resp.status_code == 404

    async def test_latest_empty_company(self, client, temporal_fake_store):
        await _seed_company(temporal_fake_store, company_id="lat-empty", user_id="u1")

        resp = await client.get(
            "/api/v1/companies/lat-empty/history/latest", headers=_headers("u1")
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["entry"] is None
        assert body["trend"]["snapshot_count"] == 0

    async def test_latest_returns_newest_entry_and_trend(
        self, client, temporal_fake_store
    ):
        await _seed_company(temporal_fake_store, company_id="lat-1", user_id="u1")
        await _seed_snapshot(
            temporal_fake_store,
            company_id="lat-1",
            snap_id="s0",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            confidence=0.7,
        )
        await _seed_snapshot(
            temporal_fake_store,
            company_id="lat-1",
            snap_id="s1",
            created_at=datetime(2025, 2, 1, tzinfo=UTC),
            confidence=0.8,
        )
        newest = await _seed_snapshot(
            temporal_fake_store,
            company_id="lat-1",
            snap_id="s2",
            created_at=datetime(2025, 3, 1, tzinfo=UTC),
            confidence=0.9,
        )

        resp = await client.get(
            "/api/v1/companies/lat-1/history/latest", headers=_headers("u1")
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert body["entry"]["snapshot_id"] == newest.snapshot.id
        assert body["trend"]["confidence"] == "increasing"
        assert body["trend"]["snapshot_count"] == 3

    async def test_latest_scope_other_user_404(self, client, temporal_fake_store):
        await _seed_company(temporal_fake_store, company_id="lat-scope", user_id="u1")

        resp = await client.get(
            "/api/v1/companies/lat-scope/history/latest", headers=_headers("u2")
        )
        assert resp.status_code == 404


# ── deltas ─────────────────────────────────────────────────────────────


class TestDeltasEndpoint:
    async def test_missing_company_404(self, client, temporal_fake_store):
        resp = await client.get(
            "/api/v1/companies/nope/deltas", headers=_headers("u1")
        )
        assert resp.status_code == 404

    async def test_deltas_returns_comparisons_newest_first(
        self, client, temporal_fake_store
    ):
        await _seed_company(temporal_fake_store, company_id="dl-1", user_id="u1")
        earlier = await _seed_snapshot(
            temporal_fake_store,
            company_id="dl-1",
            snap_id="s0",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            confidence=0.7,
        )
        later = await _seed_snapshot(
            temporal_fake_store,
            company_id="dl-1",
            snap_id="s1",
            created_at=datetime(2025, 2, 1, tzinfo=UTC),
            confidence=0.8,
        )

        resp = await client.get(
            "/api/v1/companies/dl-1/deltas", headers=_headers("u1")
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["trend"]["confidence"] == "increasing"
        delta = body["deltas"][0]
        assert delta["previous_snapshot_id"] == earlier.snapshot.id
        assert delta["current_snapshot_id"] == later.snapshot.id
        fields = {field["field"]: field for field in delta["fields"]}
        confidence = fields["confidence"]
        assert confidence["delta_type"] == "numeric"
        assert confidence["status"] == "changed"
        assert confidence["previous"] == 0.7
        assert confidence["current"] == 0.8
        assert confidence["change"] == pytest.approx(0.1)

    async def test_deltas_empty_history(self, client, temporal_fake_store):
        await _seed_company(temporal_fake_store, company_id="dl-empty", user_id="u1")

        resp = await client.get(
            "/api/v1/companies/dl-empty/deltas", headers=_headers("u1")
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["deltas"] == []
        assert body["trend"]["snapshot_count"] == 0

    async def test_deltas_pagination_validated(self, client, temporal_fake_store):
        await _seed_company(temporal_fake_store, company_id="dl-v", user_id="u1")
        resp = await client.get(
            "/api/v1/companies/dl-v/deltas",
            headers=_headers("u1"),
            params={"limit": 0},
        )
        assert resp.status_code == 422

    async def test_deltas_scope_other_user_404(self, client, temporal_fake_store):
        await _seed_company(temporal_fake_store, company_id="dl-scope", user_id="u1")

        resp = await client.get(
            "/api/v1/companies/dl-scope/deltas", headers=_headers("u2")
        )
        assert resp.status_code == 404


# ── backwards compatibility and OpenAPI wiring ────────────────────────


class TestBackwardsCompatibility:
    async def test_history_endpoint_unchanged(self, client, temporal_fake_store):
        await _seed_company(temporal_fake_store, company_id="compat-1", user_id="u1")
        await _seed_snapshot(
            temporal_fake_store,
            company_id="compat-1",
            snap_id="s0",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        )

        resp = await client.get(
            "/api/v1/companies/compat-1/history", headers=_headers("u1")
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        snapshot = body["snapshots"][0]
        assert snapshot["analysis_id"] == "analysis-s0"
        for derived in ("benchmark", "decision_explanation", "key_facts"):
            assert derived not in snapshot

    async def test_list_endpoint_shape_unchanged(self, client, temporal_fake_store):
        await _seed_company(temporal_fake_store, company_id="compat-2", user_id="u1")

        resp = await client.get(
            "/api/v1/companies", headers=_headers("u1")
        )
        assert resp.status_code == 200
        assert set(resp.json().keys()) == {"companies", "total"}


class TestOpenAPIWiring:
    async def test_temporal_paths_registered(self, client):
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json()["paths"]
        for path in (
            "/api/v1/companies/{company_id}/timeline",
            "/api/v1/companies/{company_id}/history/latest",
            "/api/v1/companies/{company_id}/deltas",
        ):
            assert path in paths
            assert set(paths[path].keys()) == {"get"}

    async def test_temporal_404s_declared(self, client):
        resp = await client.get("/openapi.json")
        paths = resp.json()["paths"]
        for path in (
            "/api/v1/companies/{company_id}/timeline",
            "/api/v1/companies/{company_id}/history/latest",
            "/api/v1/companies/{company_id}/deltas",
        ):
            assert "404" in paths[path]["get"]["responses"]

    async def test_temporal_schemas_defined(self, client):
        resp = await client.get("/openapi.json")
        schemas = resp.json()["components"]["schemas"]
        for name in (
            "CompanyHistoryLatestResponse",
            "CompanyTimelineResponse",
            "CompanyTimelineEntry",
            "CompanyTrendSummary",
            "CompanyDeltaListResponse",
            "CompanyDelta",
            "DeltaField",
        ):
            assert name in schemas

    async def test_phase1_history_schema_unchanged(self, client):
        resp = await client.get("/openapi.json")
        schemas = resp.json()["components"]["schemas"]
        assert "CompanyHistoryResponse" in schemas
        assert "CompanyTimelineResponse" in schemas
