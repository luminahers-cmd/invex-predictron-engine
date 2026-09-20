"""Due Diligence Report API tests."""

from __future__ import annotations

import pytest

VALID_DD_PAYLOAD = {
    "startup_name": "TestDD",
    "description": "A due diligence test startup that builds enterprise software solutions.",
    "website_url": "https://testdd.example.com",
}


class TestDueDiligenceGenerate:
    """Tests for POST /api/v1/due-diligence."""

    @pytest.mark.asyncio
    async def test_due_diligence_returns_200(self, client):
        resp = await client.post("/api/v1/due-diligence", json=VALID_DD_PAYLOAD)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_due_diligence_has_executive_summary(self, client):
        resp = await client.post("/api/v1/due-diligence", json=VALID_DD_PAYLOAD)
        data = resp.json()
        assert "executive_summary" in data
        assert "headline" in data["executive_summary"]

    @pytest.mark.asyncio
    async def test_due_diligence_has_strengths(self, client):
        resp = await client.post("/api/v1/due-diligence", json=VALID_DD_PAYLOAD)
        data = resp.json()
        assert isinstance(data["strengths"], list)

    @pytest.mark.asyncio
    async def test_due_diligence_has_weaknesses(self, client):
        resp = await client.post("/api/v1/due-diligence", json=VALID_DD_PAYLOAD)
        data = resp.json()
        assert isinstance(data["weaknesses"], list)

    @pytest.mark.asyncio
    async def test_due_diligence_has_opportunities(self, client):
        resp = await client.post("/api/v1/due-diligence", json=VALID_DD_PAYLOAD)
        data = resp.json()
        assert isinstance(data["opportunities"], list)

    @pytest.mark.asyncio
    async def test_due_diligence_has_risks(self, client):
        resp = await client.post("/api/v1/due-diligence", json=VALID_DD_PAYLOAD)
        data = resp.json()
        assert isinstance(data["risks"], list)

    @pytest.mark.asyncio
    async def test_due_diligence_has_evidence(self, client):
        resp = await client.post("/api/v1/due-diligence", json=VALID_DD_PAYLOAD)
        data = resp.json()
        assert isinstance(data["evidence"], list)

    @pytest.mark.asyncio
    async def test_due_diligence_has_decision_trace(self, client):
        resp = await client.post("/api/v1/due-diligence", json=VALID_DD_PAYLOAD)
        data = resp.json()
        assert "decision_trace" in data
        assert "category" in data["decision_trace"]

    @pytest.mark.asyncio
    async def test_due_diligence_has_confidence(self, client):
        resp = await client.post("/api/v1/due-diligence", json=VALID_DD_PAYLOAD)
        data = resp.json()
        assert 0 <= data["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_due_diligence_has_benchmark_context(self, client):
        resp = await client.post("/api/v1/due-diligence", json=VALID_DD_PAYLOAD)
        data = resp.json()
        assert "benchmark_context" in data
        assert "composite_score" in data["benchmark_context"]

    @pytest.mark.asyncio
    async def test_due_diligence_has_supporting_features(self, client):
        resp = await client.post("/api/v1/due-diligence", json=VALID_DD_PAYLOAD)
        data = resp.json()
        assert "supporting_features" in data
        assert "dimension_scores" in data["supporting_features"]

    @pytest.mark.asyncio
    async def test_due_diligence_has_engine_version(self, client):
        resp = await client.post("/api/v1/due-diligence", json=VALID_DD_PAYLOAD)
        data = resp.json()
        assert data.get("engine_version") is not None

    @pytest.mark.asyncio
    async def test_due_diligence_has_processing_time(self, client):
        resp = await client.post("/api/v1/due-diligence", json=VALID_DD_PAYLOAD)
        data = resp.json()
        assert data.get("processing_time_ms") is not None

    @pytest.mark.asyncio
    async def test_due_diligence_minimal_payload(self, client):
        payload = {
            "startup_name": "MinDD",
            "description": "Minimal due diligence test startup description.",
        }
        resp = await client.post("/api/v1/due-diligence", json=payload)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_due_diligence_missing_name_returns_422(self, client):
        resp = await client.post("/api/v1/due-diligence", json={
            "description": "Missing startup name field.",
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_due_diligence_strength_item_structure(self, client):
        resp = await client.post("/api/v1/due-diligence", json=VALID_DD_PAYLOAD)
        data = resp.json()
        for s in data["strengths"][:1]:
            assert "title" in s
            assert "dimension" in s
            assert "severity" in s

    @pytest.mark.asyncio
    async def test_due_diligence_risk_item_structure(self, client):
        resp = await client.post("/api/v1/due-diligence", json=VALID_DD_PAYLOAD)
        data = resp.json()
        for r in data["risks"][:1]:
            assert "title" in r
            assert "severity" in r
