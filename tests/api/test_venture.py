"""Venture Analysis API tests."""

from __future__ import annotations

import pytest

VALID_PAYLOAD = {
    "startup_name": "TestVenture",
    "description": "A test startup company that builds innovative software solutions.",
    "website_url": "https://testventure.example.com",
}


class TestVentureAnalysis:
    """Tests for POST /api/v1/venture."""

    @pytest.mark.asyncio
    async def test_venture_analyze_returns_200(self, client):
        resp = await client.post("/api/v1/venture", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert data["startup_name"] == "TestVenture"
        assert 0 <= data["overall_score"] <= 100
        assert 0 <= data["overall_confidence"] <= 1

    @pytest.mark.asyncio
    async def test_venture_analyze_has_dimension_scores(self, client):
        resp = await client.post("/api/v1/venture", json=VALID_PAYLOAD)
        data = resp.json()
        assert "dimension_scores" in data
        assert isinstance(data["dimension_scores"], list)
        assert len(data["dimension_scores"]) > 0

    @pytest.mark.asyncio
    async def test_venture_analyze_has_decision_category(self, client):
        resp = await client.post("/api/v1/venture", json=VALID_PAYLOAD)
        data = resp.json()
        assert data.get("decision_category") is not None
        assert data.get("conviction_level") is not None

    @pytest.mark.asyncio
    async def test_venture_analyze_returns_engine_version(self, client):
        resp = await client.post("/api/v1/venture", json=VALID_PAYLOAD)
        data = resp.json()
        assert data.get("engine_version") is not None

    @pytest.mark.asyncio
    async def test_venture_analyze_minimal_payload(self, client):
        payload = {
            "startup_name": "MinimalCo",
            "description": "A minimal test company description for validation.",
        }
        resp = await client.post("/api/v1/venture", json=payload)
        assert resp.status_code == 200
        assert resp.json()["startup_name"] == "MinimalCo"

    @pytest.mark.asyncio
    async def test_venture_analyze_empty_name_returns_422(self, client):
        payload = {
            "startup_name": "",
            "description": "A test startup with empty name field.",
        }
        resp = await client.post("/api/v1/venture", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_venture_analyze_missing_description_returns_422(self, client):
        resp = await client.post("/api/v1/venture", json={"startup_name": "X"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_venture_analyze_short_description_returns_422(self, client):
        resp = await client.post("/api/v1/venture", json={
            "startup_name": "X",
            "description": "short",
        })
        assert resp.status_code == 422


class TestVentureExplain:
    """Tests for POST /api/v1/venture/explain."""

    @pytest.mark.asyncio
    async def test_explain_returns_200(self, client):
        resp = await client.post("/api/v1/venture/explain", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert "company_id" in data
        assert data.get("explanation") is None or "headline" in data.get("explanation", {})

    @pytest.mark.asyncio
    async def test_explain_has_verdict(self, client):
        resp = await client.post("/api/v1/venture/explain", json=VALID_PAYLOAD)
        data = resp.json()
        assert "verdict" in data


class TestVentureTrace:
    """Tests for POST /api/v1/venture/trace."""

    @pytest.mark.asyncio
    async def test_trace_returns_200(self, client):
        resp = await client.post("/api/v1/venture/trace", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert "company_id" in data
        assert "trace_id" in data


class TestVentureFeatures:
    """Tests for POST /api/v1/venture/features."""

    @pytest.mark.asyncio
    async def test_features_returns_200(self, client):
        resp = await client.post("/api/v1/venture/features", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert "company_id" in data
        assert "feature_count" in data
        assert isinstance(data["features"], list)


class TestVentureKnowledgeGraph:
    """Tests for GET /api/v1/venture/knowledge-graph."""

    @pytest.mark.asyncio
    async def test_knowledge_graph_returns_200(self, client):
        resp = await client.get("/api/v1/venture/knowledge-graph")
        assert resp.status_code == 200
        data = resp.json()
        assert "node_count" in data
        assert "edge_count" in data
        assert "node_type_counts" in data


class TestVentureSignalsTimeline:
    """Tests for GET /api/v1/venture/signals/{company_id}."""

    @pytest.mark.asyncio
    async def test_signals_timeline_returns_200(self, client):
        resp = await client.get("/api/v1/venture/signals/test_company")
        assert resp.status_code == 200
        data = resp.json()
        assert data["company_id"] == "test_company"
        assert "signal_count" in data

    @pytest.mark.asyncio
    async def test_signals_timeline_empty_company(self, client):
        resp = await client.get("/api/v1/venture/signals/nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["signal_count"] == 0


class TestVentureSignalTrends:
    """Tests for GET /api/v1/venture/signals/{company_id}/trends."""

    @pytest.mark.asyncio
    async def test_signal_trends_returns_200(self, client):
        resp = await client.get("/api/v1/venture/signals/co/trends")
        assert resp.status_code == 200
        data = resp.json()
        assert "trends" in data


class TestVentureSignalAggregation:
    """Tests for GET /api/v1/venture/signals/{company_id}/aggregation."""

    @pytest.mark.asyncio
    async def test_signal_aggregation_returns_200(self, client):
        resp = await client.get("/api/v1/venture/signals/co/aggregation")
        assert resp.status_code == 200
        data = resp.json()
        assert "momentum_score" in data


class TestVentureBenchmark:
    """Tests for POST /api/v1/venture/benchmark."""

    @pytest.mark.asyncio
    async def test_benchmark_returns_200(self, client):
        resp = await client.post("/api/v1/venture/benchmark", json=VALID_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert "startup_name" in data
        assert "composite_score" in data
        assert "benchmark_mean" in data

    @pytest.mark.asyncio
    async def test_benchmark_percentile_range(self, client):
        resp = await client.post("/api/v1/venture/benchmark", json=VALID_PAYLOAD)
        data = resp.json()
        assert 0 <= data["percentile_rank"] <= 100
