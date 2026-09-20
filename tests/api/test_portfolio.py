"""Portfolio Analysis API tests."""

from __future__ import annotations

import pytest


def _portfolio_payload(names=None):
    return {
        "company_names": names or ["AlphaCo", "BetaCo"],
        "descriptions": {
            "AlphaCo": "AlphaCo builds enterprise AI tools for fintech sector.",
            "BetaCo": "BetaCo provides cloud infrastructure for healthcare data.",
        },
        "website_urls": {
            "AlphaCo": "https://alpha.example.com",
            "BetaCo": "https://beta.example.com",
        },
    }


class TestPortfolioAnalyze:
    """Tests for POST /api/v1/portfolio."""

    @pytest.mark.asyncio
    async def test_portfolio_returns_200(self, client):
        resp = await client.post("/api/v1/portfolio", json=_portfolio_payload())
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_portfolio_company_count(self, client):
        resp = await client.post("/api/v1/portfolio", json=_portfolio_payload())
        data = resp.json()
        assert data["company_count"] == 2

    @pytest.mark.asyncio
    async def test_portfolio_score_range(self, client):
        resp = await client.post("/api/v1/portfolio", json=_portfolio_payload())
        data = resp.json()
        assert 0 <= data["portfolio_score"] <= 100

    @pytest.mark.asyncio
    async def test_portfolio_has_sector_distribution(self, client):
        resp = await client.post("/api/v1/portfolio", json=_portfolio_payload())
        data = resp.json()
        assert "sector_distribution" in data
        assert isinstance(data["sector_distribution"], list)

    @pytest.mark.asyncio
    async def test_portfolio_has_stage_distribution(self, client):
        resp = await client.post("/api/v1/portfolio", json=_portfolio_payload())
        data = resp.json()
        assert "stage_distribution" in data

    @pytest.mark.asyncio
    async def test_portfolio_has_risk_summary(self, client):
        resp = await client.post("/api/v1/portfolio", json=_portfolio_payload())
        data = resp.json()
        assert "risk_summary" in data
        assert "high_risk_count" in data["risk_summary"]

    @pytest.mark.asyncio
    async def test_portfolio_has_diversification(self, client):
        resp = await client.post("/api/v1/portfolio", json=_portfolio_payload())
        data = resp.json()
        assert "diversification" in data
        assert 0 <= data["diversification"]["overall_diversification"] <= 1

    @pytest.mark.asyncio
    async def test_portfolio_has_concentration_risk(self, client):
        resp = await client.post("/api/v1/portfolio", json=_portfolio_payload())
        data = resp.json()
        assert "concentration_risk" in data

    @pytest.mark.asyncio
    async def test_portfolio_has_heatmap_data(self, client):
        resp = await client.post("/api/v1/portfolio", json=_portfolio_payload())
        data = resp.json()
        assert "heatmap_data" in data
        assert isinstance(data["heatmap_data"], list)

    @pytest.mark.asyncio
    async def test_portfolio_has_similarity_matrix(self, client):
        resp = await client.post("/api/v1/portfolio", json=_portfolio_payload())
        data = resp.json()
        assert "similarity_matrix" in data

    @pytest.mark.asyncio
    async def test_portfolio_has_processing_time(self, client):
        resp = await client.post("/api/v1/portfolio", json=_portfolio_payload())
        data = resp.json()
        assert data.get("processing_time_ms") is not None

    @pytest.mark.asyncio
    async def test_portfolio_single_company_rejected(self, client):
        resp = await client.post("/api/v1/portfolio", json={
            "company_names": ["OnlyCo"],
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_portfolio_empty_rejected(self, client):
        resp = await client.post("/api/v1/portfolio", json={
            "company_names": [],
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_portfolio_three_companies(self, client):
        payload = _portfolio_payload(names=["A", "B", "C"])
        resp = await client.post("/api/v1/portfolio", json=payload)
        data = resp.json()
        assert data["company_count"] == 3


class TestPortfolioSimilarity:
    """Tests for POST /api/v1/portfolio/similarity."""

    @pytest.mark.asyncio
    async def test_similarity_returns_200(self, client):
        resp = await client.post(
            "/api/v1/portfolio/similarity", json=_portfolio_payload()
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "companies" in data
        assert len(data["companies"]) == 2
