"""Company Comparison API tests."""

from __future__ import annotations

import pytest


def _comparison_payload():
    return {
        "company_names": ["AlphaCo", "BetaCo"],
        "descriptions": {
            "AlphaCo": "AlphaCo builds enterprise AI tools for the fintech sector.",
            "BetaCo": "BetaCo provides cloud infrastructure for healthcare data.",
        },
    }


class TestCompanyComparison:
    """Tests for POST /api/v1/compare."""

    @pytest.mark.asyncio
    async def test_compare_returns_200(self, client):
        resp = await client.post("/api/v1/compare", json=_comparison_payload())
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_compare_has_companies(self, client):
        resp = await client.post("/api/v1/compare", json=_comparison_payload())
        data = resp.json()
        assert data["companies"] == ["AlphaCo", "BetaCo"]

    @pytest.mark.asyncio
    async def test_compare_has_feature_diffs(self, client):
        resp = await client.post("/api/v1/compare", json=_comparison_payload())
        data = resp.json()
        assert "feature_diffs" in data
        assert isinstance(data["feature_diffs"]["feature_diffs"], list)

    @pytest.mark.asyncio
    async def test_compare_has_decision_diffs(self, client):
        resp = await client.post("/api/v1/compare", json=_comparison_payload())
        data = resp.json()
        assert "decision_diffs" in data
        assert isinstance(data["decision_diffs"]["decision_comparison"], list)

    @pytest.mark.asyncio
    async def test_compare_has_contribution_diffs(self, client):
        resp = await client.post("/api/v1/compare", json=_comparison_payload())
        data = resp.json()
        assert "contribution_diffs" in data
        assert isinstance(data["contribution_diffs"]["diffs"], list)

    @pytest.mark.asyncio
    async def test_compare_has_signal_diffs(self, client):
        resp = await client.post("/api/v1/compare", json=_comparison_payload())
        data = resp.json()
        assert "signal_diffs" in data

    @pytest.mark.asyncio
    async def test_compare_has_knowledge_graph_diffs(self, client):
        resp = await client.post("/api/v1/compare", json=_comparison_payload())
        data = resp.json()
        assert "knowledge_graph_diffs" in data

    @pytest.mark.asyncio
    async def test_compare_has_benchmark_comparison(self, client):
        resp = await client.post("/api/v1/compare", json=_comparison_payload())
        data = resp.json()
        assert "benchmark_comparison" in data

    @pytest.mark.asyncio
    async def test_compare_has_overall_summary(self, client):
        resp = await client.post("/api/v1/compare", json=_comparison_payload())
        data = resp.json()
        assert "overall_summary" in data
        assert "company_count" in data["overall_summary"]
        assert data["overall_summary"]["company_count"] == 2

    @pytest.mark.asyncio
    async def test_compare_single_company_rejected(self, client):
        resp = await client.post("/api/v1/compare", json={
            "company_names": ["OnlyCo"],
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_compare_three_companies(self, client):
        payload = {
            "company_names": ["A", "B", "C"],
            "descriptions": {
                "A": "Company A builds enterprise solutions for global fintech.",
                "B": "Company B provides cloud data services for healthcare.",
                "C": "Company C offers developer tools for modern teams.",
            },
        }
        resp = await client.post("/api/v1/compare", json=payload)
        data = resp.json()
        assert len(data["companies"]) == 3
        assert data["feature_diffs"]["companies"] == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_compare_score_range_in_summary(self, client):
        resp = await client.post("/api/v1/compare", json=_comparison_payload())
        data = resp.json()
        avg = data["overall_summary"]["avg_score"]
        assert 0 <= avg <= 100

    @pytest.mark.asyncio
    async def test_compare_decision_diffs_has_all_fields(self, client):
        resp = await client.post("/api/v1/compare", json=_comparison_payload())
        data = resp.json()
        for diff in data["decision_diffs"]["decision_comparison"]:
            assert "startup_name" in diff
            assert "overall_score" in diff
            assert "confidence" in diff
            assert "decision_category" in diff
