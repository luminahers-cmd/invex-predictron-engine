"""Tests for the Phase 1 analysis service layer."""

import pytest

from app.schemas.analysis import StartupAnalysisRequest, StartupAnalysisResponse


def _make_request(**overrides) -> StartupAnalysisRequest:
    defaults = {
        "startup_name": "TestCo",
        "website": "https://testco.example.com",
        "description": "A test startup for service layer validation.",
    }
    defaults.update(overrides)
    return StartupAnalysisRequest(**defaults)


@pytest.mark.anyio
async def test_run_analysis_returns_response():
    from app.services.analysis import run_analysis
    from predictron_engine.engine import PredictronEngine

    engine = PredictronEngine()
    request = _make_request()
    result = await run_analysis(engine, request)

    assert isinstance(result, StartupAnalysisResponse)
    assert result.startup_name == "TestCo"
    assert 0 <= result.venture_score <= 100
    assert 0 <= result.market_score <= 100
    assert 0 <= result.founder_score <= 100
    assert 0 <= result.traction_score <= 100
    assert 0 <= result.confidence <= 1
    assert isinstance(result.recommendations, list)


@pytest.mark.anyio
async def test_run_analysis_maps_scores():
    from app.services.analysis import run_analysis
    from predictron_engine.engine import PredictronEngine

    engine = PredictronEngine()
    request = _make_request()
    result = await run_analysis(engine, request)

    assert result.venture_score > 0
    assert result.market_score > 0
    assert result.founder_score > 0
    assert result.traction_score > 0


@pytest.mark.anyio
async def test_run_analysis_recommends_are_strings():
    from app.services.analysis import run_analysis
    from predictron_engine.engine import PredictronEngine

    engine = PredictronEngine()
    request = _make_request()
    result = await run_analysis(engine, request)

    for rec in result.recommendations:
        assert isinstance(rec, str)
        assert len(rec) > 0


@pytest.mark.anyio
async def test_analyze_endpoint_uses_engine(client):
    response = await client.post(
        "/api/v1/analyze",
        json={
            "startup_name": "DirectEngine",
            "website": "https://direct.example.com",
            "description": "Testing that the analyze endpoint uses the engine directly.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["startup_name"] == "DirectEngine"
    assert isinstance(body["venture_score"], float)
    assert isinstance(body["confidence"], float)
