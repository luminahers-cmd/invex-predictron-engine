"""Unit tests for the analysis persistence layer.

These tests mock the SQLAlchemy session to validate persistence logic
without requiring a running database.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas.analysis import (
    AnalysisDetailResponse,
    AnalysisListResponse,
    AnalysisSummaryResponse,
    StartupAnalysisRequest,
    StartupAnalysisResponse,
)


def _make_request(**overrides) -> StartupAnalysisRequest:
    defaults = {
        "startup_name": "PersistCo",
        "website": "https://persistco.example.com",
        "description": "A test startup for persistence validation.",
    }
    defaults.update(overrides)
    return StartupAnalysisRequest(**defaults)


def _make_response(**overrides) -> StartupAnalysisResponse:
    defaults = {
        "startup_name": "PersistCo",
        "venture_score": 72.5,
        "market_score": 80.0,
        "founder_score": 65.0,
        "traction_score": 70.0,
        "recommendations": ["Conduct due diligence", "Evaluate market fit"],
        "confidence": 0.82,
    }
    defaults.update(overrides)
    return StartupAnalysisResponse(**defaults)


def _make_report(**overrides):
    from predictron_engine.models.report import AnalysisMetadata

    report = MagicMock()
    report.overall_score = overrides.get("overall_score", 72.5)
    report.overall_confidence = overrides.get("overall_confidence", 0.82)
    report.scores = []
    report.recommendations = []
    report.analysis_metadata = AnalysisMetadata(
        engine_version="0.12.1",
        processing_time_ms=42.5,
    )
    report.model_dump = MagicMock(
        return_value={
            "startup": {"name": "PersistCo"},
            "overall_score": 72.5,
            "overall_confidence": 0.82,
            "scores": [],
            "recommendations": [],
        }
    )
    return report


# ── persist_analysis ──────────────────────────────────────────────────


@pytest.mark.anyio
async def test_persist_analysis_creates_request_and_report():
    from app.services.persistence import persist_analysis

    mock_session = AsyncMock()
    mock_session.add = MagicMock()  # add() is sync in SQLAlchemy
    request = _make_request()
    response = _make_response()
    report = _make_report()

    result = await persist_analysis(mock_session, request, report, response)

    assert mock_session.add.call_count == 2
    assert mock_session.flush.call_count == 2
    assert result.startup_name == "PersistCo"
    assert "persistco.example.com" in result.website


@pytest.mark.anyio
async def test_persist_analysis_stores_scores():
    from app.services.persistence import persist_analysis

    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    request = _make_request()
    response = _make_response(
        venture_score=85.0, market_score=90.0, founder_score=75.0, traction_score=80.0
    )
    report = _make_report()

    await persist_analysis(mock_session, request, report, response)

    # Verify the second add call (the report) has correct scores
    report_obj = mock_session.add.call_args_list[1][0][0]
    assert report_obj.venture_score == 85.0
    assert report_obj.market_score == 90.0
    assert report_obj.founder_score == 75.0
    assert report_obj.traction_score == 80.0


@pytest.mark.anyio
async def test_persist_analysis_stores_metadata():
    from app.services.persistence import persist_analysis

    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    request = _make_request()
    response = _make_response()
    report = _make_report()

    await persist_analysis(mock_session, request, report, response)

    report_obj = mock_session.add.call_args_list[1][0][0]
    assert report_obj.engine_version == "0.12.1"
    assert report_obj.processing_time_ms == 42.5
    assert isinstance(report_obj.full_report, dict)


@pytest.mark.anyio
async def test_persist_analysis_stores_request_fields():
    from app.services.persistence import persist_analysis

    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    request = _make_request(
        pitch_deck_url="https://example.com/pitch.pdf",
        founder_linkedin_urls=["https://linkedin.com/in/janedoe"],
    )
    response = _make_response()
    report = _make_report()

    result = await persist_analysis(mock_session, request, report, response)

    assert result.pitch_deck_url == "https://example.com/pitch.pdf"
    assert result.founder_linkedin_urls == ["https://linkedin.com/in/janedoe"]


@pytest.mark.anyio
async def test_persist_analysis_handles_none_optional_fields():
    from app.services.persistence import persist_analysis

    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    request = _make_request(pitch_deck_url=None, founder_linkedin_urls=[])
    response = _make_response()
    report = _make_report()

    result = await persist_analysis(mock_session, request, report, response)

    assert result.pitch_deck_url is None
    assert result.founder_linkedin_urls == []


# ── get_analysis ──────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_get_analysis_returns_detail_when_found():
    from app.models.analysis import AnalysisReport, AnalysisRequest
    from app.services.persistence import get_analysis

    mock_session = AsyncMock()
    mock_request = AnalysisRequest(
        id="test-id-123",
        startup_name="PersistCo",
        website="https://persistco.example.com",
        description="A test startup.",
        pitch_deck_url=None,
        founder_linkedin_urls=[],
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    mock_report = AnalysisReport(
        id="report-id-456",
        request_id="test-id-123",
        startup_name="PersistCo",
        venture_score=72.5,
        market_score=80.0,
        founder_score=65.0,
        traction_score=70.0,
        recommendations=["Do diligence"],
        confidence=0.82,
        engine_version="0.12.1",
        processing_time_ms=42.5,
        full_report={"scores": []},
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )

    mock_result = MagicMock()
    mock_result.one_or_none.return_value = (mock_request, mock_report)
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await get_analysis(mock_session, "test-id-123")

    assert result is not None
    assert isinstance(result, AnalysisDetailResponse)
    assert result.id == "test-id-123"
    assert result.startup_name == "PersistCo"
    assert result.venture_score == 72.5
    assert result.full_report == {"scores": []}


@pytest.mark.anyio
async def test_get_analysis_returns_none_when_not_found():
    from app.services.persistence import get_analysis

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await get_analysis(mock_session, "nonexistent-id")

    assert result is None


# ── list_analyses ─────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_list_analyses_returns_summaries():
    from app.models.analysis import AnalysisReport, AnalysisRequest
    from app.services.persistence import list_analyses

    mock_session = AsyncMock()

    mock_request = AnalysisRequest(
        id="id-1",
        startup_name="Co1",
        website="https://co1.example.com",
        description="First.",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    mock_report = AnalysisReport(
        id="rpt-1",
        request_id="id-1",
        startup_name="Co1",
        venture_score=80.0,
        market_score=75.0,
        founder_score=70.0,
        traction_score=85.0,
        recommendations=[],
        confidence=0.9,
        engine_version="0.12.1",
        processing_time_ms=30.0,
        full_report={},
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )

    count_result = MagicMock()
    count_result.scalar.return_value = 1

    list_result = MagicMock()
    list_result.all.return_value = [(mock_request, mock_report)]

    mock_session.execute = AsyncMock(side_effect=[count_result, list_result])

    result = await list_analyses(mock_session, offset=0, limit=20)

    assert isinstance(result, AnalysisListResponse)
    assert result.total == 1
    assert len(result.analyses) == 1
    assert isinstance(result.analyses[0], AnalysisSummaryResponse)
    assert result.analyses[0].startup_name == "Co1"


@pytest.mark.anyio
async def test_list_analyses_returns_empty_list():
    from app.services.persistence import list_analyses

    mock_session = AsyncMock()

    count_result = MagicMock()
    count_result.scalar.return_value = 0

    list_result = MagicMock()
    list_result.all.return_value = []

    mock_session.execute = AsyncMock(side_effect=[count_result, list_result])

    result = await list_analyses(mock_session, offset=0, limit=20)

    assert result.total == 0
    assert len(result.analyses) == 0


# ── Analysis model defaults ───────────────────────────────────────────


def test_analysis_request_model_construction():
    from app.models.analysis import AnalysisRequest

    req = AnalysisRequest(
        startup_name="Test",
        website="https://test.com",
        description="Desc",
        founder_linkedin_urls=[],
    )
    assert req.startup_name == "Test"
    assert req.website == "https://test.com"


def test_analysis_report_model_construction():
    from app.models.analysis import AnalysisReport

    rpt = AnalysisReport(
        request_id="req-1",
        startup_name="Test",
        venture_score=50.0,
        market_score=50.0,
        founder_score=50.0,
        traction_score=50.0,
        confidence=0.5,
        full_report={},
        recommendations=[],
    )
    assert rpt.request_id == "req-1"
    assert rpt.venture_score == 50.0


# ── Schema validation ─────────────────────────────────────────────────


def test_analysis_summary_response_schema():
    resp = AnalysisSummaryResponse(
        id="test-id",
        startup_name="TestCo",
        venture_score=75.0,
        market_score=80.0,
        founder_score=70.0,
        traction_score=65.0,
        confidence=0.85,
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert resp.engine_version is None
    assert resp.processing_time_ms is None


def test_analysis_detail_response_schema():
    resp = AnalysisDetailResponse(
        id="test-id",
        startup_name="TestCo",
        website="https://testco.example.com",
        description="A test startup.",
        venture_score=75.0,
        market_score=80.0,
        founder_score=70.0,
        traction_score=65.0,
        confidence=0.85,
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert resp.pitch_deck_url is None
    assert resp.founder_linkedin_urls == []
    assert resp.full_report == {}


def test_analysis_list_response_schema():
    resp = AnalysisListResponse(analyses=[], total=0)
    assert len(resp.analyses) == 0
    assert resp.total == 0
