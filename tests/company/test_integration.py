"""Integration tests for the CIH ingest hook in app/services/analysis.py.

These exercise ``_ingest_company_hook`` directly and verify the additive
contract: the hook runs only after a successful persist, never raises, and
never masks an analysis response even when the registry fails.

The hook imports ``AsyncSessionLocal`` from ``app.db.session`` at call time,
so tests patch that symbol (not a local reference) when substituting the
real session factory.
"""

from __future__ import annotations

import pytest

from app.services.analysis import _ingest_company_hook, run_analysis

pytestmark = pytest.mark.asyncio


def _make_request():
    from app.schemas.analysis import StartupAnalysisRequest

    return StartupAnalysisRequest(
        startup_name="Acme Inc",
        website="https://acme.com",
        description="A sufficiently long description for validation purposes.",
    )


def _make_analysis(id: str = "req-1", user_id: str = "u1"):
    from app.models import AnalysisRequest

    return AnalysisRequest(
        id=id,
        user_id=user_id,
        startup_name="Acme Inc",
        website="https://acme.com",
        description="d",
    )


def _make_report():
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import (
        DecisionCategory,
        InvestmentDecision,
        InvestmentReadiness,
        Report,
        ScoreResult,
    )
    from predictron_engine.models.startup import Startup

    return Report(
        startup=Startup(
            name="Acme Inc",
            website="https://acme.com",
            description="desc",
        ),
        features=ExtractedFeatures(),
        overall_score=77.5,
        overall_confidence=0.83,
        scores=[
            ScoreResult(dimension="market_opportunity", score=80.0),
            ScoreResult(dimension="founder_quality", score=70.0),
        ],
        investment_decision=InvestmentDecision(
            category=DecisionCategory.INVEST,
            conviction="high",
            composite_score=75.2,
        ),
        investment_readiness=InvestmentReadiness(readiness_score=68.0),
    )


def _make_response():
    from app.schemas.analysis import StartupAnalysisResponse

    return StartupAnalysisResponse(
        startup_name="Acme Inc",
        venture_score=77.5,
        market_score=80.0,
        founder_score=70.0,
        traction_score=60.0,
        confidence=0.83,
    )


async def _seed_persisted_analysis(session, analysis_id: str = "req-1") -> str:
    """Insert the AnalysisRequest + AnalysisReport rows the hook resolves."""
    from app.models import AnalysisReport, AnalysisRequest

    request = AnalysisRequest(
        id=analysis_id,
        user_id="u1",
        startup_name="Acme Inc",
        website="https://acme.com",
        description="d",
    )
    session.add(request)
    report = AnalysisReport(
        id="report-for-" + analysis_id,
        request_id=analysis_id,
        startup_name="Acme Inc",
        venture_score=77.5,
        market_score=80.0,
        founder_score=70.0,
        traction_score=60.0,
        recommendations=[],
        confidence=0.83,
        full_report={},
    )
    session.add(report)
    await session.flush()
    return report.id


def _fake_session_factory(session):
    """Build an ``AsyncSessionLocal`` stand-in yielding a real session."""

    class FakeCtx:
        def __init__(self, inner):
            self._inner = inner

        async def __aenter__(self):
            return self._inner

        async def __aexit__(self, *exc):
            return None

        async def commit(self):
            await self._inner.commit()

    return lambda: FakeCtx(session)


async def test_hook_returns_for_healthy_path(sqlite_session):
    """The ingest service resolves a persisted report to a snapshot."""
    from app.services.companies import (
        CompanyIngestService,
        build_snapshot_inputs,
    )
    from app.services.company_postgres import PostgresCompanyStore

    analysis = _make_analysis()
    await _seed_persisted_analysis(sqlite_session, analysis.id)

    svc = CompanyIngestService(store=PostgresCompanyStore())
    result = await svc.ingest_after_persist(
        sqlite_session,
        analysis=analysis,
        inputs=build_snapshot_inputs(
            _make_request(), _make_report(), _make_response()
        ),
        user_id="u1",
    )
    assert result is not None
    assert result.created_snapshot is True


async def test_hook_swallows_store_failures(monkeypatch, sqlite_session):
    """A store failure must not raise out of the hook."""
    import app.db.session as session_mod
    from app.services import companies as companies_mod

    class BoomService:
        async def ingest_after_persist(self, session, *, analysis, inputs, user_id=None):
            raise RuntimeError("registry down")

    class BoomSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(companies_mod, "CompanyIngestService", lambda: BoomService())
    monkeypatch.setattr(session_mod, "AsyncSessionLocal", lambda: BoomSession())

    await _ingest_company_hook(
        _make_analysis(), _make_request(), _make_report(), _make_response(),
        user_id="u1",
    )


async def test_hook_swallows_build_inputs_failure(monkeypatch, sqlite_session):
    import app.db.session as session_mod
    from app.services import companies as companies_mod

    def boom(*args, **kwargs):
        raise ValueError("report malformed")

    monkeypatch.setattr(companies_mod, "build_snapshot_inputs", boom)
    monkeypatch.setattr(
        session_mod, "AsyncSessionLocal", lambda: (_ for _ in ()).throw(RuntimeError())
    )

    await _ingest_company_hook(
        _make_analysis(), _make_request(), _make_report(), _make_response(),
        user_id="u1",
    )


async def test_hook_commits_on_success(monkeypatch, sqlite_session):
    """The hook should commit after ingesting."""
    import app.db.session as session_mod
    from app.services import companies as companies_mod

    committed = []

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def commit(self):
            committed.append(True)

    class FakeService:
        async def ingest_after_persist(self, session, *, analysis, inputs, user_id=None):
            return True

    monkeypatch.setattr(session_mod, "AsyncSessionLocal", lambda: FakeSession())
    monkeypatch.setattr(companies_mod, "CompanyIngestService", lambda: FakeService())

    await _ingest_company_hook(
        _make_analysis(), _make_request(), _make_report(), _make_response(),
        user_id="u1",
    )
    assert committed == [True]


async def test_run_analysis_persists_and_ingests(monkeypatch):
    """End-to-end: successful persist triggers the hook; response carries id."""
    from app.services import analysis as analysis_mod

    calls = []

    class FakeEngine:
        def analyze(self, request):
            return _make_report()

    async def fake_persist(request, report, response, user_id=None):
        analysis = _make_analysis()
        response.id = analysis.id
        calls.append("persist")
        return analysis

    async def fake_hook(analysis, request, report, response, user_id=None):
        calls.append("ingest")

    monkeypatch.setattr(analysis_mod, "_persist_async", fake_persist)
    monkeypatch.setattr(analysis_mod, "_ingest_company_hook", fake_hook)

    response = await run_analysis(FakeEngine(), _make_request(), user_id="u1")
    assert response.id is not None
    assert calls == ["persist", "ingest"]


async def test_run_analysis_hook_skipped_when_persist_returns_none(monkeypatch):
    from app.services import analysis as analysis_mod

    calls = []

    class FakeEngine:
        def analyze(self, request):
            return _make_report()

    async def fake_persist(request, report, response, user_id=None):
        calls.append("persist")
        return None

    async def fake_hook(analysis, request, report, response, user_id=None):
        calls.append("ingest")

    monkeypatch.setattr(analysis_mod, "_persist_async", fake_persist)
    monkeypatch.setattr(analysis_mod, "_ingest_company_hook", fake_hook)

    response = await run_analysis(FakeEngine(), _make_request(), user_id="u1")
    assert response.id is None
    assert calls == ["persist"]


async def test_run_analysis_hook_does_not_mask_errors(monkeypatch):
    """The real hook swallows its own failures; the analysis still returns."""
    import app.db.session as session_mod
    from app.services import analysis as analysis_mod

    class FakeEngine:
        def analyze(self, request):
            return _make_report()

    async def fake_persist(request, report, response, user_id=None):
        return _make_analysis()

    class BoomSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def __anext__(self):
            raise StopAsyncIteration

    monkeypatch.setattr(analysis_mod, "_persist_async", fake_persist)
    monkeypatch.setattr(session_mod, "AsyncSessionLocal", lambda: BoomSession())

    response = await run_analysis(FakeEngine(), _make_request(), user_id="u1")
    assert response.startup_name == "Acme Inc"
    assert response.venture_score == 77.5


async def test_run_analysis_hook_wraps_in_try(monkeypatch):
    from app.services import analysis as analysis_mod

    real_hook = analysis_mod._ingest_company_hook
    assert real_hook is not None

    class FakeEngine:
        def analyze(self, request):
            return _make_report()

    async def fake_persist(request, report, response, user_id=None):
        return _make_analysis()

    monkeypatch.setattr(analysis_mod, "_persist_async", fake_persist)
    # don't patch the hook; it swallows its own exceptions internally.
    response = await run_analysis(FakeEngine(), _make_request(), user_id="u1")
    assert response.startup_name == "Acme Inc"


async def test_real_hook_persists_to_registry_sqlite(monkeypatch, sqlite_session):
    """Hook end-to-end with SQLite session replacing AsyncSessionLocal."""
    import app.db.session as session_mod
    from app.services.company_postgres import PostgresCompanyStore

    analysis = _make_analysis()
    await _seed_persisted_analysis(sqlite_session, analysis.id)
    monkeypatch.setattr(
        session_mod, "AsyncSessionLocal", _fake_session_factory(sqlite_session)
    )

    await _ingest_company_hook(
        analysis, _make_request(), _make_report(), _make_response(), user_id="u1"
    )

    store = PostgresCompanyStore()
    page = await store.list_companies(sqlite_session, user_id="u1")
    assert page.total == 1
    snap_page = await store.list_snapshots(
        sqlite_session, page.companies[0].company_id, user_id="u1"
    )
    assert snap_page.total == 1
    assert snap_page.snapshots[0].analysis_id == "req-1"
    assert snap_page.snapshots[0].decision == "invest"


async def test_report_to_response_maps_scores(monkeypatch):
    from app.services.analysis import _report_to_response

    response = _report_to_response("Acme Inc", _make_report())
    assert response.venture_score == 77.5
    assert response.market_score == 80.0
    assert response.confidence == 0.83


async def test_build_snapshot_inputs_maps_full_report():
    from app.services.companies import build_snapshot_inputs

    inputs = build_snapshot_inputs(_make_request(), _make_report(), _make_response())
    assert inputs.decision == "invest"
    assert inputs.composite_score == 75.2
    assert inputs.readiness_score == 68.0
    assert inputs.dimension_scores == {
        "market_opportunity": 80.0,
        "founder_quality": 70.0,
    }
    assert inputs.confidence == 0.83


async def test_hook_uses_analysis_id_as_snapshot_key(sqlite_session, monkeypatch):
    import app.db.session as session_mod
    from app.services.company_postgres import PostgresCompanyStore

    analysis = _make_analysis(id="unique-analysis-id")
    await _seed_persisted_analysis(sqlite_session, analysis.id)
    monkeypatch.setattr(
        session_mod, "AsyncSessionLocal", _fake_session_factory(sqlite_session)
    )

    await _ingest_company_hook(
        analysis, _make_request(), _make_report(), _make_response(), user_id="u1"
    )
    # Running twice must not duplicate: unique analysis_id.
    await _ingest_company_hook(
        analysis, _make_request(), _make_report(), _make_response(), user_id="u1"
    )

    store = PostgresCompanyStore()
    page = await store.list_companies(sqlite_session, user_id="u1")
    assert page.total == 1
    snap_page = await store.list_snapshots(
        sqlite_session, page.companies[0].company_id, user_id="u1"
    )
    assert snap_page.total == 1


async def test_hook_snapshot_created_at_uses_report_metadata(
    sqlite_session, monkeypatch
):
    import app.db.session as session_mod
    from app.services.company_postgres import PostgresCompanyStore

    analysis = _make_analysis()
    await _seed_persisted_analysis(sqlite_session, analysis.id)
    monkeypatch.setattr(
        session_mod, "AsyncSessionLocal", _fake_session_factory(sqlite_session)
    )

    report = _make_report()
    report.investment_readiness.readiness_score = 50.0

    await _ingest_company_hook(
        analysis, _make_request(), report, _make_response(), user_id="u1"
    )
    store = PostgresCompanyStore()
    page = await store.list_companies(sqlite_session, user_id="u1")
    snap_page = await store.list_snapshots(
        sqlite_session, page.companies[0].company_id, user_id="u1"
    )
    assert snap_page.snapshots[0].readiness_score == 50.0
    assert snap_page.snapshots[0].created_at is not None
