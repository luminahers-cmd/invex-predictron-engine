"""Integration tests for the feature-flagged analysis -> forecast hook."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import app.core.config as config_module
import app.db.session as db_session_module
from app.services.analysis import _register_forecast_hook, _report_engine_version

SNAPSHOT_AT = datetime.now(UTC) - timedelta(days=30)


class _FakeSettings:
    def __init__(self, *, enabled: bool) -> None:
        self.FORECAST_ENABLED = enabled
        self.FORECAST_DEFAULT_HORIZON_DAYS = 365


async def _seed_company(session, *, name: str, user_id: str | None = "u1") -> str:
    from app.services.companies import CompanyIdentityResolver
    from app.services.company_postgres import PostgresCompanyStore

    identity = CompanyIdentityResolver().resolve(name, None)
    record = await PostgresCompanyStore().upsert_company(session, identity, user_id=user_id)
    return record.company_id


async def _seed_snapshot(session, *, company_id: str):
    from app.services.company_postgres import PostgresCompanyStore
    from app.services.company_protocols import CompanySnapshotPayload

    payload = CompanySnapshotPayload(
        company_id=company_id,
        analysis_id="analysis-hook",
        report_id="report-hook",
        decision="watch",
        confidence=0.6,
        composite_score=55.0,
        readiness_score=50.0,
        dimension_scores={"market": 60.0},
        created_at=SNAPSHOT_AT,
    )
    append = await PostgresCompanyStore().append_snapshot(session, payload)
    return append.snapshot


def _test_settings(**overrides) -> object:
    return _FakeSettings(**overrides)


class TestRegisterForecastHook:
    @pytest.mark.asyncio
    async def test_noop_when_disabled(self, sqlite_session, monkeypatch) -> None:
        monkeypatch.setattr(config_module, "get_settings", lambda: _test_settings(enabled=False))
        cid = await _seed_company(sqlite_session, name="cmp-hook")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        await sqlite_session.commit()

        forecast_id = await _register_forecast_hook(
            company_id=cid,
            snapshot_id=snapshot.id,
            engine_version="1.0",
            user_id="u1",
        )
        assert forecast_id is None

    @pytest.mark.asyncio
    async def test_registers_forecast_when_enabled(
        self, sqlite_engine, sqlite_session, monkeypatch
    ) -> None:
        from app.models.forecast import Forecast

        monkeypatch.setattr(config_module, "get_settings", lambda: _test_settings(enabled=True))
        factory = async_sessionmaker(
            sqlite_engine, class_=AsyncSession, expire_on_commit=False
        )
        monkeypatch.setattr(db_session_module, "AsyncSessionLocal", factory)

        cid = await _seed_company(sqlite_session, name="cmp-hook")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        await sqlite_session.commit()

        forecast_id = await _register_forecast_hook(
            company_id=cid,
            snapshot_id=snapshot.id,
            engine_version="9.9.9",
            user_id="u1",
        )
        assert forecast_id is not None
        assert len(forecast_id) == 64

        async with factory() as check:
            raw = await check.get(Forecast, forecast_id)
            assert raw is not None
            assert raw.engine_version == "9.9.9"
            assert raw.schema_version == "1.0"
            assert raw.status == "active"
            assert raw.company_id == cid
            assert raw.snapshot_id == snapshot.id

    @pytest.mark.asyncio
    async def test_registration_failure_is_swallowed(
        self, sqlite_engine, sqlite_session, monkeypatch
    ) -> None:
        monkeypatch.setattr(config_module, "get_settings", lambda: _test_settings(enabled=True))
        factory = async_sessionmaker(
            sqlite_engine, class_=AsyncSession, expire_on_commit=False
        )
        monkeypatch.setattr(db_session_module, "AsyncSessionLocal", factory)

        cid = await _seed_company(sqlite_session, name="cmp-hook")
        await _seed_snapshot(sqlite_session, company_id=cid)
        await sqlite_session.commit()

        forecast_id = await _register_forecast_hook(
            company_id=cid,
            snapshot_id="no-such-snapshot",
            engine_version="1.0",
            user_id="u1",
        )
        assert forecast_id is None


class TestReportEngineVersion:
    def test_returns_none_without_metadata(self) -> None:
        assert _report_engine_version(object()) is None

    def test_reads_engine_version_from_metadata(self) -> None:
        class FakeMetadata:
            engine_version = "7.7.7"

        class FakeReport:
            analysis_metadata = FakeMetadata()

        assert _report_engine_version(FakeReport()) == "7.7.7"


class TestStartupResponse:
    def test_forecast_id_defaults_none(self) -> None:
        from app.schemas.analysis import StartupAnalysisResponse

        response = StartupAnalysisResponse(
            startup_name="x",
            venture_score=50.0,
            market_score=50.0,
            founder_score=50.0,
            traction_score=50.0,
            recommendations=[],
            confidence=0.5,
        )
        assert response.forecast_id is None
