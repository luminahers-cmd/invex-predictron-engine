"""Service tests for the Live Prediction Ledger (CIH Phase 5).

Runs the real ``ForecastService`` against the isolated in-memory SQLite
database, seeding companies/snapshots/outcomes through the existing stores
so the exact query paths (including deterministic outcome/evaluation ids)
are exercised.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from app.models.forecast import Forecast
from app.services.company_outcomes import CompanyOutcomeService
from app.services.forecasts import (
    ForecastService,
    ForecastSnapshotNotFoundError,
)
from predictron_engine.dataset.forecast_lifecycle import (
    ForecastEventType,
    ForecastStatus,
)
from predictron_engine.dataset.outcomes import StartupOutcome

SNAPSHOT_AT = datetime.now(UTC) - timedelta(days=30)
EARLY_OUTCOME_AT = datetime.now(UTC) - timedelta(days=20)
OUTCOME_AT = datetime.now(UTC) - timedelta(days=10)
AFTER_HORIZON = datetime.now(UTC) + timedelta(days=400)


async def _seed_company(session, *, name: str, user_id: str | None = "u1") -> str:
    from app.services.companies import CompanyIdentityResolver
    from app.services.company_postgres import PostgresCompanyStore

    identity = CompanyIdentityResolver().resolve(name, None)
    record = await PostgresCompanyStore().upsert_company(session, identity, user_id=user_id)
    return record.company_id


async def _seed_snapshot(session, *, company_id: str, created_at: datetime = SNAPSHOT_AT):
    from app.services.company_postgres import PostgresCompanyStore
    from app.services.company_protocols import CompanySnapshotPayload

    payload = CompanySnapshotPayload(
        company_id=company_id,
        analysis_id=f"analysis-{company_id[:24]}",
        report_id=f"report-{company_id[:24]}",
        decision="invest",
        confidence=0.8,
        composite_score=70.0,
        readiness_score=60.0,
        dimension_scores={"market": 80.0},
        created_at=created_at,
    )
    append = await PostgresCompanyStore().append_snapshot(session, payload)
    return append.snapshot


def _success_outcome_payload() -> dict:
    return {
        "status": "fully_verified",
        "acquisition": "Acquirer Corp",
        "acquisition_price_usd": 50_000_000.0,
        "exit_type": "acquisition",
    }


async def _seed_outcome(
    session,
    *,
    company_id: str,
    occurred_at: datetime = OUTCOME_AT,
    user_id: str | None = "u1",
):
    from app.schemas.outcome import OutcomeCreateRequest

    request = OutcomeCreateRequest(
        outcome=StartupOutcome.model_validate(_success_outcome_payload()),
        occurred_at=occurred_at,
        source="manual",
        notes="seeded outcome",
    )
    return await CompanyOutcomeService().record(
        session, company_id, request, user_id=user_id
    )


class TestRegister:
    @pytest.mark.asyncio
    async def test_registers_forecast_with_events(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)

        result = await ForecastService().register(
            sqlite_session,
            company_id=cid,
            snapshot_id=snapshot.id,
            engine_version="1.2.3",
            user_id="u1",
        )
        assert result is not None
        assert result.created is True
        forecast = result.forecast
        assert len(forecast.id) == 64
        assert forecast.company_id == cid
        assert forecast.snapshot_id == snapshot.id
        assert forecast.horizon_days == 365
        assert forecast.status == ForecastStatus.ACTIVE.value
        assert forecast.decision == "invest"
        assert forecast.confidence == 0.8
        assert forecast.composite_score == 70.0
        assert forecast.engine_version == "1.2.3"
        assert forecast.schema_version == "1.0"
        assert forecast.prediction["decision"] == "invest"
        assert forecast.prediction["composite_score"] == 70.0
        assert forecast.outcome_id is None
        assert forecast.reconciled_at is None

        events = await ForecastService().list_events(sqlite_session, forecast.id)
        assert [e.event_type for e in events] == [
            ForecastEventType.REGISTERED.value
        ]

    @pytest.mark.asyncio
    async def test_register_is_idempotent(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        service = ForecastService()

        first = await service.register(
            sqlite_session, company_id=cid, snapshot_id=snapshot.id, user_id="u1"
        )
        second = await service.register(
            sqlite_session, company_id=cid, snapshot_id=snapshot.id, user_id="u1"
        )
        assert first is not None and second is not None
        assert first.created is True
        assert second.created is False
        assert second.forecast.id == first.forecast.id
        assert second.forecast.prediction == first.forecast.prediction

        events = await ForecastService().list_events(sqlite_session, first.forecast.id)
        assert sum(1 for e in events) == 1

    @pytest.mark.asyncio
    async def test_duplicate_registration_never_overwrites_metadata(
        self, sqlite_session
    ) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        service = ForecastService()

        first = await service.register(
            sqlite_session,
            company_id=cid,
            snapshot_id=snapshot.id,
            engine_version="1.0.0",
            user_id="u1",
        )
        second = await service.register(
            sqlite_session,
            company_id=cid,
            snapshot_id=snapshot.id,
            engine_version="9.9.9",
            user_id="u1",
        )
        assert first is not None and second is not None
        assert second.created is False
        assert second.forecast.engine_version == "1.0.0"
        assert second.forecast.schema_version == "1.0"

    @pytest.mark.asyncio
    async def test_engine_version_resolved_from_report(self, sqlite_session) -> None:
        from app.models.analysis import AnalysisReport, AnalysisRequest

        cid = await _seed_company(sqlite_session, name="cmp-1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        sqlite_session.add(
            AnalysisRequest(
                id=snapshot.analysis_id,
                startup_name="cmp-1",
                website="https://example.com",
                description="seeded request",
                founder_linkedin_urls=[],
            )
        )
        sqlite_session.add(
            AnalysisReport(
                id="report-lookup",
                request_id=snapshot.analysis_id,
                startup_name="cmp-1",
                venture_score=70.0,
                market_score=70.0,
                founder_score=70.0,
                traction_score=70.0,
                recommendations=[],
                confidence=0.8,
                full_report={},
                engine_version="0.9.9",
            )
        )
        await sqlite_session.flush()

        result = await ForecastService().register(
            sqlite_session, company_id=cid, snapshot_id=snapshot.id, user_id="u1"
        )
        assert result is not None
        assert result.forecast.engine_version == "0.9.9"

    @pytest.mark.asyncio
    async def test_engine_version_falls_back_to_unknown(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        result = await ForecastService().register(
            sqlite_session, company_id=cid, snapshot_id=snapshot.id, user_id="u1"
        )
        assert result is not None
        assert result.forecast.engine_version == "unknown"

    @pytest.mark.asyncio
    async def test_register_due_when_as_of_past_horizon(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        result = await ForecastService().register(
            sqlite_session,
            company_id=cid,
            snapshot_id=snapshot.id,
            user_id="u1",
            as_of=AFTER_HORIZON,
        )
        assert result is not None
        forecast = result.forecast
        assert forecast.status == ForecastStatus.DUE.value

        events = await ForecastService().list_events(sqlite_session, forecast.id)
        assert {e.event_type for e in events} == {
            ForecastEventType.REGISTERED.value,
            ForecastEventType.DUE.value,
        }

    @pytest.mark.asyncio
    async def test_unknown_company_returns_none(self, sqlite_session) -> None:
        result = await ForecastService().register(
            sqlite_session, company_id="nope", snapshot_id="s", user_id="u1"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_out_of_scope_company_returns_none(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-other", user_id="other")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        result = await ForecastService().register(
            sqlite_session, company_id=cid, snapshot_id=snapshot.id, user_id="u1"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_snapshot_not_found_raises(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        with pytest.raises(ForecastSnapshotNotFoundError):
            await ForecastService().register(
                sqlite_session,
                company_id=cid,
                snapshot_id="no-such-snapshot",
                user_id="u1",
            )

    @pytest.mark.asyncio
    async def test_snapshot_of_other_company_raises(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        other = await _seed_company(sqlite_session, name="cmp-2")
        snapshot = await _seed_snapshot(sqlite_session, company_id=other)
        with pytest.raises(ForecastSnapshotNotFoundError):
            await ForecastService().register(
                sqlite_session,
                company_id=cid,
                snapshot_id=snapshot.id,
                user_id="u1",
            )

    @pytest.mark.asyncio
    async def test_invalid_horizon_raises(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        with pytest.raises(ValueError):
            await ForecastService().register(
                sqlite_session,
                company_id=cid,
                snapshot_id=snapshot.id,
                horizon_days=0,
                user_id="u1",
            )

    @pytest.mark.asyncio
    async def test_custom_horizon_stored(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        result = await ForecastService().register(
            sqlite_session,
            company_id=cid,
            snapshot_id=snapshot.id,
            horizon_days=30,
            user_id="u1",
        )
        assert result is not None
        assert result.forecast.horizon_days == 30


class TestReconcile:
    @pytest.mark.asyncio
    async def test_reconcile_attaches_earliest_outcome(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        await _seed_outcome(sqlite_session, company_id=cid, occurred_at=OUTCOME_AT)
        registered = await ForecastService().register(
            sqlite_session, company_id=cid, snapshot_id=snapshot.id, user_id="u1"
        )
        assert registered is not None

        result = await ForecastService().reconcile(
            sqlite_session, registered.forecast.id, user_id="u1"
        )
        assert result is not None
        assert result.reconciled is True
        assert result.outcome_id is not None
        assert result.outcome_verdict == "success"
        assert result.evaluations_created >= 0
        assert result.forecast.outcome_id == result.outcome_id
        assert result.forecast.status == ForecastStatus.RESOLVED.value

        events = await ForecastService().list_events(
            sqlite_session, registered.forecast.id
        )
        types = [e.event_type for e in events]
        assert types[-1] == ForecastEventType.RESOLVED.value
        assert events[-1].payload is not None
        assert events[-1].payload["outcome_id"] == result.outcome_id

    @pytest.mark.asyncio
    async def test_reconcile_picks_earliest_eligible(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        later = await _seed_outcome(
            sqlite_session, company_id=cid, occurred_at=OUTCOME_AT
        )
        earlier = await _seed_outcome(
            sqlite_session, company_id=cid, occurred_at=EARLY_OUTCOME_AT
        )
        assert earlier is not None and later is not None
        registered = await ForecastService().register(
            sqlite_session, company_id=cid, snapshot_id=snapshot.id, user_id="u1"
        )
        assert registered is not None

        result = await ForecastService().reconcile(
            sqlite_session, registered.forecast.id, user_id="u1"
        )
        assert result is not None
        assert result.outcome_id == earlier.outcome.id

    @pytest.mark.asyncio
    async def test_reconcile_ignores_outcome_predating_analysis(
        self, sqlite_session
    ) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        await _seed_outcome(
            sqlite_session,
            company_id=cid,
            occurred_at=SNAPSHOT_AT - timedelta(days=10),
        )
        registered = await ForecastService().register(
            sqlite_session, company_id=cid, snapshot_id=snapshot.id, user_id="u1"
        )
        assert registered is not None

        result = await ForecastService().reconcile(
            sqlite_session, registered.forecast.id, user_id="u1"
        )
        assert result is not None
        assert result.reconciled is False
        assert result.outcome_id is None
        assert result.forecast.status == ForecastStatus.ACTIVE.value

    @pytest.mark.asyncio
    async def test_reconcile_is_idempotent(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        await _seed_outcome(sqlite_session, company_id=cid)
        registered = await ForecastService().register(
            sqlite_session, company_id=cid, snapshot_id=snapshot.id, user_id="u1"
        )
        assert registered is not None
        service = ForecastService()

        first = await service.reconcile(
            sqlite_session, registered.forecast.id, user_id="u1"
        )
        second = await service.reconcile(
            sqlite_session, registered.forecast.id, user_id="u1"
        )
        assert first is not None and second is not None
        assert first.reconciled is True
        assert second.reconciled is False
        assert second.outcome_id == first.outcome_id
        assert second.evaluations_created == 0

        events = await ForecastService().list_events(
            sqlite_session, registered.forecast.id
        )
        resolved = [e for e in events if e.event_type == ForecastEventType.RESOLVED.value]
        assert len(resolved) == 1

    @pytest.mark.asyncio
    async def test_reconcile_unknown_forecast_returns_none(self, sqlite_session) -> None:
        result = await ForecastService().reconcile(
            sqlite_session, "no-such-forecast", user_id="u1"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_reconcile_out_of_scope_returns_none(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-other", user_id="other")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        registered = await ForecastService().register(
            sqlite_session, company_id=cid, snapshot_id=snapshot.id, user_id="other"
        )
        assert registered is not None
        result = await ForecastService().reconcile(
            sqlite_session, registered.forecast.id, user_id="u1"
        )
        assert result is None


class TestListAndGet:
    @pytest.mark.asyncio
    async def test_list_scoped_and_filtered(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1", user_id="u1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        await ForecastService().register(
            sqlite_session, company_id=cid, snapshot_id=snapshot.id, user_id="u1"
        )
        other = await _seed_company(sqlite_session, name="cmp-2", user_id="u2")
        other_snapshot = await _seed_snapshot(sqlite_session, company_id=other)
        await ForecastService().register(
            sqlite_session, company_id=other, snapshot_id=other_snapshot.id, user_id="u2"
        )

        page = await ForecastService().list_forecasts(
            sqlite_session, user_id="u1"
        )
        assert page.total == 1
        assert page.forecasts[0].company_id == cid

        page_for_two = await ForecastService().list_forecasts(
            sqlite_session, user_id="u2"
        )
        assert page_for_two.total == 1
        assert page_for_two.forecasts[0].company_id == other

        filtered = await ForecastService().list_forecasts(
            sqlite_session, user_id="u1", status_filter="active"
        )
        assert filtered.total == 1
        resolved = await ForecastService().list_forecasts(
            sqlite_session, user_id="u1", status_filter="resolved"
        )
        assert resolved.total == 0

        by_company = await ForecastService().list_forecasts(
            sqlite_session, user_id="u1", company_id=cid
        )
        assert by_company.total == 1

    @pytest.mark.asyncio
    async def test_get_respects_scope(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1", user_id="u1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        registered = await ForecastService().register(
            sqlite_session, company_id=cid, snapshot_id=snapshot.id, user_id="u1"
        )
        assert registered is not None
        assert (
            await ForecastService().get(
                sqlite_session, registered.forecast.id, user_id="u1"
            )
        ) is not None
        assert (
            await ForecastService().get(
                sqlite_session, registered.forecast.id, user_id="other"
            )
        ) is None

    @pytest.mark.asyncio
    async def test_reconcile_all_resolves_in_scope(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1", user_id="u1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        registered = await ForecastService().register(
            sqlite_session, company_id=cid, snapshot_id=snapshot.id, user_id="u1"
        )
        assert registered is not None
        await _seed_outcome(sqlite_session, company_id=cid, user_id="u1")

        forecasts, reconciled = await ForecastService().reconcile_all(
            sqlite_session, user_id="u1"
        )
        assert reconciled == 1
        assert forecasts[0].id == registered.forecast.id
        assert forecasts[0].status == ForecastStatus.RESOLVED.value

    @pytest.mark.asyncio
    async def test_summary_counts_and_recent(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1", user_id="u1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        await ForecastService().register(
            sqlite_session, company_id=cid, snapshot_id=snapshot.id, user_id="u1"
        )
        await _seed_outcome(sqlite_session, company_id=cid, user_id="u1")

        total, by_status, outcome_linked, recent = await ForecastService().summary(
            sqlite_session, user_id="u1"
        )
        assert total == 1
        assert by_status["active"] == 1
        assert by_status["due"] == 0
        assert by_status["resolved"] == 0
        assert outcome_linked == 0
        assert len(recent) == 1

        count = (
            await sqlite_session.execute(
                select(func.count()).select_from(Forecast)
            )
        ).scalar_one()
        assert count == 1


class TestOutcomeLinkLifecycle:
    @pytest.mark.asyncio
    async def test_resolution_marks_forecast_resolved(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        await _seed_outcome(sqlite_session, company_id=cid)
        registered = await ForecastService().register(
            sqlite_session, company_id=cid, snapshot_id=snapshot.id, user_id="u1"
        )
        assert registered is not None
        forecast_id = registered.forecast.id

        await ForecastService().reconcile(sqlite_session, forecast_id, user_id="u1")

        raw: Forecast | None = await sqlite_session.get(Forecast, forecast_id)
        assert raw is not None
        assert raw.status == ForecastStatus.RESOLVED.value
        assert raw.outcome_id is not None
        assert raw.reconciled_at is not None
        assert len(await ForecastService().list_events(sqlite_session, forecast_id)) == 2
        types = {
            e.event_type for e in await ForecastService().list_events(sqlite_session, forecast_id)
        }
        assert types == {
            ForecastEventType.REGISTERED.value,
            ForecastEventType.RESOLVED.value,
        }
