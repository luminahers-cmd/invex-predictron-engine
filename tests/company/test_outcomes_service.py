"""Tests for CompanyOutcomeService (Phase 4 validity loop).

Exercises recording, deterministic identity, verdict derivation, idempotent
re-submission, and listing against a real PostgresCompanyStore on an
in-memory SQLite database.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.services.company_outcomes import (
    CompanyOutcomeService,
    OutcomeSnapshotNotFoundError,
    compute_outcome_id,
    outcome_verdict,
)
from app.services.company_postgres import PostgresCompanyStore
from predictron_engine.dataset.outcomes import OutcomeStatus, OutcomeVerdict, StartupOutcome


def _uid() -> str:
    return "u1"


async def _seed_company(session, *, name: str, user_id: str | None = "u1") -> str:
    """Create a company through the real store; returns its stable id."""
    from app.services.companies import CompanyIdentityResolver

    identity = CompanyIdentityResolver().resolve(name, None)
    record = await PostgresCompanyStore().upsert_company(session, identity, user_id=user_id)
    return record.company_id


async def _seed_snapshot(session, *, company_id: str, created_at: datetime):
    from app.services.company_protocols import CompanySnapshotPayload

    payload = CompanySnapshotPayload(
        company_id=company_id,
        analysis_id=f"analysis-{created_at.date()}",
        report_id=f"report-{created_at.date()}",
        decision="invest",
        confidence=0.8,
        composite_score=70.0,
        readiness_score=60.0,
        dimension_scores={"market": 80.0},
        created_at=created_at,
    )
    return await PostgresCompanyStore().append_snapshot(session, payload)


def _success_outcome() -> StartupOutcome:
    return StartupOutcome(
        status=OutcomeStatus.FULLY_VERIFIED,
        acquisition="Acquirer Corp",
        acquisition_price_usd=50_000_000.0,
        exit_type="acquisition",
        latest_verification_date=datetime(2026, 3, 1, tzinfo=UTC),
    )


class TestComputeOutcomeId:
    def test_deterministic(self) -> None:
        outcome = _success_outcome()
        first = compute_outcome_id(
            company_id="c1",
            snapshot_id=None,
            source="manual",
            occurred_at=datetime(2026, 3, 1, tzinfo=UTC),
            outcome=outcome,
        )
        second = compute_outcome_id(
            company_id="c1",
            snapshot_id=None,
            source="manual",
            occurred_at=datetime(2026, 3, 1, tzinfo=UTC),
            outcome=outcome,
        )
        assert first == second
        assert len(first) == 64

    def test_sensitive_to_facts(self) -> None:
        outcome = _success_outcome()
        base = dict(
            company_id="c1",
            snapshot_id=None,
            source="manual",
            occurred_at=datetime(2026, 3, 1, tzinfo=UTC),
            outcome=outcome,
        )
        changed = dict(base, source="filing")
        assert compute_outcome_id(**base) != compute_outcome_id(**changed)


class TestOutcomeVerdict:
    def test_acquisition_is_success(self) -> None:
        assert outcome_verdict(_success_outcome()) == OutcomeVerdict.SUCCESS

    def test_shutdown_is_failure(self) -> None:
        outcome = StartupOutcome(
            status=OutcomeStatus.FULLY_VERIFIED,
            shutdown=True,
            exit_type="shutdown",
        )
        assert outcome_verdict(outcome) == OutcomeVerdict.FAILURE

    def test_unknown_remains_unknown(self) -> None:
        assert outcome_verdict(StartupOutcome()) == OutcomeVerdict.UNKNOWN


class TestRecord:
    @pytest.mark.asyncio
    async def test_record_creates_row_and_derives_verdict(
        self, sqlite_session
    ) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        await _seed_snapshot(
            sqlite_session,
            company_id=cid,
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        result = await CompanyOutcomeService().record(
            sqlite_session, cid, _request(_success_outcome()), user_id=_uid()
        )
        assert result is not None
        assert result.created is True
        assert result.outcome.verdict == "success"
        assert result.outcome.company_id == cid
        # The seeded snapshot predates the outcome, so an evaluation is created.
        assert result.evaluations_created >= 1

    @pytest.mark.asyncio
    async def test_duplicate_record_is_idempotent(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        service = CompanyOutcomeService()
        request = _request(
            _success_outcome(), occurred_at=datetime(2026, 3, 1, tzinfo=UTC)
        )
        first = await service.record(sqlite_session, cid, request, user_id=_uid())
        second = await service.record(sqlite_session, cid, request, user_id=_uid())
        assert first is not None and second is not None
        assert first.created is True
        assert second.created is False
        assert second.outcome.id == first.outcome.id

    @pytest.mark.asyncio
    async def test_unknown_company_returns_none(self, sqlite_session) -> None:
        result = await CompanyOutcomeService().record(
            sqlite_session, "no-such-company", _request(_success_outcome()), user_id=_uid()
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_out_of_scope_company_returns_none(self, sqlite_session) -> None:
        await _seed_company(sqlite_session, name="cmp-of-2", user_id="other")
        result = await CompanyOutcomeService().record(
            sqlite_session, "cmp-of-2", _request(_success_outcome()), user_id=_uid()
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_foreign_snapshot_raises(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        request = _request(_success_outcome(), snapshot_id="foreign-snap")
        with pytest.raises(OutcomeSnapshotNotFoundError):
            await CompanyOutcomeService().record(
                sqlite_session, cid, request, user_id=_uid()
            )

    @pytest.mark.asyncio
    async def test_pinned_snapshot_records_horizon(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        snap = await _seed_snapshot(
            sqlite_session,
            company_id=cid,
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        request = _request(
            _success_outcome(),
            snapshot_id=snap.snapshot.id,
            occurred_at=datetime(2025, 5, 1, tzinfo=UTC),
        )
        result = await CompanyOutcomeService().record(
            sqlite_session, cid, request, user_id=_uid()
        )
        assert result is not None
        assert result.outcome.snapshot_id == snap.snapshot.id
        assert result.outcome.time_horizon_days == 120


class TestList:
    @pytest.mark.asyncio
    async def test_list_returns_newest_first(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        service = CompanyOutcomeService()
        await service.record(
            sqlite_session,
            cid,
            _request(_success_outcome(), occurred_at=datetime(2026, 1, 1, tzinfo=UTC)),
            user_id=_uid(),
        )
        await service.record(
            sqlite_session,
            cid,
            _request(_success_outcome(), occurred_at=datetime(2026, 6, 1, tzinfo=UTC)),
            user_id=_uid(),
        )
        page = await service.list(sqlite_session, cid, user_id=_uid())
        assert page is not None
        assert page.total == 2
        # SQLite round-trips timestamps as naive; compare normalized values.
        assert page.outcomes[0].occurred_at.replace(tzinfo=UTC) == datetime(
            2026, 6, 1, tzinfo=UTC
        )
        assert page.outcomes[1].occurred_at.replace(tzinfo=UTC) == datetime(
            2026, 1, 1, tzinfo=UTC
        )

    @pytest.mark.asyncio
    async def test_list_missing_company_returns_none(self, sqlite_session) -> None:
        page = await CompanyOutcomeService().list(sqlite_session, "nope", user_id=_uid())
        assert page is None


def _request(
    outcome: StartupOutcome,
    *,
    snapshot_id: str | None = None,
    occurred_at: datetime | None = None,
):
    from app.schemas.outcome import OutcomeCreateRequest

    return OutcomeCreateRequest(
        outcome=outcome,
        snapshot_id=snapshot_id,
        occurred_at=occurred_at,
        source="manual",
        notes="test",
    )
