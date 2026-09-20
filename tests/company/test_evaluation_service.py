"""Tests for CompanyEvaluationService (Phase 4 validity loop).

Exercises idempotent reconciliation of outcomes against immutable snapshots,
per-company performance aggregation, and the cross-company calibration and
summary views — all against the real store on an in-memory SQLite database.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.services.company_evaluation import (
    CompanyEvaluationService,
    compute_evaluation_id,
)
from app.services.company_outcomes import CompanyOutcomeService
from app.services.company_postgres import PostgresCompanyStore
from predictron_engine.dataset.outcomes import OutcomeStatus, StartupOutcome

SNAPSHOT_AT = datetime(2025, 1, 1, tzinfo=UTC)
OUTCOME_AT = datetime(2025, 6, 1, tzinfo=UTC)


async def _seed_company(session, *, name: str, user_id: str | None = "u1") -> str:
    from app.services.companies import CompanyIdentityResolver

    identity = CompanyIdentityResolver().resolve(name, None)
    record = await PostgresCompanyStore().upsert_company(session, identity, user_id=user_id)
    return record.company_id


async def _seed_snapshot(session, *, company_id: str, created_at: datetime = SNAPSHOT_AT):
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


async def _record_success_outcome(session, *, company_id: str, occurred_at=OUTCOME_AT):
    from app.schemas.outcome import OutcomeCreateRequest

    outcome = StartupOutcome(
        status=OutcomeStatus.FULLY_VERIFIED,
        acquisition="Acquirer Corp",
        acquisition_price_usd=50_000_000.0,
        exit_type="acquisition",
        latest_verification_date=OUTCOME_AT,
    )
    return await CompanyOutcomeService().record(
        session,
        company_id,
        OutcomeCreateRequest(
            outcome=outcome, occurred_at=occurred_at, source="manual", notes="test"
        ),
        user_id="u1",
    )


async def _insert_outcome_row(session, *, company_id: str, occurred_at=OUTCOME_AT):
    """Insert an outcome row directly, bypassing the auto-reconciling service."""
    from app.models.company_outcome import CompanyOutcome
    from app.services.company_outcomes import compute_outcome_id, outcome_verdict

    outcome = StartupOutcome(
        status=OutcomeStatus.FULLY_VERIFIED,
        acquisition="Acquirer Corp",
        acquisition_price_usd=50_000_000.0,
        exit_type="acquisition",
        latest_verification_date=occurred_at,
    )
    row = CompanyOutcome(
        id=compute_outcome_id(
            company_id=company_id,
            snapshot_id=None,
            source="manual",
            occurred_at=occurred_at,
            outcome=outcome,
        ),
        company_id=company_id,
        snapshot_id=None,
        source="manual",
        occurred_at=occurred_at,
        time_horizon_days=None,
        outcome_data=outcome.model_dump(mode="json"),
        verdict=outcome_verdict(outcome).value,
        verdict_reasoning="",
        status=outcome.status.value,
        notes="test",
    )
    session.add(row)
    await session.flush()
    return row


class TestReconcile:
    @pytest.mark.asyncio
    async def test_creates_evaluation_for_applicable_snapshot(
        self, sqlite_session
    ) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        await _insert_outcome_row(sqlite_session, company_id=cid)

        service = CompanyEvaluationService()
        created = await service.reconcile(sqlite_session, cid, user_id="u1")
        assert created == 1

        perf = await service.performance(sqlite_session, cid, user_id="u1")
        assert perf is not None
        assert perf.evaluation_count == 1
        assert perf.history[0].snapshot_id == snapshot.snapshot.id
        assert perf.history[0].verdict == "correct"
        assert perf.history[0].decision_match is True

    @pytest.mark.asyncio
    async def test_reconcile_is_idempotent(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        await _seed_snapshot(sqlite_session, company_id=cid)
        await _insert_outcome_row(sqlite_session, company_id=cid)

        service = CompanyEvaluationService()
        assert await service.reconcile(sqlite_session, cid, user_id="u1") == 1
        assert await service.reconcile(sqlite_session, cid, user_id="u1") == 0

    @pytest.mark.asyncio
    async def test_outcome_before_snapshot_is_skipped(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        await _seed_snapshot(
            sqlite_session,
            company_id=cid,
            created_at=datetime(2025, 7, 1, tzinfo=UTC),
        )
        await _insert_outcome_row(
            sqlite_session, company_id=cid, occurred_at=OUTCOME_AT
        )
        created = await CompanyEvaluationService().reconcile(
            sqlite_session, cid, user_id="u1"
        )
        assert created == 0

    @pytest.mark.asyncio
    async def test_evaluation_id_is_deterministic(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        snapshot = await _seed_snapshot(sqlite_session, company_id=cid)
        result = await _record_success_outcome(sqlite_session, company_id=cid)

        service = CompanyEvaluationService()
        await service.reconcile(sqlite_session, cid, user_id="u1")
        perf = await service.performance(sqlite_session, cid, user_id="u1")
        expected = compute_evaluation_id(
            company_id=cid,
            snapshot_id=snapshot.snapshot.id,
            outcome_id=result.outcome.id,
        )
        assert perf.history[0].id == expected


class TestPerformance:
    @pytest.mark.asyncio
    async def test_performance_aggregates(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        await _seed_snapshot(sqlite_session, company_id=cid)
        await _record_success_outcome(sqlite_session, company_id=cid)

        perf = await CompanyEvaluationService().performance(
            sqlite_session, cid, user_id="u1"
        )
        assert perf is not None
        assert perf.outcome_count == 1
        assert perf.metrics.scoreable >= 1
        assert perf.calibration.total_samples >= 1
        assert perf.alignment.get("strong_match", 0) >= 1
        assert "correct" in perf.confidence_vs_outcome

    @pytest.mark.asyncio
    async def test_performance_out_of_scope_is_none(self, sqlite_session) -> None:
        await _seed_company(sqlite_session, name="cmp-other", user_id="other")
        perf = await CompanyEvaluationService().performance(
            sqlite_session, "cmp-other", user_id="u1"
        )
        assert perf is None


class TestGlobalViews:
    @pytest.mark.asyncio
    async def test_summary_scopes_to_user(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        await _seed_snapshot(sqlite_session, company_id=cid)
        await _record_success_outcome(sqlite_session, company_id=cid)

        summary = await CompanyEvaluationService().summary(sqlite_session, user_id="u1")
        assert summary.company_count == 1
        assert summary.outcome_count == 1
        assert summary.evaluation_count == 1
        assert summary.by_company[0].company_id == cid
        assert summary.by_company[0].accuracy is not None

        other = await CompanyEvaluationService().summary(sqlite_session, user_id="u2")
        assert other.company_count == 0

    @pytest.mark.asyncio
    async def test_calibration_uses_scoreable_outcomes(self, sqlite_session) -> None:
        cid = await _seed_company(sqlite_session, name="cmp-1")
        await _seed_snapshot(sqlite_session, company_id=cid)
        await _record_success_outcome(sqlite_session, company_id=cid)

        calibration = await CompanyEvaluationService().calibration(
            sqlite_session, user_id="u1"
        )
        assert calibration.company_count == 1
        assert calibration.calibration.total_samples >= 1
