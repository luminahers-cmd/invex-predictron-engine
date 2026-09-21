"""ForecastService — the Live Prediction Ledger service layer (CIH Phase 5).

The ledger is deterministic and append-only:

* ``register`` derives the forecast primary key from ``(company_id,
  snapshot_id, horizon_days)`` so re-registration is an idempotent no-op.
  The frozen :class:`PredictionSummary` captured from the immutable snapshot
  is stored once and never referenced after registration; ``engine_version``
  and ``schema_version`` are immutable row metadata pinned at registration.
* ``status`` is always derived by
  :func:`~predictron_engine.dataset.forecast_lifecycle.derive_forecast_status`
  and then persisted — derive -> persist, never trusted from a caller.
* ``reconcile`` attaches the earliest eligible outcome (``occurred_at >=
  analysis_timestamp``, tie-broken by outcome id), records a deterministic
  ``resolved`` event, and triggers :class:`CompanyEvaluationService` so the
  existing evaluation machinery is reused verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import desc, func, select

from app.models.company import Company, CompanySnapshot
from app.models.company_outcome import CompanyOutcome
from app.models.forecast import Forecast, ForecastEvent
from app.services.company_evaluation import prediction_summary_from_snapshot
from app.services.company_outcomes import as_utc
from app.services.company_postgres import PostgresCompanyStore
from predictron_engine.dataset.forecast_lifecycle import (
    FORECAST_SCHEMA_VERSION,
    ForecastEventType,
    ForecastStatus,
    derive_forecast_status,
    forecast_due_at,
    stable_event_id,
    stable_forecast_id,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_SENTINEL = datetime.min.replace(tzinfo=UTC)


def _normalized(value: datetime) -> datetime:
    """Normalize a datetime to aware UTC (never returns None)."""
    converted = as_utc(value)
    if converted is not None:
        return converted
    return value


class ForecastSnapshotNotFoundError(LookupError):
    """Raised when a requested ``snapshot_id`` does not belong to the company."""


@dataclass(frozen=True)
class ForecastWriteResult:
    """Result of registering one forecast plus its idempotency info."""

    forecast: Forecast
    created: bool


@dataclass(frozen=True)
class ForecastPageResult:
    """One page of forecasts (newest-first) and the total count."""

    forecasts: list[Forecast]
    total: int


@dataclass(frozen=True)
class ForecastReconcileResult:
    """Result of reconciling one forecast against observed outcomes."""

    forecast: Forecast
    reconciled: bool
    outcome_id: str | None
    outcome_verdict: str | None
    evaluations_created: int


class ForecastService:
    """Persist and list deterministic forecasts (stateless; session per call)."""

    def __init__(self, *, store: PostgresCompanyStore | None = None) -> None:
        self._store = store or PostgresCompanyStore()

    async def register(
        self,
        session: AsyncSession,
        *,
        company_id: str,
        snapshot_id: str,
        horizon_days: int | None = None,
        engine_version: str | None = None,
        notes: str = "",
        user_id: str | None = None,
        as_of: datetime | None = None,
    ) -> ForecastWriteResult | None:
        """Register one forecast, idempotently.

        Returns ``None`` when the company is not found in the caller's scope;
        raises :class:`ForecastSnapshotNotFoundError` when ``snapshot_id``
        does not belong to the company. A duplicate registration (same
        deterministic id) is returned with ``created=False`` and never
        overwrites the frozen prediction or version metadata.
        """
        company = await self._store.get_company(session, company_id, user_id=user_id)
        if company is None:
            return None

        snapshot = await self._load_snapshot(session, company_id, snapshot_id)
        if snapshot is None:
            raise ForecastSnapshotNotFoundError(
                f"Snapshot {snapshot_id} not found for company {company_id}"
            )

        if horizon_days is None:
            horizon = self._default_horizon_days()
        else:
            horizon = horizon_days
        if horizon <= 0:
            raise ValueError("horizon_days must be greater than zero")

        analysis_timestamp = as_utc(snapshot.created_at) or _SENTINEL
        forecast_id = stable_forecast_id(
            company_id=company_id,
            snapshot_id=snapshot_id,
            horizon_days=horizon,
        )
        effective_at = _normalized(as_of) if as_of is not None else datetime.now(UTC)

        existing = await session.get(Forecast, forecast_id)
        if existing is not None:
            await self._persist_derived_status(session, existing, as_of=effective_at)
            return ForecastWriteResult(forecast=existing, created=False)

        resolved_engine = (
            engine_version
            or await self._lookup_engine_version(session, snapshot.analysis_id)
            or "unknown"
        )
        prediction = prediction_summary_from_snapshot(snapshot)
        due_at = forecast_due_at(
            analysis_timestamp=analysis_timestamp, horizon_days=horizon
        )
        status = derive_forecast_status(
            analysis_timestamp=analysis_timestamp,
            horizon_days=horizon,
            as_of=effective_at,
        )

        row = Forecast(
            id=forecast_id,
            company_id=company_id,
            snapshot_id=snapshot_id,
            outcome_id=None,
            analysis_timestamp=analysis_timestamp,
            due_at=due_at,
            horizon_days=horizon,
            status=status.value,
            decision=snapshot.decision or prediction.decision.value,
            confidence=float(snapshot.confidence or 0.0),
            composite_score=float(snapshot.composite_score or 0.0),
            prediction=prediction.model_dump(mode="json"),
            engine_version=resolved_engine,
            schema_version=FORECAST_SCHEMA_VERSION,
            notes=notes,
        )
        session.add(row)
        session.add(
            ForecastEvent(
                id=stable_event_id(
                    forecast_id=forecast_id, event_type=ForecastEventType.REGISTERED
                ),
                forecast_id=forecast_id,
                event_type=ForecastEventType.REGISTERED.value,
                occurred_at=effective_at,
                payload=None,
            )
        )
        if status is ForecastStatus.DUE:
            session.add(
                ForecastEvent(
                    id=stable_event_id(
                        forecast_id=forecast_id, event_type=ForecastEventType.DUE
                    ),
                    forecast_id=forecast_id,
                    event_type=ForecastEventType.DUE.value,
                    occurred_at=due_at,
                    payload=None,
                )
            )
        await session.flush()
        return ForecastWriteResult(forecast=row, created=True)

    async def get(
        self,
        session: AsyncSession,
        forecast_id: str,
        *,
        user_id: str | None = None,
    ) -> Forecast | None:
        """Fetch one forecast, scoped to the caller's namespace."""
        return await self._scoped_forecast(session, forecast_id, user_id=user_id)

    async def list_forecasts(
        self,
        session: AsyncSession,
        *,
        user_id: str | None = None,
        company_id: str | None = None,
        status_filter: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> ForecastPageResult:
        """List forecasts newest-first, scoped and optionally filtered."""
        count_over = func.count().over().label("total_count")
        stmt = (
            select(Forecast, count_over)
            .join(Company, Company.company_id == Forecast.company_id)
            .order_by(desc(Forecast.created_at), desc(Forecast.id))
            .offset(offset)
            .limit(limit)
        )
        stmt = self._scope_company(stmt, user_id)
        if company_id is not None:
            stmt = stmt.where(Forecast.company_id == company_id)
        if status_filter is not None:
            stmt = stmt.where(Forecast.status == status_filter)

        rows = (await session.execute(stmt)).all()
        total = rows[0].total_count if rows else 0
        return ForecastPageResult(forecasts=[row for row, _ in rows], total=total)

    async def list_events(
        self, session: AsyncSession, forecast_id: str
    ) -> list[ForecastEvent]:
        """Append-only event log for one forecast, oldest-first."""
        stmt = (
            select(ForecastEvent)
            .where(ForecastEvent.forecast_id == forecast_id)
            .order_by(ForecastEvent.occurred_at, ForecastEvent.id)
        )
        return list((await session.execute(stmt)).scalars().all())

    async def reconcile(
        self,
        session: AsyncSession,
        forecast_id: str,
        *,
        user_id: str | None = None,
        as_of: datetime | None = None,
    ) -> ForecastReconcileResult | None:
        """Attach the earliest eligible outcome to one forecast (idempotent).

        Returns ``None`` when the forecast is not found in the caller's
        scope. Re-attaching an already-matched outcome is a no-op: the
        deterministic ``resolved`` event id collapses onto the existing row.
        """
        forecast = await self._scoped_forecast(session, forecast_id, user_id=user_id)
        if forecast is None:
            return None

        effective_at = _normalized(as_of) if as_of is not None else datetime.now(UTC)
        outcome = await self._matched_outcome(session, forecast)
        reconciled = False
        if outcome is not None and forecast.outcome_id is None:
            forecast.outcome_id = outcome.id
            forecast.reconciled_at = effective_at
            session.add(forecast)
            await session.flush()
            reconciled = True

        outcome_id = outcome.id if outcome is not None else None
        outcome_verdict = outcome.verdict if outcome is not None else None
        if outcome is not None:
            await self._ensure_event(
                session,
                forecast.id,
                ForecastEventType.RESOLVED,
                occurred_at=effective_at,
                outcome_id=outcome.id,
                payload={
                    "outcome_id": outcome.id,
                    "outcome_verdict": outcome.verdict,
                },
            )

        await self._persist_derived_status(session, forecast, as_of=effective_at)

        evaluations_created = 0
        if outcome is not None:
            from app.services.company_evaluation import CompanyEvaluationService

            evaluations_created = await CompanyEvaluationService(
                store=self._store
            ).reconcile(session, forecast.company_id, user_id=user_id)

        return ForecastReconcileResult(
            forecast=forecast,
            reconciled=reconciled,
            outcome_id=outcome_id,
            outcome_verdict=outcome_verdict,
            evaluations_created=evaluations_created,
        )

    async def reconcile_all(
        self,
        session: AsyncSession,
        *,
        user_id: str | None = None,
        as_of: datetime | None = None,
    ) -> tuple[list[Forecast], int]:
        """Reconcile every forecast in the caller's scope.

        Returns the (possibly updated) in-scope forecasts and the number
        that were newly matched to an outcome this pass.
        """
        forecasts: list[Forecast] = []
        freshly_resolved: set[str] = set()
        offset = 0
        page_size = 100
        while True:
            page = await self.list_forecasts(
                session, user_id=user_id, offset=offset, limit=page_size
            )
            for forecast in page.forecasts:
                result = await self.reconcile(
                    session, forecast.id, user_id=user_id, as_of=as_of
                )
                if result is not None and result.reconciled:
                    freshly_resolved.add(forecast.id)
            forecasts.extend(page.forecasts)
            if len(page.forecasts) < page_size:
                break
            offset += page_size
        return forecasts, len(freshly_resolved)

    async def summary(
        self,
        session: AsyncSession,
        *,
        user_id: str | None = None,
        recent_limit: int = 5,
    ) -> tuple[int, dict[str, int], int, list[Forecast]]:
        """Counts plus the most recent forecasts for the caller's scope."""
        scoped = self._scope_company(
            select(func.count()).select_from(Forecast).join(
                Company, Company.company_id == Forecast.company_id
            ),
            user_id,
        )
        total = int((await session.execute(scoped)).scalar_one())

        by_status: dict[str, int] = {}
        for status in ForecastStatus:
            stmt = self._scope_company(
                select(func.count())
                .select_from(Forecast)
                .join(Company, Company.company_id == Forecast.company_id)
                .where(Forecast.status == status.value),
                user_id,
            )
            by_status[status.value] = int((await session.execute(stmt)).scalar_one())

        linked_stmt = self._scope_company(
            select(func.count())
            .select_from(Forecast)
            .join(Company, Company.company_id == Forecast.company_id)
            .where(Forecast.outcome_id.is_not(None)),
            user_id,
        )
        outcome_linked = int((await session.execute(linked_stmt)).scalar_one())

        page = await self.list_forecasts(session, user_id=user_id, offset=0, limit=recent_limit)
        return total, by_status, outcome_linked, page.forecasts

    # ---- Internal helpers ----

    def _default_horizon_days(self) -> int:
        from app.core.config import get_settings

        return int(get_settings().FORECAST_DEFAULT_HORIZON_DAYS)

    async def _load_snapshot(
        self, session: AsyncSession, company_id: str, snapshot_id: str
    ) -> CompanySnapshot | None:
        stmt = select(CompanySnapshot).where(
            CompanySnapshot.id == snapshot_id,
            CompanySnapshot.company_id == company_id,
        )
        return (await session.execute(stmt)).scalar_one_or_none()

    async def _lookup_engine_version(
        self, session: AsyncSession, analysis_id: str
    ) -> str | None:
        from app.models.analysis import AnalysisReport

        stmt = select(AnalysisReport.engine_version).where(
            AnalysisReport.request_id == analysis_id
        )
        return (await session.execute(stmt)).scalar_one_or_none()

    async def _scoped_forecast(
        self, session: AsyncSession, forecast_id: str, *, user_id: str | None
    ) -> Forecast | None:
        stmt = (
            select(Forecast)
            .join(Company, Company.company_id == Forecast.company_id)
            .where(Forecast.id == forecast_id)
        )
        stmt = self._scope_company(stmt, user_id)
        return (await session.execute(stmt)).scalar_one_or_none()

    async def _matched_outcome(
        self, session: AsyncSession, forecast: Forecast
    ) -> CompanyOutcome | None:
        if forecast.outcome_id is not None:
            return await session.get(CompanyOutcome, forecast.outcome_id)
        stmt = select(CompanyOutcome).where(
            CompanyOutcome.company_id == forecast.company_id
        )
        rows = list((await session.execute(stmt)).scalars().all())
        analysis_time = as_utc(forecast.analysis_timestamp) or _SENTINEL
        eligible = [
            row
            for row in rows
            if (as_utc(row.occurred_at) or _SENTINEL) >= analysis_time
        ]
        if not eligible:
            return None
        return sorted(
            eligible,
            key=lambda row: (as_utc(row.occurred_at) or _SENTINEL, row.id),
        )[0]

    async def _persist_derived_status(
        self, session: AsyncSession, forecast: Forecast, *, as_of: datetime
    ) -> None:
        """Derive the forecast status and persist it if it changed (derive->persist)."""
        status = derive_forecast_status(
            analysis_timestamp=as_utc(forecast.analysis_timestamp) or _SENTINEL,
            horizon_days=forecast.horizon_days,
            as_of=as_of,
            resolved=forecast.outcome_id is not None,
        )
        derived = status.value
        if derived != forecast.status:
            forecast.status = derived
            session.add(forecast)
            await session.flush()
        if status is ForecastStatus.DUE:
            due_at = as_utc(forecast.due_at) or as_of
            await self._ensure_event(
                session, forecast.id, ForecastEventType.DUE, occurred_at=due_at
            )

    async def _ensure_event(
        self,
        session: AsyncSession,
        forecast_id: str,
        event_type: ForecastEventType,
        *,
        occurred_at: datetime,
        outcome_id: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        """Record one lifecycle event only if it does not exist (idempotent)."""
        event_id = stable_event_id(
            forecast_id=forecast_id, event_type=event_type, outcome_id=outcome_id
        )
        existing = await session.get(ForecastEvent, event_id)
        if existing is None:
            session.add(
                ForecastEvent(
                    id=event_id,
                    forecast_id=forecast_id,
                    event_type=event_type.value,
                    occurred_at=occurred_at,
                    payload=payload,
                )
            )
            await session.flush()

    def _scope_company(self, stmt, user_id: str | None):
        if user_id is None:
            return stmt.where(Company.user_id.is_(None))
        return stmt.where(Company.user_id == user_id)


__all__ = [
    "ForecastPageResult",
    "ForecastReconcileResult",
    "ForecastService",
    "ForecastSnapshotNotFoundError",
    "ForecastWriteResult",
]
