"""MonitoringService — the Phase 6 continuous-intelligence service layer.

Read-only projections over the live prediction ledger.  All analytics are
delegated to the deterministic engine builders in
``predictron_engine.monitoring`` — nothing is recomputed here:

* live :class:`MonitorSnapshot` summary + derived forecast health;
* re-analysis recommendations;
* time-series trends and drift over the append-only snapshot history.

``reconcile`` is delegated to :class:`CompanyEvaluationService` (the
existing idempotent pass) so the monitoring views always aggregate the same
evaluations the validity loop does.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import desc, select

from app.core.config import get_settings
from app.models.company import Company, CompanySnapshot
from app.models.company_outcome import CompanyEvaluation, CompanyOutcome
from app.models.forecast import Forecast
from app.schemas.monitoring import (
    MonitorDriftResponse,
    MonitorHealthListResponse,
    MonitorReanalysisResponse,
    MonitorSummaryResponse,
    MonitorTrendsResponse,
)
from app.services.company_evaluation import prediction_evaluation_from_rows
from app.services.company_outcomes import as_utc
from predictron_engine.dataset.evaluation import PredictionEvaluation
from predictron_engine.monitoring.drift import detect_monitor_drift
from predictron_engine.monitoring.health import (
    build_health_entries,
    health_distribution,
)
from predictron_engine.monitoring.history import MonitorHistory
from predictron_engine.monitoring.models import (
    ForecastHealth,
    ForecastRecord,
    MonitorPeriodKind,
    MonitorSnapshot,
)
from predictron_engine.monitoring.reanalysis import build_reanalysis_recommendations
from predictron_engine.monitoring.snapshot import (
    build_monitor_rolling_snapshot,
    build_monitor_snapshot,
)
from predictron_engine.monitoring.trends import compute_trend

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_SENTINEL = datetime.min.replace(tzinfo=UTC)
_ENGINE_VERSION = "6.0.0"


def _normalized(value: datetime) -> datetime:
    """Normalize any stored timestamp to aware UTC (never ``None``)."""
    converted = as_utc(value)
    if converted is not None:
        return converted.astimezone(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _denormalize_scope(user_id: str | None) -> str:
    """Scope label for namespaced (user) telemetry."""
    return f"user:{user_id}" if user_id is not None else "anonymous"


def _scope_forecast_stmt(stmt, user_id: str | None):
    if user_id is None:
        return stmt.where(Company.user_id.is_(None))
    return stmt.where(Company.user_id == user_id)


class MonitoringService:
    """Derive monitoring telemetry from the live ledger and snapshot files."""

    def __init__(self, *, history: MonitorHistory | None = None) -> None:
        settings = get_settings()
        self._history = history or MonitorHistory(settings.MONITOR_HISTORY_DIR)
        self._stale_after_days = int(settings.MONITOR_STALE_AFTER_DAYS)
        self._reanalysis_max_age_days = int(
            settings.MONITOR_REANALYSIS_SNAPSHOT_MAX_AGE_DAYS
        )
        self._rolling_window = int(settings.MONITOR_DEFAULT_ROLLING_WINDOW)

    # ------------------------------------------------------------------
    # Live projections (user-scoped)
    # ------------------------------------------------------------------

    async def summary(
        self,
        session: AsyncSession,
        *,
        user_id: str | None = None,
        as_of: datetime | None = None,
    ) -> MonitorSummaryResponse:
        """Live repo-wide summary over the caller's scope (full population)."""
        as_of_utc = _normalized(as_of) if as_of is not None else datetime.now(UTC)
        records = await self._forecast_records(session, user_id=user_id)
        evaluations = await self._evaluation_rows(session, user_id=user_id)
        snapshot = build_monitor_snapshot(
            records,
            evaluations,
            anchor_date=as_of_utc.date(),
            period_kind=MonitorPeriodKind.DAILY,
            scope=_denormalize_scope(user_id),
            engine_version=_ENGINE_VERSION,
            recorded_at=as_of_utc,
            as_of=as_of_utc,
        )
        return self._summary_from_snapshot(snapshot)

    async def rolling(
        self,
        session: AsyncSession,
        *,
        user_id: str | None = None,
        as_of: datetime | None = None,
        window: int | None = None,
    ) -> MonitorSummaryResponse:
        """Live summary with rolling-window aggregates over the population."""
        as_of_utc = _normalized(as_of) if as_of is not None else datetime.now(UTC)
        window = window or self._rolling_window
        records = await self._forecast_records(session, user_id=user_id)
        evaluations = await self._evaluation_rows(session, user_id=user_id)
        snapshot = build_monitor_rolling_snapshot(
            records,
            evaluations,
            anchor_date=as_of_utc.date(),
            period_kind=MonitorPeriodKind.DAILY,
            window=window,
            scope=_denormalize_scope(user_id),
            engine_version=_ENGINE_VERSION,
            recorded_at=as_of_utc,
            as_of=as_of_utc,
        )
        return self._summary_from_snapshot(snapshot)

    async def health(
        self,
        session: AsyncSession,
        *,
        user_id: str | None = None,
        as_of: datetime | None = None,
        health_filter: str | None = None,
    ) -> MonitorHealthListResponse:
        """Derived per-forecast health rows for the caller's scope.

        ``health_filter`` (``active | due | overdue | stale | resolved``)
        restricts the returned rows; the ``distribution`` always spans the
        full population so it stays comparable across calls.
        """
        as_of_utc = _normalized(as_of) if as_of is not None else datetime.now(UTC)
        records = await self._forecast_records(session, user_id=user_id)
        entries = build_health_entries(
            records, as_of=as_of_utc, stale_after_days=self._stale_after_days
        )
        scope = _denormalize_scope(user_id)
        if health_filter is not None:
            try:
                wanted = ForecastHealth(health_filter)
            except ValueError as exc:
                raise ValueError(
                    f"unknown health filter {health_filter!r}; expected one of "
                    f"{[item.value for item in ForecastHealth]}"
                ) from exc
            entries = [entry for entry in entries if entry.health == wanted]
        return MonitorHealthListResponse(
            as_of=as_of_utc,
            generated_at=datetime.now(UTC),
            scope=scope,
            entries=entries,
            distribution=health_distribution(
                build_health_entries(
                    records, as_of=as_of_utc, stale_after_days=self._stale_after_days
                )
            ),
        )

    async def reanalysis(
        self,
        session: AsyncSession,
        *,
        user_id: str | None = None,
        as_of: datetime | None = None,
    ) -> MonitorReanalysisResponse:
        """Deterministic re-analysis recommendations for the caller's scope."""
        as_of_utc = _normalized(as_of) if as_of is not None else datetime.now(UTC)
        records = await self._forecast_records(session, user_id=user_id)
        recommendations = build_reanalysis_recommendations(
            records,
            as_of=as_of_utc,
            snapshot_max_age_days=self._reanalysis_max_age_days,
        )
        return MonitorReanalysisResponse(
            as_of=as_of_utc,
            generated_at=datetime.now(UTC),
            scope=_denormalize_scope(user_id),
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Stored history (repo-wide)
    # ------------------------------------------------------------------

    def trends(
        self,
        *,
        period_kind: MonitorPeriodKind = MonitorPeriodKind.DAILY,
        metric_keys: list[str] | None = None,
    ) -> MonitorTrendsResponse:
        """Endpoint-slope trends over the recorded snapshot history."""
        snapshots = self._history.snapshots("repository", period_kind.value)
        keys = metric_keys or [
            "metrics.accuracy",
            "metrics.precision",
            "metrics.recall",
            "metrics.f1",
            "metrics.coverage",
            "calibration.ece",
            "calibration.max_ece",
            "confidence.mean",
        ]
        trends = [
            trend
            for key in keys
            if (trend := compute_trend(snapshots, key)) is not None
        ]
        latest = snapshots[-1].anchor_date if snapshots else datetime.now(UTC).date()
        return MonitorTrendsResponse(
            period_kind=period_kind,
            as_of=latest,
            trends=trends,
        )

    def drift(
        self,
        *,
        before_id: str | None = None,
        after_id: str | None = None,
        period_kind: MonitorPeriodKind = MonitorPeriodKind.DAILY,
    ) -> MonitorDriftResponse:
        """Drift between two recorded snapshots (defaults: latest two).

        Raises ``LookupError`` when fewer than two snapshots exist in the
        period history or when a given id is not present.
        """
        snapshots = self._history.snapshots("repository", period_kind.value)
        by_id = {snapshot.snapshot_id: snapshot for snapshot in snapshots}

        if before_id is None:
            if len(snapshots) < 2:
                raise LookupError(
                    "drift requires at least two recorded snapshots; run "
                    "`predictron-monitor snapshot` on separate days first."
                )
            before, after = snapshots[-2], snapshots[-1]
        else:
            before = by_id.get(before_id)
            if before is None:
                raise LookupError(f"snapshot {before_id} not found in history")
            if after_id is None:
                candidates = [
                    snapshot
                    for snapshot in snapshots
                    if snapshot.snapshot_id != before.snapshot_id
                ]
                if not candidates:
                    raise LookupError(
                        "drift needs a second (different) snapshot; record one "
                        "on another day before comparing."
                    )
                after = candidates[-1]
            else:
                after = by_id.get(after_id)
                if after is None:
                    raise LookupError(f"snapshot {after_id} not found in history")

        report = detect_monitor_drift(before, after)
        return MonitorDriftResponse(
            baseline_id=report.baseline_id,
            comparison_id=report.comparison_id,
            baseline_period=report.baseline_period,
            comparison_period=report.comparison_period,
            signals=report.signals,
        )

    async def repository_snapshot(
        self,
        session: AsyncSession,
        *,
        as_of: datetime | None = None,
        window: int | None = None,
    ) -> MonitorSnapshot:
        """Repository-wide snapshot over the whole platform (CLI-recordable).

        Ignores user scoping — this is the operator-facing view over every
        company in the database; the live API keeps user-scoped telemetry.
        """
        as_of_utc = _normalized(as_of) if as_of is not None else datetime.now(UTC)
        records = await self._forecast_records(session, user_id=None, scope_all=True)
        evaluations = await self._evaluation_rows(session, user_id=None, scope_all=True)
        builder = (
            build_monitor_rolling_snapshot
            if window is not None
            else build_monitor_snapshot
        )
        kwargs: dict[str, object] = {}
        if window is not None:
            kwargs["window"] = window
        snapshot = builder(
            records,
            evaluations,
            anchor_date=as_of_utc.date(),
            period_kind=MonitorPeriodKind.DAILY,
            scope="repository",
            engine_version=_ENGINE_VERSION,
            recorded_at=as_of_utc,
            as_of=as_of_utc,
            **kwargs,
        )
        return snapshot

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    async def _forecast_records(
        self,
        session: AsyncSession,
        *,
        user_id: str | None,
        scope_all: bool = False,
    ) -> list[ForecastRecord]:
        stmt = select(Forecast).join(Company, Company.company_id == Forecast.company_id)
        if not scope_all:
            stmt = _scope_forecast_stmt(stmt, user_id)
        forecasts = list((await session.execute(stmt)).scalars().all())

        evaluation_lookup = await self._snapshot_evaluation_lookup(
            session, user_id, scope_all=scope_all
        )
        outcome_verdicts = await self._outcome_verdict_lookup(
            session, user_id, scope_all=scope_all
        )
        companies = {row.company_id for row in forecasts}
        latest_outcome = await self._latest_outcome_at(session, companies)
        latest_snapshot = await self._latest_snapshot_at(session, companies)

        records: list[ForecastRecord] = []
        for forecast in forecasts:
            eval_meta = evaluation_lookup.get(forecast.snapshot_id)
            records.append(
                ForecastRecord(
                    id=forecast.id,
                    company_id=forecast.company_id,
                    snapshot_id=forecast.snapshot_id,
                    decision=forecast.decision,
                    confidence=float(forecast.confidence),
                    composite_score=float(forecast.composite_score),
                    status=forecast.status,
                    analysis_timestamp=_normalized(forecast.analysis_timestamp),
                    due_at=_normalized(forecast.due_at),
                    horizon_days=forecast.horizon_days,
                    outcome_id=forecast.outcome_id,
                    outcome_verdict=outcome_verdicts.get(forecast.outcome_id or ""),
                    evaluation_verdict=eval_meta[0] if eval_meta else None,
                    evaluation_created_at=eval_meta[1] if eval_meta else None,
                    latest_outcome_at=latest_outcome.get(forecast.company_id),
                    latest_snapshot_at=latest_snapshot.get(forecast.company_id),
                )
            )
        return records

    async def _evaluation_rows(
        self,
        session: AsyncSession,
        *,
        user_id: str | None,
        scope_all: bool = False,
    ) -> list[PredictionEvaluation]:
        """Evaluations in scope, rebuilt through the canonical row mapper."""
        stmt = (
            select(CompanyEvaluation, CompanyOutcome)
            .join(
                CompanyOutcome,
                CompanyOutcome.id == CompanyEvaluation.outcome_id,
            )
            .join(Company, Company.company_id == CompanyEvaluation.company_id)
            .order_by(
                desc(CompanyEvaluation.created_at), desc(CompanyEvaluation.id)
            )
        )
        if not scope_all:
            stmt = _scope_forecast_stmt(stmt, user_id)
        rows = (await session.execute(stmt)).all()
        return [
            prediction_evaluation_from_rows(evaluation, outcome)
            for evaluation, outcome in rows
        ]

    async def _snapshot_evaluation_lookup(
        self,
        session: AsyncSession,
        user_id: str | None,
        *,
        scope_all: bool = False,
    ) -> dict[str, tuple[str, datetime]]:
        """Latest evaluation per ``snapshot_id``: ``(verdict, created_at)``."""
        stmt = select(CompanyEvaluation).join(
            Company, Company.company_id == CompanyEvaluation.company_id
        )
        if not scope_all:
            stmt = _scope_forecast_stmt(stmt, user_id)
        rows = list((await session.execute(stmt)).scalars().all())
        lookup: dict[str, tuple[str, datetime]] = {}
        for row in rows:
            existing = lookup.get(row.snapshot_id)
            row_time = _normalized(row.created_at)
            if existing is None or (row_time, row.id) > (existing[1], ""):
                lookup[row.snapshot_id] = (row.verdict, row_time)
        return lookup

    async def _outcome_verdict_lookup(
        self,
        session: AsyncSession,
        user_id: str | None,
        *,
        scope_all: bool = False,
    ) -> dict[str, str]:
        stmt = select(CompanyOutcome).join(
            Company, Company.company_id == CompanyOutcome.company_id
        )
        if not scope_all:
            stmt = _scope_forecast_stmt(stmt, user_id)
        rows = list((await session.execute(stmt)).scalars().all())
        return {row.id: row.verdict for row in rows}

    async def _latest_outcome_at(
        self, session: AsyncSession, companies: set[str]
    ) -> dict[str, datetime]:
        if not companies:
            return {}
        stmt = select(
            CompanyOutcome.company_id, CompanyOutcome.occurred_at
        ).where(CompanyOutcome.company_id.in_(companies))
        rows = (await session.execute(stmt)).all()
        latest: dict[str, datetime] = {}
        for company_id, occurred_at in rows:
            current = latest.get(company_id)
            value = _normalized(occurred_at)
            if current is None or value > current:
                latest[company_id] = value
        return latest

    async def _latest_snapshot_at(
        self, session: AsyncSession, companies: set[str]
    ) -> dict[str, datetime]:
        if not companies:
            return {}
        stmt = select(
            CompanySnapshot.company_id, CompanySnapshot.created_at
        ).where(CompanySnapshot.company_id.in_(companies))
        rows = (await session.execute(stmt)).all()
        latest: dict[str, datetime] = {}
        for company_id, created_at in rows:
            current = latest.get(company_id)
            value = _normalized(created_at)
            if current is None or value > current:
                latest[company_id] = value
        return latest

    def _summary_from_snapshot(
        self, snapshot: MonitorSnapshot
    ) -> MonitorSummaryResponse:
        return MonitorSummaryResponse(
            scope=snapshot.scope,
            period_kind=snapshot.period_kind,
            anchor_date=snapshot.anchor_date,
            generated_at=snapshot.recorded_at,
            counts=snapshot.counts,
            metrics=snapshot.metrics,
            distributions=snapshot.distributions,
            horizon_breakdown=snapshot.horizon_breakdown,
            sector_breakdown=snapshot.sector_breakdown,
            health=snapshot.health,
        )


__all__ = ["MonitoringService"]
