"""Phase 6 — prediction health derivation.

Pure, deterministic projection of the persisted forecast lifecycle onto the
monitoring health vocabulary (``active`` / ``due`` / ``overdue`` / ``stale``
/ ``resolved``).  The base ``due`` / ``active`` / ``resolved`` classification
is **not** re-implemented here: it delegates to the canonical
:func:`~predictron_engine.dataset.forecast_lifecycle.derive_forecast_status`
(the single service-owned lifecycle derivation).  ``overdue`` and ``stale``
are the only additions, and both are defined in terms of named constants.

Mathematical definitions (deterministic)
----------------------------------------
Let ``as_of`` be the analysis moment (UTC) and ``MONITOR_DEFAULT_STALE_AFTER_DAYS``
the named staleness threshold.  For one forecast with outcome ``o``,
analysis time ``t``, due time ``d``, horizon ``h``:

* ``age_days = max(0, (as_of - t).days)``
* ``days_until_due = (d - as_of) / 86400`` (reuses ``PredictionForecast.days_until_due``)
* ``days_overdue = max(0, -(d - as_of) / 86400)``
* health precedence (highest to lowest):
  1. ``resolved`` iff ``o`` is attached,
  2. ``overdue`` iff ``as_of >= d`` (the horizon has fully elapsed),
  3. ``stale`` iff ``age_days > MONITOR_DEFAULT_STALE_AFTER_DAYS``,
  4. otherwise the canonical lifecycle status: ``due`` / ``active``.

No heuristic or hidden threshold exists: every rule above references either
the canonical lifecycle derivation or a named module constant.
"""

from __future__ import annotations

from datetime import UTC, datetime

from predictron_engine.dataset.forecast_lifecycle import (
    ForecastStatus,
    derive_forecast_status,
)
from predictron_engine.monitoring.models import (
    MONITOR_DEFAULT_STALE_AFTER_DAYS,
    ForecastHealth,
    ForecastRecord,
    HealthEntry,
)

DEFAULT_STALE_AFTER_DAYS = MONITOR_DEFAULT_STALE_AFTER_DAYS


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def derive_forecast_health(
    *,
    horizon_days: int,
    due_at: datetime,
    analysis_timestamp: datetime,
    resolved: bool,
    as_of: datetime,
    stale_after_days: int = DEFAULT_STALE_AFTER_DAYS,
) -> ForecastHealth:
    """Derive the monitoring health state of one forecast at ``as_of``.

    The base lifecycle state is obtained through the canonical
    :func:`derive_forecast_status` with the stored horizon; the additive
    ``overdue`` / ``stale`` states layer on top deterministically.
    """
    status_value = derive_forecast_status(
        analysis_timestamp=analysis_timestamp,
        horizon_days=horizon_days,
        as_of=as_of,
        resolved=resolved,
    )

    as_of_utc = _as_utc(as_of)
    due_at_utc = _as_utc(due_at)
    if resolved:
        return ForecastHealth.RESOLVED
    if as_of_utc >= due_at_utc:
        return ForecastHealth.OVERDUE
    age_days = max(0, (as_of_utc - _as_utc(analysis_timestamp)).days)
    if age_days > stale_after_days:
        return ForecastHealth.STALE
    if status_value is ForecastStatus.DUE:
        return ForecastHealth.DUE
    return ForecastHealth.ACTIVE


def build_health_entries(
    records: list[ForecastRecord],
    *,
    as_of: datetime,
    stale_after_days: int = DEFAULT_STALE_AFTER_DAYS,
) -> list[HealthEntry]:
    """Project a list of ledger forecasts onto health rows (deterministic).

    Ordered by ``(due_at, forecast_id)`` so overdue forecasts surface first.
    """
    entries: list[HealthEntry] = []
    as_of_utc = _as_utc(as_of)
    for record in records:
        resolved = record.outcome_id is not None
        due_at_utc = _as_utc(record.due_at)
        a_utc = _as_utc(record.analysis_timestamp)
        health = derive_forecast_health(
            horizon_days=record.horizon_days,
            due_at=record.due_at,
            analysis_timestamp=record.analysis_timestamp,
            resolved=resolved,
            as_of=as_of,
            stale_after_days=stale_after_days,
        )
        age_days = max(0, (as_of_utc - a_utc).days)
        seconds_until_due = (due_at_utc - as_of_utc).total_seconds()
        days_until_due = round(seconds_until_due / 86400.0, 4)
        days_overdue = round(max(0.0, -seconds_until_due) / 86400.0, 4)
        entries.append(
            HealthEntry(
                forecast_id=record.id,
                company_id=record.company_id,
                snapshot_id=record.snapshot_id,
                decision=record.decision,
                confidence=record.confidence,
                status=record.status,
                health=health,
                analysis_timestamp=record.analysis_timestamp,
                due_at=record.due_at,
                as_of=as_of_utc,
                age_days=age_days,
                days_until_due=days_until_due,
                days_overdue=days_overdue,
                outcome_id=record.outcome_id,
                outcome_verdict=record.outcome_verdict,
                evaluation_status=(
                    "evaluated"
                    if record.evaluation_verdict is not None
                    else "pending"
                ),
                evaluation_verdict=record.evaluation_verdict,
            )
        )
    entries.sort(key=lambda entry: (entry.due_at, entry.forecast_id))
    return entries


def health_distribution(entries: list[HealthEntry]) -> dict[str, int]:
    """Distribution of health states over the projected forecasts."""
    counts: dict[str, int] = {}
    for entry in entries:
        key = entry.health.value
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "DEFAULT_STALE_AFTER_DAYS",
    "build_health_entries",
    "derive_forecast_health",
    "health_distribution",
]
