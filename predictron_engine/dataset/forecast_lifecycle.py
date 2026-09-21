"""Deterministic forecast lifecycle primitives (Phase 5 — Live Prediction Ledger).

Pure, additive machinery shared by the forecast service, the migration, and
the API layer:

* :class:`ForecastStatus` / :class:`ForecastEventType` — the explicit,
  service-owned lifecycle vocabulary;
* :func:`stable_forecast_id` / :func:`stable_event_id` — deterministic
  SHA-256 identities so re-registration and re-reconciliation collapse onto
  the same rows;
* :func:`forecast_due_at` / :func:`derive_forecast_status` — the pure
  status derivation reused by every persistence path (derive -> persist).

No storage is touched here; nothing fabricates outcomes. Status precedence:
resolved wins, then due (once the horizon window — including the tolerance
grace mirroring :class:`ForecastHorizon`) has elapsed, otherwise active.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from enum import Enum

from predictron_engine.dataset.forecast import ForecastHorizon

FORECAST_SCHEMA_VERSION = "1.0"

_ID_SEPARATOR = "\x00"


class ForecastStatus(str, Enum):
    """Lifecycle status of one persisted forecast (service-owned, derived)."""

    ACTIVE = "active"
    DUE = "due"
    RESOLVED = "resolved"


class ForecastEventType(str, Enum):
    """Append-only lifecycle events recorded against a forecast."""

    REGISTERED = "registered"
    DUE = "due"
    RESOLVED = "resolved"


def stable_forecast_id(*, company_id: str, snapshot_id: str, horizon_days: int) -> str:
    """Deterministic forecast ID derived from (company, snapshot, horizon).

    Two registrations describing the same analysis and horizon always yield
    the same primary key, which is what makes forecast registration
    idempotent.
    """
    material = _ID_SEPARATOR.join(
        [company_id, snapshot_id, str(int(horizon_days))]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def stable_event_id(
    *,
    forecast_id: str,
    event_type: ForecastEventType | str,
    outcome_id: str | None = None,
) -> str:
    """Deterministic event ID for one lifecycle event.

    ``resolved`` events include the attached ``outcome_id`` so re-running a
    reconciliation is an idempotent no-op instead of a duplicate event.
    """
    label = (
        event_type.value if isinstance(event_type, ForecastEventType) else event_type
    )
    material = _ID_SEPARATOR.join([forecast_id, label, outcome_id or ""])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def forecast_due_at(*, analysis_timestamp: datetime, horizon_days: int) -> datetime:
    """Earliest moment a matching outcome observation is due."""
    return ForecastHorizon(days=horizon_days, tolerance_days=0).due_at(
        analysis_timestamp
    )


def forecast_earliest_due_detection(
    *, analysis_timestamp: datetime, horizon_days: int, tolerance_days: int = 1
) -> datetime:
    """Earliest moment the forecast becomes ``due`` (horizon minus grace)."""
    return forecast_due_at(
        analysis_timestamp=analysis_timestamp, horizon_days=horizon_days
    ) - timedelta(days=tolerance_days)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def derive_forecast_status(
    *,
    analysis_timestamp: datetime,
    horizon_days: int,
    as_of: datetime,
    resolved: bool = False,
    tolerance_days: int = 1,
) -> ForecastStatus:
    """Derive the forecast status at ``as_of`` (pure, deterministic).

    Precedence: a resolved forecast stays ``resolved``; otherwise the
    forecast is ``due`` once ``as_of`` reaches the horizon minus the
    tolerance grace (mirroring ``PredictionForecast.is_due``); otherwise it
    is ``active``.
    """
    if resolved:
        return ForecastStatus.RESOLVED
    earliest = forecast_earliest_due_detection(
        analysis_timestamp=analysis_timestamp,
        horizon_days=horizon_days,
        tolerance_days=tolerance_days,
    )
    if _as_utc(as_of) >= _as_utc(earliest):
        return ForecastStatus.DUE
    return ForecastStatus.ACTIVE


__all__ = [
    "FORECAST_SCHEMA_VERSION",
    "ForecastEventType",
    "ForecastStatus",
    "derive_forecast_status",
    "forecast_due_at",
    "forecast_earliest_due_detection",
    "stable_event_id",
    "stable_forecast_id",
]
