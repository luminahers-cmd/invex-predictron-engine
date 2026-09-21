"""Phase 6 — deterministic re-analysis recommendations.

A frozen prediction (forecast pinned to a snapshot) is recommended for
re-analysis for exactly four named reasons, each defined by a deterministic
predicate over stored fields.  No heuristic exists.

Mathematical definitions
------------------------
For a forecast with analysis time ``t``, outcome ``o``, due time ``d``,
evaluation time ``e``, and latest outcome/snapshot times ``lo`` / ``ls``:

* ``new_outcome_recorded`` — ``o`` exists and the forecast is not yet
  evaluated (``e is None``).
* ``evidence_changed`` — the forecast is evaluated but a newer outcome was
  observed after the evaluation (``lo > e``) or a newer snapshot exists
  (``ls > e``).  Postcondition: re-snapshot + re-evaluate would change
  evidence.
* ``prediction_expired`` — the horizon has fully elapsed (``as_of >= d``)
  with no outcome attached (``o is None``).
* ``snapshot_age_exceeded`` — the frozen snapshot is older than
  ``MONITOR_REANALYSIS_SNAPSHOT_MAX_AGE_DAYS`` at ``as_of``.

A concluded forecast (outcome attached and evaluated) never triggers a
recommendation.  Results are ordered by deterministic priority
(``prediction_expired`` > ``new_outcome_recorded`` > ``evidence_changed`` >
``snapshot_age_exceeded``) then by ``forecast_id``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from predictron_engine.monitoring.models import (
    MONITOR_REANALYSIS_SNAPSHOT_MAX_AGE_DAYS,
    ForecastRecord,
    ReanalysisReason,
    ReanalysisRecommendation,
)

_CANONICAL_ORDER: tuple[ReanalysisReason, ...] = (
    ReanalysisReason.PREDICTION_EXPIRED,
    ReanalysisReason.NEW_OUTCOME_RECORDED,
    ReanalysisReason.EVIDENCE_CHANGED,
    ReanalysisReason.SNAPSHOT_AGE_EXCEEDED,
)

_PRIORITY = {
    reason: index for index, reason in enumerate(_CANONICAL_ORDER)
}


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def build_reanalysis_recommendations(
    forecasts: list[ForecastRecord],
    *,
    as_of: datetime,
    snapshot_max_age_days: int = MONITOR_REANALYSIS_SNAPSHOT_MAX_AGE_DAYS,
) -> list[ReanalysisRecommendation]:
    """Deterministic re-analysis recommendations for the live ledger."""
    recommendations: list[ReanalysisRecommendation] = []
    as_of_utc = _as_utc(as_of)

    for forecast in forecasts:
        reasons: list[str] = []

        if forecast.outcome_id is not None and forecast.evaluation_verdict is None:
            reasons.append(ReanalysisReason.NEW_OUTCOME_RECORDED.value)
        elif forecast.evaluation_verdict is not None and forecast.evaluation_created_at is not None:
            evaluation_at = _as_utc(forecast.evaluation_created_at)
            has_new_evidence = (
                (forecast.latest_outcome_at is not None
                 and _as_utc(forecast.latest_outcome_at) > evaluation_at)
                or (forecast.latest_snapshot_at is not None
                    and _as_utc(forecast.latest_snapshot_at) > evaluation_at)
            )
            if has_new_evidence:
                reasons.append(ReanalysisReason.EVIDENCE_CHANGED.value)

        if (
            forecast.outcome_id is None
            and forecast.due_at is not None
            and as_of_utc >= _as_utc(forecast.due_at)
        ):
            reasons.append(ReanalysisReason.PREDICTION_EXPIRED.value)

        age_days = max(0, (as_of_utc - _as_utc(forecast.analysis_timestamp)).days)
        if (
            forecast.outcome_id is None
            and age_days > snapshot_max_age_days
        ):
            reasons.append(ReanalysisReason.SNAPSHOT_AGE_EXCEEDED.value)

        reasons = sorted(set(reasons), key=lambda value: _PRIORITY[ReanalysisReason(value)])
        if not reasons:
            continue

        recommendations.append(
            ReanalysisRecommendation(
                company_id=forecast.company_id,
                forecast_id=forecast.id,
                snapshot_id=forecast.snapshot_id,
                reasons=reasons,
                latest_outcome_at=forecast.latest_outcome_at,
                latest_snapshot_at=forecast.latest_snapshot_at,
            )
        )

    recommendations.sort(
        key=lambda item: (
            _PRIORITY[ReanalysisReason(item.reasons[0])],
            item.forecast_id,
        )
    )
    return recommendations


__all__ = ["build_reanalysis_recommendations"]
