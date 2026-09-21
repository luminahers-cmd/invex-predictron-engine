"""Phase 6 — deterministic time-series / trend extraction over snapshot history.

The time-series contract (per the Phase 6 requirement "expose accuracy for
Day 1, Day 2, Day 3, ...") is a :class:`MetricTrend`: a series of
:class:`TimePoint` values ordered by ``anchor`` ascending plus a
deterministic endpoint slope.

Mathematical definitions
------------------------
* ``metric_series(snapshots, key)`` — points ``(anchor_date, value)`` for
  every snapshot whose stored value for ``key`` is not ``None``, ordered by
  ``anchor_date`` ascending (duplicate anchors for the same key are
  impossible across a single history; if they ever occur the latest recorded
  wins).
* ``compute_trend(...)`` — ``from_value`` is the first point value,
  ``to_value`` the last, ``direction = direction_between(from, to)``
  (canonical :func:`predictron_engine.intelligence.history.direction_between`).
  ``None`` is returned when fewer than two usable points exist.

No heuristic or hidden threshold exists.
"""

from __future__ import annotations

from predictron_engine.intelligence.history import direction_between
from predictron_engine.monitoring.models import (
    METRIC_KEYS,
    MetricTrend,
    MonitorSnapshot,
    TimePoint,
)


def _metric_value(snapshot: MonitorSnapshot, key: str) -> float | None:
    """Resolve a metric map key to its float value (metrics then counts)."""
    if key in snapshot.metrics and snapshot.metrics[key] is not None:
        return float(snapshot.metrics[key])  # type: ignore[arg-type]
    if key in snapshot.counts:
        return float(snapshot.counts[key])
    return None


def metric_series(
    snapshots: list[MonitorSnapshot],
    key: str,
) -> list[TimePoint]:
    """Ordered (by anchor) series of values for a metric map key.

    Keys may be given either as the stable map key (e.g. ``"accuracy"``) or
    the canonical full path (e.g. ``"metrics.accuracy"``); both resolve to
    the same stored value.
    """
    resolved = METRIC_KEYS.get(key, key)
    ordered = sorted(snapshots, key=lambda entry: entry.anchor_date)
    latest_by_anchor: dict[str, TimePoint] = {}
    for snapshot in ordered:
        value = _metric_value(snapshot, resolved)
        if value is None:
            continue
        latest_by_anchor[snapshot.anchor_date.isoformat()] = TimePoint(
            anchor=snapshot.anchor_date, value=value
        )
    return [latest_by_anchor[key] for key in sorted(latest_by_anchor)]


def compute_trend(
    snapshots: list[MonitorSnapshot],
    key: str,
) -> MetricTrend | None:
    """Endpoint-slope trend for a metric over its anchor-ordered series.

    Returns ``None`` when fewer than two usable points exist (a single
    snapshot cannot evidence a trend).  The series reuses
    :func:`metric_series` so ``MetricTrend.series`` is exactly what
    dashboards plot.
    """
    series = metric_series(snapshots, key)
    if len(series) < 2:
        return None
    from_point, to_point = series[0], series[-1]
    return MetricTrend(
        metric=key,
        direction=direction_between(from_point.value, to_point.value),
        from_value=from_point.value,
        to_value=to_point.value,
        series=series,
    )


__all__ = ["compute_trend", "metric_series"]
