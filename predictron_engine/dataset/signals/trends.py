"""Deterministic, explainable trend analysis (Project E4).

Every trend returns a :class:`TrendResult` carrying the measured value, a
human-readable direction label, a short explanation, and the supporting
metrics.  No trend makes a forecast — the names and descriptions refer
exclusively to what has already happened in the timeline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from predictron_engine.dataset.signals.model import CompanySignal, SignalType
from predictron_engine.dataset.signals.timeline import CompanyTimeline

_SECONDS_PER_DAY = 86400.0
_DAYS_PER_YEAR = 365.25

_LINEAR_THRESHOLD_FLAT = 0.05
_SLOPE_ANNUALIZATION = _DAYS_PER_YEAR


def _as_of_ref(as_of: datetime | None) -> datetime:
    return as_of or datetime.now(UTC)


def _age_days(timestamp: datetime, as_of: datetime) -> float:
    return (as_of - timestamp).total_seconds() / _SECONDS_PER_DAY


@dataclass(frozen=True)
class TrendResult:
    """Output of a single deterministic trend analysis."""

    name: str
    value: float
    direction: str
    window_days: int
    explanation: str
    metrics: Mapping[str, object] = field(default_factory=dict)
    available: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "direction": self.direction,
            "window_days": self.window_days,
            "explanation": self.explanation,
            "metrics": dict(self.metrics),
            "available": self.available,
        }


def _lsq_slope(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return (slope_per_day, intercept, r_squared) for a linear fit."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0
    for x, y in zip(xs, ys):
        sum_x += x
        sum_y += y
        sum_xx += x * x
        sum_xy += x * y
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.0, sum_y / n if n else 0.0, 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    mean_y = sum_y / n if n else 0.0
    ss_res = sum((y - slope * x - intercept) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - (ss_res / ss_tot) if abs(ss_tot) > 1e-12 else 0.0
    return slope, intercept, max(min(r2, 1.0), 0.0)


class TrendEngine:
    """Facade exposing every deterministic trend analysis.

    Instantiate once per reporting pass with a fixed ``as_of`` so every
    analysis runs against the same reference time.
    """

    def __init__(self, as_of: datetime | None = None) -> None:
        self._as_of = _as_of_ref(as_of)

    @property
    def as_of(self) -> datetime:
        return self._as_of

    def funding_velocity(self, timeline: CompanyTimeline) -> TrendResult:
        return compute_funding_velocity(timeline, as_of=self._as_of)

    def funding_acceleration(self, timeline: CompanyTimeline) -> TrendResult:
        return compute_funding_acceleration(timeline, as_of=self._as_of)

    def hiring_trend(self, timeline: CompanyTimeline) -> TrendResult:
        return compute_hiring_trend(timeline, as_of=self._as_of)

    def growth_consistency(self, timeline: CompanyTimeline) -> TrendResult:
        return compute_growth_consistency(timeline)

    def stagnation(self, timeline: CompanyTimeline) -> TrendResult:
        return compute_stagnation(timeline, as_of=self._as_of)

    def decline(self, timeline: CompanyTimeline) -> TrendResult:
        return compute_decline(timeline, as_of=self._as_of)

    def evaluate(self, timeline: CompanyTimeline) -> dict[str, TrendResult]:
        return {
            "funding_velocity": self.funding_velocity(timeline),
            "funding_acceleration": self.funding_acceleration(timeline),
            "hiring_trend": self.hiring_trend(timeline),
            "growth_consistency": self.growth_consistency(timeline),
            "stagnation": self.stagnation(timeline),
            "decline": self.decline(timeline),
        }

    def summarize(self, timeline: CompanyTimeline) -> dict[str, dict[str, object]]:
        return {
            name: result.to_dict()
            for name, result in self.evaluate(timeline).items()
        }


def _timeline_window_days(timeline: CompanyTimeline, as_of: datetime) -> int:
    if timeline.first is None:
        return 0
    return max(int(_age_days(timeline.first.timestamp, as_of)), 0)


def compute_funding_velocity(
    timeline: CompanyTimeline, *, as_of: datetime | None = None
) -> TrendResult:
    reference = _as_of_ref(as_of)
    funding = sorted(
        (s for s in timeline.signals if s.signal_type == SignalType.FUNDING_ROUND),
        key=lambda s: s.timestamp,
    )
    if not funding:
        return TrendResult(
            "funding_velocity", 0.0, "flat", 0, "no funding signals recorded", {}, False
        )
    amounts: list[float] = []
    for signal in funding:
        raw = signal.metadata.get("amount_usd")
        if isinstance(raw, int | float) and not isinstance(raw, bool):
            amounts.append(float(raw))
    total_amount = sum(amounts)
    span_days = max(_age_days(funding[0].timestamp, reference), 1e-9)
    span_years = max(span_days / _DAYS_PER_YEAR, 1e-9)
    if amounts:
        value = round(total_amount / span_years, 4)
        unit = "usd_per_year"
        metrics: dict[str, object] = {
            "funding_count": len(funding),
            "total_amount_usd": round(total_amount, 4),
            "span_years": round(span_years, 4),
            "unit": unit,
        }
        direction = "fundraising" if value > 0.0 else "flat"
        explanation = (
            f"{len(funding)} funding rounds totalling "
            f"{total_amount:,.0f} over {span_years:.2f} years "
            f"(≈ {value:,.0f}/year)"
        )
    else:
        value = round(len(funding) / span_years, 4)
        metrics = {
            "funding_count": len(funding),
            "total_amount_usd": 0.0,
            "span_years": round(span_years, 4),
            "unit": "rounds_per_year",
        }
        direction = "fundraising" if value > 0.0 else "flat"
        explanation = f"{len(funding)} funding rounds over {span_years:.2f} years"
    window = _timeline_window_days(timeline, reference)
    return TrendResult("funding_velocity", value, direction, window, explanation, metrics)


def compute_funding_acceleration(
    timeline: CompanyTimeline, *, as_of: datetime | None = None
) -> TrendResult:
    reference = _as_of_ref(as_of)
    funding = sorted(
        (s for s in timeline.signals if s.signal_type == SignalType.FUNDING_ROUND),
        key=lambda s: s.timestamp,
    )
    if len(funding) < 2:
        return TrendResult(
            "funding_acceleration",
            0.0,
            "flat",
            _timeline_window_days(timeline, reference),
            "fewer than two funding events",
            {},
            False,
        )
    first_ts = funding[0].timestamp
    total_days = _age_days(first_ts, reference)
    if total_days < 1.0:
        return TrendResult(
            "funding_acceleration",
            0.0,
            "flat",
            _timeline_window_days(timeline, reference),
            "all funding events within one day",
            {},
        )
    mid_ts = datetime.fromtimestamp(
        first_ts.timestamp() + total_days * _SECONDS_PER_DAY / 2.0, tz=UTC
    )

    def _rate(events: list[CompanySignal]) -> float:
        if not events:
            return 0.0
        days = max(_age_days(first_ts, events[-1].timestamp), 1.0)
        years = max(days / _DAYS_PER_YEAR, 1e-9)
        return len(events) / years

    first_half = [s for s in funding if s.timestamp <= mid_ts]
    second_half = [s for s in funding if s.timestamp > mid_ts]
    rate_early = _rate(first_half)
    rate_late = _rate(second_half)
    accel = round(rate_late - rate_early, 4)
    if accel > 0.01:
        direction = "accelerating"
    elif accel < -0.01:
        direction = "decelerating"
    else:
        direction = "flat"
    explanation = (
        f"early rate {rate_early:.2f} rounds/year vs "
        f"later {rate_late:.2f} rounds/year "
        f"(Δ {accel:+.2f})"
    )
    window = _timeline_window_days(timeline, reference)
    metrics: dict[str, object] = {
        "early_round_count": len(first_half),
        "later_round_count": len(second_half),
        "early_rate": round(rate_early, 4),
        "later_rate": round(rate_late, 4),
        "acceleration": accel,
    }
    return TrendResult(
        "funding_acceleration", accel, direction, window, explanation, metrics
    )


def _hiring_slope(timeline: CompanyTimeline) -> tuple[float, float, int]:
    """Return (slope_per_year, r_squared, points) for headcount milestones."""
    points: list[tuple[float, float]] = []
    base: datetime | None = None
    for s in timeline.signals:
        if s.signal_type != SignalType.EMPLOYEE_MILESTONE:
            continue
        raw = s.metadata.get("count")
        if not isinstance(raw, int | float) or isinstance(raw, bool):
            continue
        if base is None:
            base = s.timestamp
        days = (s.timestamp - base).total_seconds() / _SECONDS_PER_DAY
        points.append((days, float(raw)))
    if len(points) < 2:
        return 0.0, 0.0, len(points)
    points.sort(key=lambda p: (p[0], p[1]))
    slope_per_day, _, r2 = _lsq_slope([p[0] for p in points], [p[1] for p in points])
    return round(slope_per_day * _SLOPE_ANNUALIZATION, 4), round(r2, 4), len(points)


def compute_hiring_trend(
    timeline: CompanyTimeline, *, as_of: datetime | None = None
) -> TrendResult:
    reference = _as_of_ref(as_of)
    slope, r2, points = _hiring_slope(timeline)
    window = _timeline_window_days(timeline, reference)
    if points < 2:
        return TrendResult(
            "hiring_trend",
            0.0,
            "flat",
            window,
            "fewer than two employee count snapshots",
            {"points": points, "r_squared": 0.0, "annualized_growth": 0.0},
            False,
        )
    if slope > _LINEAR_THRESHOLD_FLAT:
        direction = "up"
    elif slope < -_LINEAR_THRESHOLD_FLAT:
        direction = "down"
    else:
        direction = "flat"
    explanation = (
        f"least-squares annualized headcount slope {slope:+.2f} employees/year "
        f"(R²={r2:.2f}, {points} points)"
    )
    return TrendResult(
        "hiring_trend",
        slope,
        direction,
        window,
        explanation,
        {
            "points": points,
            "r_squared": r2,
            "annualized_growth": slope,
        },
    )


def compute_growth_consistency(timeline: CompanyTimeline) -> TrendResult:
    milestone_types = frozenset(
        {
            SignalType.ARR_MILESTONE,
            SignalType.EMPLOYEE_MILESTONE,
            SignalType.REVENUE_MILESTONE,
        }
    )
    dates = sorted(
        {
            s.timestamp
            for s in timeline.signals
            if s.signal_type in milestone_types
        }
    )
    if len(dates) < 3:
        return TrendResult(
            "growth_consistency",
            0.0,
            "stable",
            0,
            "insufficient milestone density",
            {"interval_count": max(len(dates) - 1, 0), "cv": 0.0},
            False,
        )
    intervals: list[float] = []
    for i in range(1, len(dates)):
        delta = (dates[i] - dates[i - 1]).total_seconds() / _SECONDS_PER_DAY
        intervals.append(max(delta, 0.0))
    if not intervals:
        return TrendResult(
            "growth_consistency", 0.0, "stable", 0, "no intervals", {}, False
        )
    mean = sum(intervals) / len(intervals)
    if abs(mean) < 1e-9:
        return TrendResult(
            "growth_consistency", 0.0, "stable", 0, "all intervals zero", {}
        )
    var = sum((v - mean) ** 2 for v in intervals) / len(intervals)
    cv = round(math.sqrt(var) / mean, 4)
    if cv <= 0.6:
        direction = "consistent"
        explanation = f"coefficient of variation {cv:.2f} across {len(intervals)} intervals"
    else:
        direction = "variable"
        explanation = (
            f"coefficient of variation {cv:.2f} across {len(intervals)} intervals "
            f"(mean {mean:.1f} days)"
        )
    window = (
        int((dates[-1] - dates[0]).total_seconds() / _SECONDS_PER_DAY)
        if dates[-1] != dates[0]
        else 0
    )
    return TrendResult(
        "growth_consistency",
        cv,
        direction,
        window,
        explanation,
        {"interval_count": len(intervals), "cv": cv, "mean_interval_days": round(mean, 4)},
    )


def compute_stagnation(
    timeline: CompanyTimeline, *, as_of: datetime | None = None
) -> TrendResult:
    reference = _as_of_ref(as_of)
    window = 365
    if timeline.last is None:
        return TrendResult(
            "stagnation", 1.0, "stagnant", window, "no signals recorded", {}, True
        )
    since_last = _age_days(timeline.last.timestamp, reference)
    stagnant = 1.0 if since_last > window else 0.0
    if stagnant:
        direction = "stagnant"
        explanation = (
            f"no signals in the last {window} days "
            f"(last signal {since_last:.0f} days ago)"
        )
    else:
        direction = "active"
        explanation = (
            f"last signal {since_last:.0f} days ago "
            f"(within {window}-day activity window)"
        )
    return TrendResult(
        "stagnation",
        stagnant,
        direction,
        window,
        explanation,
        {"days_since_last_signal": round(since_last, 4), "window_days": window},
    )


def compute_decline(
    timeline: CompanyTimeline, *, as_of: datetime | None = None
) -> TrendResult:
    reference = _as_of_ref(as_of)
    layoffs = sum(
        1 for s in timeline.signals if s.signal_type == SignalType.LAYOFFS
    )
    shutdown = any(s.signal_type == SignalType.SHUTDOWN for s in timeline.signals)
    slope, _, points = _hiring_slope(timeline)
    negative_slope = slope < -_LINEAR_THRESHOLD_FLAT if points >= 2 else False
    indicators = [layoffs > 0, shutdown, negative_slope]
    triggered = sum(indicators)
    value = round(triggered / len(indicators), 4) if indicators else 0.0
    if triggered >= 1:
        direction = "declining"
        parts: list[str] = []
        if layoffs > 0:
            parts.append(f"{layoffs} layoff signals")
        if shutdown:
            parts.append("shutdown recorded")
        if negative_slope:
            parts.append(f"declining headcount slope ({slope:+.2f}/year)")
        explanation = "negative indicators: " + "; ".join(parts)
    else:
        direction = "healthy"
        explanation = "no negative indicators present"
    window = _timeline_window_days(timeline, reference)
    return TrendResult(
        "decline",
        value,
        direction,
        window,
        explanation,
        {
            "layoff_signals": layoffs,
            "shutdown": shutdown,
            "headcount_slope": slope,
            "negative_slope": negative_slope,
        },
    )
