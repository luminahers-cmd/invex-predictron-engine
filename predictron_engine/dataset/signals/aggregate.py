"""Signal aggregation metrics (Project E4).

Aggregates reduce a :class:`CompanyTimeline` to a handful of
deterministic, explainable numbers — recent activity, momentum, funding
and growth cadence, signal frequency, and freshness.  None of these make
predictions; they summarize what the timeline already records.

Every function accepts an explicit ``as_of`` reference timestamp.  When
omitted the current UTC time is used, but callers that need byte-level
deterministic output (reporting, tests) should always pass one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime

from predictron_engine.dataset.signals.model import SignalType
from predictron_engine.dataset.signals.timeline import CompanyTimeline

_FUNDING_TYPES = frozenset({SignalType.FUNDING_ROUND})
_GROWTH_TYPES = frozenset(
    {
        SignalType.ARR_MILESTONE,
        SignalType.EMPLOYEE_MILESTONE,
        SignalType.REVENUE_MILESTONE,
    }
)

_SECONDS_PER_DAY = 86400.0
_DAYS_PER_YEAR = 365.25


def _reference(as_of: datetime | None) -> datetime:
    return as_of or datetime.now(UTC)


def _age_days(timestamp: datetime, as_of: datetime) -> float:
    return (as_of - timestamp).total_seconds() / _SECONDS_PER_DAY


def _span_years(timeline: CompanyTimeline) -> float:
    first = timeline.first
    last = timeline.last
    if first is None or last is None or first is last:
        return 0.0
    days = (last.timestamp - first.timestamp).total_seconds() / (_SECONDS_PER_DAY)
    return max(days / _DAYS_PER_YEAR, 0.0)


def recent_activity(
    timeline: CompanyTimeline,
    *,
    window_days: int = 365,
    as_of: datetime | None = None,
) -> dict[str, object]:
    """Summarize how recently a company produced signals.

    ``is_active`` is True when at least one signal falls inside the
    trailing ``window_days`` window.
    """
    reference = _reference(as_of)
    recent = [
        s
        for s in timeline.signals
        if _age_days(s.timestamp, reference) <= window_days
    ]
    total = timeline.signal_count
    last_age: float | None = None
    if timeline.last is not None:
        last_age = round(_age_days(timeline.last.timestamp, reference), 4)
    return {
        "signal_count": total,
        "recent_signal_count": len(recent),
        "recent_ratio": round(len(recent) / total, 4) if total else 0.0,
        "days_since_last_signal": last_age,
        "is_active": bool(recent),
    }


def momentum_score(
    timeline: CompanyTimeline,
    *,
    as_of: datetime | None = None,
    half_life_days: float = 180.0,
) -> float:
    """Recency-weighted sum of signal confidence.

    Older signals decay exponentially with a ``half_life_days`` of 180
    days by default, so a company with recent, confident signals scores
    higher than one whose activity is stale.  Deterministic for a fixed
    ``as_of``.
    """
    reference = _reference(as_of)
    weight = 0.0
    for signal in timeline.signals:
        age = max(_age_days(signal.timestamp, reference), 0.0)
        weight += signal.confidence * math.exp(-age / half_life_days)
    return round(weight, 4)


def momentum_summary(
    timeline: CompanyTimeline,
    *,
    as_of: datetime | None = None,
    half_life_days: float = 180.0,
) -> dict[str, object]:
    """Explainable breakdown behind :func:`momentum_score`."""
    reference = _reference(as_of)
    score = momentum_score(timeline, as_of=reference, half_life_days=half_life_days)
    weighted_contributions: dict[str, float] = {}
    for signal in timeline.signals:
        age = max(_age_days(signal.timestamp, reference), 0.0)
        contribution = round(
            signal.confidence * math.exp(-age / half_life_days), 6
        )
        key = signal.signal_type.value
        weighted_contributions[key] = round(
            weighted_contributions.get(key, 0.0) + contribution, 6
        )
    return {
        "score": score,
        "signal_count": timeline.signal_count,
        "half_life_days": half_life_days,
        "weighted_contributions": dict(
            sorted(weighted_contributions.items(), key=lambda kv: kv[0])
        ),
    }


def funding_cadence(
    timeline: CompanyTimeline,
    *,
    as_of: datetime | None = None,
) -> dict[str, object]:
    """Funding rhythm: how often and how much a company raises."""
    reference = _reference(as_of)
    funding = [s for s in timeline.signals if s.signal_type in _FUNDING_TYPES]
    total_amount = 0.0
    for signal in funding:
        amount = signal.metadata.get("amount_usd")
        if isinstance(amount, int | float) and not isinstance(amount, bool):
            total_amount += float(amount)
    count = len(funding)
    span_days = 0.0
    if count >= 2:
        span_days = round(
            (funding[-1].timestamp - funding[0].timestamp).total_seconds()
            / _SECONDS_PER_DAY,
            4,
        )
    span_years = max(span_days / _DAYS_PER_YEAR, 1e-9)
    avg_interval_days: float | None = None
    if count > 1:
        avg_interval_days = round(span_days / (count - 1), 4)
    days_since_last: float | None = None
    if funding:
        days_since_last = round(
            _age_days(funding[-1].timestamp, reference), 4
        )
    return {
        "round_count": count,
        "span_days": span_days,
        "rounds_per_year": round(count / span_years, 4),
        "avg_interval_days": avg_interval_days,
        "total_amount_usd": round(total_amount, 4),
        "days_since_last_round": days_since_last,
    }


def growth_cadence(timeline: CompanyTimeline) -> dict[str, object]:
    """Growth rhythm: cadence of milestone signals.

    Mirrors :func:`funding_cadence` for growth milestones (ARR, revenue,
    and employee milestones).
    """
    milestones = [s for s in timeline.signals if s.signal_type in _GROWTH_TYPES]
    count = len(milestones)
    span_days = 0.0
    if count >= 2:
        span_days = round(
            (milestones[-1].timestamp - milestones[0].timestamp).total_seconds()
            / _SECONDS_PER_DAY,
            4,
        )
    span_years = max(span_days / _DAYS_PER_YEAR, 1e-9)
    avg_interval_days: float | None = None
    if count > 1:
        avg_interval_days = round(span_days / (count - 1), 4)
    return {
        "milestone_count": count,
        "span_days": span_days,
        "milestones_per_year": round(count / span_years, 4),
        "avg_interval_days": avg_interval_days,
    }


def signal_frequency(timeline: CompanyTimeline) -> dict[str, object]:
    """Overall signalling frequency and per-type breakdown."""
    by_type: dict[str, int] = {}
    for signal in timeline.signals:
        key = signal.signal_type.value
        by_type[key] = by_type.get(key, 0) + 1
    span_years = _span_years(timeline) or 1e-9
    per_year = (
        round(timeline.signal_count / span_years, 4)
        if timeline.signal_count
        else 0.0
    )
    return {
        "total_signals": timeline.signal_count,
        "signals_per_year": per_year,
        "span_years": round(_span_years(timeline), 4),
        "by_type": dict(sorted(by_type.items())),
    }


def _freshness_bucket(days_since_last: float) -> str:
    if days_since_last <= 30:
        return "0-30"
    if days_since_last <= 90:
        return "31-90"
    if days_since_last <= 365:
        return "91-365"
    return ">365"


def signal_freshness(
    timeline: CompanyTimeline,
    *,
    as_of: datetime | None = None,
) -> dict[str, object]:
    """How fresh a company's signal history is relative to ``as_of``."""
    reference = _reference(as_of)
    days_since_last: float | None = None
    if timeline.last is not None:
        days_since_last = round(_age_days(timeline.last.timestamp, reference), 4)
    bucket: str | None = None
    if days_since_last is not None:
        bucket = _freshness_bucket(days_since_last)
    return {
        "signal_count": timeline.signal_count,
        "last_signal_at": (
            timeline.last.timestamp.isoformat() if timeline.last is not None else None
        ),
        "days_since_last_signal": days_since_last,
        "freshness_bucket": bucket,
    }


def aggregate_all(
    timeline: CompanyTimeline,
    *,
    as_of: datetime | None = None,
) -> dict[str, object]:
    """Convenience bundle of every aggregation for a single timeline."""
    reference = _reference(as_of)
    return {
        "recent_activity": recent_activity(timeline, as_of=reference),
        "momentum": momentum_summary(timeline, as_of=reference),
        "funding_cadence": funding_cadence(timeline, as_of=reference),
        "growth_cadence": growth_cadence(timeline),
        "signal_frequency": signal_frequency(timeline),
        "signal_freshness": signal_freshness(timeline, as_of=reference),
    }


def momentum_ranking(
    timelines: Sequence[CompanyTimeline],
    *,
    as_of: datetime | None = None,
) -> list[str]:
    """Company ids ranked by momentum score, highest first (ties: by id)."""
    reference = _reference(as_of)
    scored = [
        (momentum_score(timeline, as_of=reference), timeline.company_id)
        for timeline in timelines
    ]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [company_id for _score, company_id in scored]
