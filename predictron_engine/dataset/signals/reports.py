"""Deterministic signal reports (Project E4).

Report builders turn timeline collections into JSON-ready dictionaries —
timeline summaries, signal distributions, activity heatmaps, trend
summaries, momentum statistics, dataset freshness, and coverage.  Keys
are always sorted and derived numbers are pre-rounded so identical inputs
produce byte-identical output.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import datetime

from predictron_engine.dataset.signals.aggregate import (
    momentum_ranking,
    momentum_score,
    signal_freshness,
)
from predictron_engine.dataset.signals.metrics import compute_signal_statistics
from predictron_engine.dataset.signals.timeline import CompanyTimeline
from predictron_engine.dataset.signals.trends import TrendEngine

_MOMENTUM_BUCKETS = ("0.0-0.49", "0.5-0.99", "1.0-1.99", "2.0-3.99", "4.0+")


def _period_key(timestamp: datetime, granularity: str) -> str:
    if granularity == "quarter":
        quarter = (timestamp.month - 1) // 3 + 1
        return f"{timestamp.year:04d}-Q{quarter}"
    return f"{timestamp.year:04d}-{timestamp.month:02d}"


def build_timeline_report(timeline: CompanyTimeline) -> dict[str, object]:
    """Full human- and machine-readable view of one timeline."""
    signals: list[dict[str, object]] = []
    for signal in timeline.signals:
        signals.append(
            {
                "signal_id": signal.signal_id,
                "signal_type": signal.signal_type.value,
                "timestamp": signal.timestamp.isoformat(),
                "source": signal.source,
                "provenance": signal.provenance,
                "confidence": signal.confidence,
                "metadata": dict(signal.metadata),
            }
        )
    return {
        "company_id": timeline.company_id,
        "signal_count": timeline.signal_count,
        "span_days": timeline.span_days,
        "first_signal_at": (
            timeline.first.timestamp.isoformat() if timeline.first else None
        ),
        "last_signal_at": (
            timeline.last.timestamp.isoformat() if timeline.last else None
        ),
        "by_type": dict(sorted(timeline.types().items())),
        "signals": signals,
    }


def build_signal_distribution(
    timelines: Sequence[CompanyTimeline], *, expected_company_count: int = 0
) -> dict[str, object]:
    """Distribution of signal types, sources, years, and months."""
    stats = compute_signal_statistics(
        timelines, expected_company_count=expected_company_count
    )
    return {
        "total_signals": stats.total_signals,
        "companies_with_signals": stats.companies_with_signals,
        "expected_company_count": stats.expected_company_count,
        "by_type": stats.by_type,
        "by_source": stats.by_source,
        "by_year": stats.by_year,
        "by_month": stats.by_month,
        "average_confidence": stats.average_confidence,
        "coverage": stats.coverage,
    }


def build_activity_heatmap(
    timelines: Sequence[CompanyTimeline], *, granularity: str = "month"
) -> dict[str, object]:
    """Signal count per period, zero-filled across the observed range."""
    counts: Counter[str] = Counter()
    periods: list[str] = []

    def add(timestamp: datetime) -> None:
        key = _period_key(timestamp, granularity)
        counts[key] += 1
        periods.append(key)

    for timeline in timelines:
        for signal in timeline.signals:
            add(signal.timestamp)

    full: dict[str, int] = {}
    if periods:
        start = min(periods)
        end = max(periods)
        cursor = start
        while cursor <= end:
            full[cursor] = counts.get(cursor, 0)
            cursor = _next_period(cursor, granularity)
    return {
        "granularity": granularity,
        "period_count": len(full),
        "periods": full,
    }


def _next_period(key: str, granularity: str) -> str:
    year = int(key[:4])
    if granularity == "quarter":
        quarter = int(key[-1])
        if quarter < 4:
            return f"{year:04d}-Q{quarter + 1}"
        return f"{year + 1:04d}-Q1"
    month = int(key[5:7])
    if month < 12:
        return f"{year:04d}-{month + 1:02d}"
    return f"{year + 1:04d}-01"


def build_momentum_statistics(
    timelines: Sequence[CompanyTimeline], *, as_of: datetime | None = None
) -> dict[str, object]:
    """Momentum scores, ranking, and distribution across companies."""
    scores = {
        timeline.company_id: momentum_score(timeline, as_of=as_of)
        for timeline in timelines
    }
    ordering = momentum_ranking(list(timelines), as_of=as_of)
    distribution: Counter[str] = Counter()
    for score in scores.values():
        bucket = _MOMENTUM_BUCKETS[-1]
        for candidate in _MOMENTUM_BUCKETS[:-1]:
            threshold = float(candidate.split("-")[1])
            if score <= threshold:
                bucket = candidate
                break
        distribution[bucket] += 1
    ordered_scores = [
        {"company_id": company_id, "momentum_score": scores[company_id]}
        for company_id in ordering
    ]
    average = (
        round(sum(scores.values()) / len(scores), 4) if scores else None
    )
    return {
        "company_count": len(scores),
        "average_momentum_score": average,
        "distribution": dict(sorted(distribution.items())),
        "ranking": ordered_scores,
    }


def build_dataset_freshness_report(
    timelines: Sequence[CompanyTimeline], *, as_of: datetime | None = None
) -> dict[str, object]:
    """Per-company freshness and an aggregate freshness picture."""
    per_company: dict[str, dict[str, object]] = {}
    days: list[float] = []
    for timeline in sorted(timelines, key=lambda t: t.company_id):
        freshness = signal_freshness(timeline, as_of=as_of)
        last_days = freshness["days_since_last_signal"]
        if isinstance(last_days, int | float):
            days.append(float(last_days))
        per_company[timeline.company_id] = freshness
    return {
        "as_of": as_of.isoformat() if as_of else None,
        "company_count": len(timelines),
        "average_days_since_last_signal": (
            round(sum(days) / len(days), 4) if days else None
        ),
        "max_days_since_last_signal": round(max(days), 4) if days else None,
        "per_company": per_company,
    }


def build_coverage_report(
    timelines: Sequence[CompanyTimeline],
    expected_company_count: int,
    expected_company_ids: list[str] | None = None,
) -> dict[str, object]:
    """Coverage of the expected companies by signal timelines."""
    present = {t.company_id for t in timelines}
    expected_set = set(expected_company_ids or [])
    missing: list[str] = []
    if expected_set:
        missing = sorted(expected_set - present)
    coverage = (
        round(len(present) / expected_company_count, 4)
        if expected_company_count > 0
        else None
    )
    return {
        "companies_with_signals": len(present),
        "expected_company_count": expected_company_count,
        "coverage": coverage,
        "companies_without_signals": missing,
    }


def build_trend_summary_report(
    timelines: Sequence[CompanyTimeline],
    *,
    as_of: datetime | None = None,
    engine: TrendEngine | None = None,
) -> dict[str, object]:
    """Per-company trend results from a shared :class:`TrendEngine`."""
    active = engine or TrendEngine(as_of)
    return {
        "as_of": active.as_of.isoformat(),
        "companies": {
            timeline.company_id: active.summarize(timeline)
            for timeline in sorted(timelines, key=lambda t: t.company_id)
        },
    }


def build_signal_dataset_report(
    timelines: Sequence[CompanyTimeline],
    *,
    as_of: datetime | None = None,
    expected_company_count: int | None = None,
    expected_company_ids: list[str] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """One combined report document over a dataset's signal timelines."""
    statistics = compute_signal_statistics(
        timelines,
        expected_company_count=expected_company_count or 0,
        as_of=as_of,
    )
    expected = expected_company_count if expected_company_count is not None else 0
    per_company: list[dict[str, object]] = []
    for timeline in sorted(timelines, key=lambda t: t.company_id):
        per_company.append(
            {
                "company_id": timeline.company_id,
                "signal_count": timeline.signal_count,
                "first_signal_at": (
                    timeline.first.timestamp.isoformat() if timeline.first else None
                ),
                "last_signal_at": (
                    timeline.last.timestamp.isoformat() if timeline.last else None
                ),
                "span_days": timeline.span_days,
                "momentum_score": momentum_score(timeline, as_of=as_of),
            }
        )
    return {
        "report_type": "company_signals_report",
        "as_of": as_of.isoformat() if as_of else None,
        "statistics": statistics.to_dict(),
        "distribution": build_signal_distribution(timelines, expected_company_count=expected),
        "coverage": build_coverage_report(
            timelines, expected, expected_company_ids=expected_company_ids
        ),
        "momentum": build_momentum_statistics(timelines, as_of=as_of),
        "freshness": build_dataset_freshness_report(timelines, as_of=as_of),
        "heatmap": build_activity_heatmap(timelines, granularity="month"),
        "companies": per_company,
        "generated_at": (
            generated_at.isoformat() if generated_at is not None else None
        ),
    }
