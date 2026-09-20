"""Signal statistics across a collection of timelines (Project E4).

Computes deterministic aggregate statistics — totals, per-type/source/
period counts, confidence and freshness distributions, and per-company
coverage — over a set of :class:`CompanyTimeline` objects.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from predictron_engine.dataset.signals.aggregate import signal_freshness
from predictron_engine.dataset.signals.timeline import CompanyTimeline


@dataclass
class SignalStatistics:
    """Aggregate statistics over a collection of timelines."""

    total_signals: int = 0
    expected_company_count: int = 0
    companies_with_signals: int = 0
    signals_per_company: dict[str, float] = field(default_factory=dict)
    by_type: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    by_year: dict[str, int] = field(default_factory=dict)
    by_month: dict[str, int] = field(default_factory=dict)
    average_confidence: float | None = None
    confidence_histogram: dict[str, int] = field(default_factory=dict)
    freshness_histogram: dict[str, int] = field(default_factory=dict)
    funding_events: int = 0
    total_amount_usd: float = 0.0

    @property
    def coverage(self) -> float | None:
        if self.expected_company_count <= 0:
            return None
        return round(self.companies_with_signals / self.expected_company_count, 4)

    def to_dict(self) -> dict[str, object]:
        return {
            "total_signals": self.total_signals,
            "expected_company_count": self.expected_company_count,
            "companies_with_signals": self.companies_with_signals,
            "signals_per_company": self.signals_per_company,
            "by_type": dict(sorted(self.by_type.items())),
            "by_source": dict(sorted(self.by_source.items())),
            "by_year": dict(sorted(self.by_year.items())),
            "by_month": dict(sorted(self.by_month.items())),
            "average_confidence": self.average_confidence,
            "confidence_histogram": dict(sorted(self.confidence_histogram.items())),
            "freshness_histogram": dict(sorted(self.freshness_histogram.items())),
            "funding_events": self.funding_events,
            "total_amount_usd": self.total_amount_usd,
            "coverage": self.coverage,
        }


def _confidence_bucket(confidence: float) -> str:
    if confidence < 0.5:
        return "0.0-0.5"
    if confidence < 0.8:
        return "0.5-0.8"
    return "0.8-1.0"


def compute_signal_statistics(
    timelines: Sequence[CompanyTimeline],
    *,
    expected_company_count: int = 0,
    as_of: datetime | None = None,
) -> SignalStatistics:
    """Compute statistics over ``timelines``.

    ``expected_company_count`` is the total number of companies in the
    dataset (brands, records, or resolved identities) used for coverage.
    ``as_of`` is the reference time for freshness buckets.
    """
    reference = as_of or datetime.now(UTC)
    stats = SignalStatistics(
        expected_company_count=expected_company_count,
        companies_with_signals=len(timelines),
    )
    by_type: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    by_year: Counter[str] = Counter()
    by_month: Counter[str] = Counter()
    confidence_histogram: Counter[str] = Counter()
    freshness_histogram: Counter[str] = Counter()

    confidences: list[float] = []
    per_company: list[int] = []

    for timeline in timelines:
        per_company.append(timeline.signal_count)
        bucket = signal_freshness(timeline, as_of=reference)["freshness_bucket"]
        freshness_histogram[str(bucket) if bucket else "no_signals"] += 1
        for signal in timeline.signals:
            stats.total_signals += 1
            by_type[signal.signal_type.value] += 1
            by_source[signal.source] += 1
            by_year[str(signal.timestamp.year)] += 1
            by_month[f"{signal.timestamp.year:04d}-{signal.timestamp.month:02d}"] += 1
            confidence_histogram[_confidence_bucket(signal.confidence)] += 1
            confidences.append(signal.confidence)
            if signal.signal_type.value == "funding_round":
                stats.funding_events += 1
                raw = signal.metadata.get("amount_usd")
                if isinstance(raw, int | float) and not isinstance(raw, bool):
                    stats.total_amount_usd += float(raw)

    stats.by_type = dict(by_type)
    stats.by_source = dict(by_source)
    stats.by_year = dict(by_year)
    stats.by_month = dict(by_month)
    stats.confidence_histogram = dict(confidence_histogram)
    stats.freshness_histogram = dict(freshness_histogram)
    stats.total_amount_usd = round(stats.total_amount_usd, 4)

    if confidences:
        stats.average_confidence = round(
            sum(confidences) / len(confidences), 4
        )
    if per_company:
        stats.signals_per_company = {
            "min": float(min(per_company)),
            "max": float(max(per_company)),
            "average": round(sum(per_company) / len(per_company), 4),
        }
    return stats
