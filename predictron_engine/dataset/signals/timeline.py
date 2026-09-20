"""Per-company signal timelines (Project E4).

A :class:`CompanyTimeline` is an **immutable, chronologically ordered**
sequence of :class:`~predictron_engine.dataset.signals.model.CompanySignal`
objects.  History is never mutated: ``append`` and ``merge`` always return
a *new* timeline, so earlier references stay valid snapshots of the past.

Construction guarantees
-----------------------
- Signals are always stored in ascending ``(timestamp, signal_id)`` order.
- Duplicate signals (same ``signal_id``) are collapsed to a single entry.
- All signals in one timeline belong to one company.

The companion :class:`TimelineStore` is an in-memory container keyed by
company id, used by data pipelines and tests.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from predictron_engine.dataset.signals.model import CompanySignal


def _day_delta(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 86400.0


def _ordering_key(signal: CompanySignal) -> tuple[datetime, str]:
    return (signal.timestamp, signal.signal_id)


def sort_and_dedupe(signals: list[CompanySignal]) -> list[CompanySignal]:
    """Sort by ``(timestamp, signal_id)`` and collapse duplicate ids.

    The id is content-derived (see :func:`compute_signal_id`), so this is
    deterministic.  The first occurrence of an id is kept.
    """
    ordered = sorted(signals, key=_ordering_key)
    seen: set[str] = set()
    deduped: list[CompanySignal] = []
    for signal in ordered:
        if signal.signal_id not in seen:
            seen.add(signal.signal_id)
            deduped.append(signal)
    return deduped


@dataclass(frozen=True)
class CompanyTimeline:
    """An immutable, ordered timeline of signals for one company."""

    company_id: str
    signals: tuple[CompanySignal, ...] = ()

    @classmethod
    def build(
        cls,
        company_id: str,
        signals: list[CompanySignal] | tuple[CompanySignal, ...],
    ) -> CompanyTimeline:
        """Construct a timeline, sorting and de-duplicating deterministically."""
        return cls(company_id=company_id, signals=tuple(sort_and_dedupe(list(signals))))

    def append(self, *new_signals: CompanySignal) -> CompanyTimeline:
        """Return a new timeline with ``new_signals`` added.

        The receiver is unchanged; the result is rebuilt sorted and
        de-duplicated.  Signals for a different company are rejected.
        """
        for signal in new_signals:
            if signal.company_id != self.company_id:
                raise ValueError(
                    f"cannot append signal for '{signal.company_id}' "
                    f"to timeline for '{self.company_id}'"
                )
        combined = list(self.signals) + list(new_signals)
        return CompanyTimeline.build(self.company_id, combined)

    def merge(self, other: CompanyTimeline) -> CompanyTimeline:
        """Return a new timeline merging ``other`` into this one.

        Raises ValueError when the timelines describe different companies.
        """
        if other.company_id != self.company_id:
            raise ValueError(
                f"cannot merge timeline '{other.company_id}' "
                f"into timeline '{self.company_id}'"
            )
        return CompanyTimeline.build(
            self.company_id, list(self.signals) + list(other.signals)
        )

    @property
    def signal_count(self) -> int:
        return len(self.signals)

    def has(self, signal_id: str) -> bool:
        """Return True when a signal with ``signal_id`` is present."""
        return any(s.signal_id == signal_id for s in self.signals)

    def find(self, signal_id: str) -> CompanySignal | None:
        """Return the signal with ``signal_id``, or None."""
        for signal in self.signals:
            if signal.signal_id == signal_id:
                return signal
        return None

    @property
    def first(self) -> CompanySignal | None:
        return self.signals[0] if self.signals else None

    @property
    def last(self) -> CompanySignal | None:
        return self.signals[-1] if self.signals else None

    @property
    def span_days(self) -> float:
        """Number of calendar days between first and last signal."""
        if len(self.signals) < 2:
            return 0.0
        first = self.signals[0].timestamp
        last = self.signals[-1].timestamp
        return round(_day_delta(first, last), 4)

    def types(self) -> Counter[str]:
        """Count of signals per signal type value."""
        return Counter(s.signal_type.value for s in self.signals)

    def by_type(self, signal_type_value: str) -> list[CompanySignal]:
        """All signals of a given signal type value, in chronological order."""
        return [s for s in self.signals if s.signal_type.value == signal_type_value]

    def signals_since(self, since: datetime) -> list[CompanySignal]:
        """Chronological signals at or after ``since``."""
        return [s for s in self.signals if s.timestamp >= since]

    def to_dict(self) -> dict[str, object]:
        from predictron_engine.dataset.signals.persistence import timeline_to_dict

        return timeline_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CompanyTimeline:
        from predictron_engine.dataset.signals.persistence import timeline_from_dict

        return timeline_from_dict(data)


@dataclass
class TimelineStore:
    """In-memory collection of timelines keyed by company id."""

    _timelines: dict[str, CompanyTimeline] = field(default_factory=dict)

    def add(self, timeline: CompanyTimeline) -> None:
        """Store (or replace) a timeline by its company id."""
        self._timelines[timeline.company_id] = timeline

    def put(self, company_id: str, signals: list[CompanySignal]) -> CompanyTimeline:
        """Build a timeline from signals and store it; returns it."""
        timeline = CompanyTimeline.build(company_id, signals)
        self._timelines[company_id] = timeline
        return timeline

    def get(self, company_id: str) -> CompanyTimeline | None:
        return self._timelines.get(company_id)

    def remove(self, company_id: str) -> bool:
        if company_id in self._timelines:
            del self._timelines[company_id]
            return True
        return False

    def list_companies(self) -> list[str]:
        return sorted(self._timelines.keys())

    def count(self) -> int:
        return len(self._timelines)

    def all(self) -> list[CompanyTimeline]:
        return [self._timelines[company_id] for company_id in self.list_companies()]

    def total_signals(self) -> int:
        return sum(t.signal_count for t in self._timelines.values())
