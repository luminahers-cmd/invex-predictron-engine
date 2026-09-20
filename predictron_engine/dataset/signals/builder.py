"""Grounded signal extraction and timeline building (Project E4).

The builder transforms stored :class:`DatasetRecord`\\ s and their linked
:class:`OutcomeRecord`\\ s into per-company signal timelines.  Every
extracted signal is **grounded**: it is emitted only when the source data
carries the required timestamp (a funding round without a date produces no
funding signal; an acquisition without an exit date produces no
acquisition signal).  Nothing is fabricated.

Signals that the schema cannot ground (founder changes, product launches,
partnerships, regulatory approvals, revenue milestones) are deliberately
not synthesized here — they must enter the dataset explicitly through
``signal-import``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from predictron_engine.dataset.graph.builder import company_node_id
from predictron_engine.dataset.models import DatasetRecord
from predictron_engine.dataset.outcomes import FundingEvent, OutcomeRecord
from predictron_engine.dataset.signals.model import (
    CompanySignal,
    EvidenceReference,
    SignalType,
)
from predictron_engine.dataset.signals.timeline import CompanyTimeline

# Deterministic confidence assignments per derived signal type.  These
# express how much of the observation the source structure guarantees.
_FUNDING_CONFIDENCE = 0.9
_INVESTOR_CONFIDENCE = 0.8
_VALUATION_CONFIDENCE = 0.85
_MILESTONE_CONFIDENCE = 0.85
_EMPLOYEE_CONFIDENCE = 0.8
_TERMINAL_CONFIDENCE = 0.95


def _as_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC, assuming naive datetimes are UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _parse_snapshot_date(value: object) -> datetime | None:
    """Parse a snapshot date that may be a datetime or ISO string."""
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return _as_utc(parsed)
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


class OutcomeSignalExtractor:
    """Deterministic extraction of signals from an outcome record.

    ``extract`` never raises on missing data; it simply emits fewer
    signals and records skip reasons for entries that lack a grounded
    date.
    """

    def extract(
        self, outcome: OutcomeRecord, company_id: str
    ) -> tuple[list[CompanySignal], list[str]]:
        signals: list[CompanySignal] = []
        skips: list[str] = []
        data = outcome.outcome

        for index, event in enumerate(data.funding_rounds):
            round_signals, round_skips = self._from_funding_event(
                company_id, event, index
            )
            signals.extend(round_signals)
            skips.extend(round_skips)

        signals.extend(self._arr_milestones(company_id, data.arr_milestones, skips))
        signals.extend(
            self._valuation_history(company_id, data.valuation_history, skips)
        )
        signals.extend(
            self._employee_history(company_id, data.employee_count_history, skips)
        )

        if data.acquisition:
            if data.exit_date is not None:
                signals.append(
                    CompanySignal(
                        company_id=company_id,
                        signal_type=SignalType.ACQUISITION,
                        timestamp=_as_utc(data.exit_date),
                        source="outcome",
                        provenance="derived:outcome",
                        confidence=_TERMINAL_CONFIDENCE,
                        evidence=EvidenceReference(
                            reference="outcome",
                            description=f"acquired by {data.acquisition}",
                        ),
                        metadata={"acquirer": data.acquisition},
                    )
                )
            else:
                skips.append(f"{company_id}: acquisition without exit_date")

        if data.exit_type == "ipo" and data.exit_date is not None:
            signals.append(
                CompanySignal(
                    company_id=company_id,
                    signal_type=SignalType.IPO,
                    timestamp=_as_utc(data.exit_date),
                    source="outcome",
                    provenance="derived:outcome",
                    confidence=_TERMINAL_CONFIDENCE,
                    evidence=EvidenceReference(
                        reference="outcome", description="IPO exit recorded"
                    ),
                    metadata={},
                )
            )

        if data.shutdown is True:
            if data.shutdown_date is not None:
                signals.append(
                    CompanySignal(
                        company_id=company_id,
                        signal_type=SignalType.SHUTDOWN,
                        timestamp=_as_utc(data.shutdown_date),
                        source="outcome",
                        provenance="derived:outcome",
                        confidence=_TERMINAL_CONFIDENCE,
                        evidence=EvidenceReference(
                            reference="outcome", description="shutdown recorded"
                        ),
                        metadata={},
                    )
                )
            else:
                skips.append(f"{company_id}: shutdown without shutdown_date")

        if data.bankruptcy is True:
            if data.bankruptcy_date is not None:
                signals.append(
                    CompanySignal(
                        company_id=company_id,
                        signal_type=SignalType.BANKRUPTCY,
                        timestamp=_as_utc(data.bankruptcy_date),
                        source="outcome",
                        provenance="derived:outcome",
                        confidence=_TERMINAL_CONFIDENCE,
                        evidence=EvidenceReference(
                            reference="outcome", description="bankruptcy recorded"
                        ),
                        metadata={},
                    )
                )
            else:
                skips.append(f"{company_id}: bankruptcy without bankruptcy_date")

        return signals, skips

    def _from_funding_event(
        self, company_id: str, event: FundingEvent, index: int
    ) -> tuple[list[CompanySignal], list[str]]:
        if event.date is None:
            return [], [f"{company_id}: funding round {index} without date"]
        date = _as_utc(event.date)
        source = event.source or "outcome"
        metadata: dict[str, object] = {
            "round_type": event.round_type,
            "round_index": index,
        }
        if event.amount_usd is not None:
            metadata["amount_usd"] = event.amount_usd
        if event.valuation_usd is not None:
            metadata["valuation_usd"] = event.valuation_usd

        signals: list[CompanySignal] = []
        signals.append(
            CompanySignal(
                company_id=company_id,
                signal_type=SignalType.FUNDING_ROUND,
                timestamp=date,
                source=source,
                provenance="derived:funding_round",
                confidence=_FUNDING_CONFIDENCE,
                evidence=EvidenceReference(
                    reference=source,
                    description=f"{event.round_type} funding round",
                ),
                metadata=metadata,
            )
        )
        for investor in sorted(set(event.investors)):
            signals.append(
                CompanySignal(
                    company_id=company_id,
                    signal_type=SignalType.INVESTOR_ADDED,
                    timestamp=date,
                    source=source,
                    provenance="derived:funding_round",
                    confidence=_INVESTOR_CONFIDENCE,
                    evidence=EvidenceReference(
                        reference=source,
                        description=(
                            f"{investor} participated in {event.round_type} round"
                        ),
                    ),
                    metadata={"investor": investor, "round_type": event.round_type},
                )
            )
        return signals, []

    def _arr_milestones(
        self,
        company_id: str,
        snapshots: list[dict[str, Any]],
        skips: list[str],
    ) -> list[CompanySignal]:
        signals: list[CompanySignal] = []
        for snapshot in snapshots:
            date = _parse_snapshot_date(snapshot.get("date"))
            amount = _float_or_none(snapshot.get("arr_usd"))
            if date is None:
                skips.append(f"{company_id}: arr_milestone without date")
                continue
            source = str(snapshot.get("source", "outcome"))
            metadata: dict[str, object] = {}
            if amount is not None:
                metadata["arr_usd"] = amount
            signals.append(
                CompanySignal(
                    company_id=company_id,
                    signal_type=SignalType.ARR_MILESTONE,
                    timestamp=date,
                    source=source,
                    provenance="derived:outcome",
                    confidence=_MILESTONE_CONFIDENCE,
                    evidence=EvidenceReference(
                        reference=source, description="ARR milestone snapshot"
                    ),
                    metadata=metadata,
                )
            )
        return signals

    def _valuation_history(
        self,
        company_id: str,
        snapshots: list[dict[str, Any]],
        skips: list[str],
    ) -> list[CompanySignal]:
        signals: list[CompanySignal] = []
        for snapshot in snapshots:
            date = _parse_snapshot_date(snapshot.get("date"))
            valuation = _float_or_none(snapshot.get("valuation_usd"))
            if date is None:
                skips.append(f"{company_id}: valuation snapshot without date")
                continue
            source = str(snapshot.get("source", "outcome"))
            metadata: dict[str, object] = {}
            if valuation is not None:
                metadata["valuation_usd"] = valuation
            signals.append(
                CompanySignal(
                    company_id=company_id,
                    signal_type=SignalType.VALUATION_UPDATE,
                    timestamp=date,
                    source=source,
                    provenance="derived:outcome",
                    confidence=_VALUATION_CONFIDENCE,
                    evidence=EvidenceReference(
                        reference=source, description="valuation snapshot"
                    ),
                    metadata=metadata,
                )
            )
        return signals

    def _employee_history(
        self,
        company_id: str,
        snapshots: list[dict[str, Any]],
        skips: list[str],
    ) -> list[CompanySignal]:
        dated: list[tuple[datetime, float, str]] = []
        for snapshot in snapshots:
            date = _parse_snapshot_date(snapshot.get("date"))
            if date is None:
                skips.append(f"{company_id}: employee snapshot without date")
                continue
            count = _float_or_none(snapshot.get("count"))
            if count is None:
                skips.append(f"{company_id}: employee snapshot without count")
                continue
            source = str(snapshot.get("source", "outcome"))
            dated.append((date, count, source))

        dated.sort(key=lambda point: (point[0], point[1]))
        signals: list[CompanySignal] = []
        for date, count, source in dated:
            signals.append(
                CompanySignal(
                    company_id=company_id,
                    signal_type=SignalType.EMPLOYEE_MILESTONE,
                    timestamp=date,
                    source=source,
                    provenance="derived:employee_snapshot",
                    confidence=_EMPLOYEE_CONFIDENCE,
                    evidence=EvidenceReference(
                        reference=source, description="employee count snapshot"
                    ),
                    metadata={"count": count},
                )
            )

        for (prev_date, prev_count, _prev_source), (date, count, source) in zip(
            dated, dated[1:]
        ):
            del prev_date
            delta = count - prev_count
            if abs(delta) < 1e-9:
                continue
            if delta > 0:
                signal_type = SignalType.HIRING_GROWTH
                confidence = _EMPLOYEE_CONFIDENCE
                description = "headcount increased between snapshots"
            else:
                signal_type = SignalType.LAYOFFS
                confidence = _EMPLOYEE_CONFIDENCE
                description = "headcount decreased between snapshots"
            signals.append(
                CompanySignal(
                    company_id=company_id,
                    signal_type=signal_type,
                    timestamp=date,
                    source=source,
                    provenance="derived:headcount_delta",
                    confidence=confidence,
                    evidence=EvidenceReference(
                        reference=source, description=description
                    ),
                    metadata={
                        "from_count": prev_count,
                        "to_count": count,
                        "delta": delta,
                    },
                )
            )
        return signals


@dataclass
class SignalBuildReport:
    """Summary of a signal build pass."""

    timeline_count: int = 0
    signal_count: int = 0
    companies: list[str] = field(default_factory=list)
    per_type: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "timeline_count": self.timeline_count,
            "signal_count": self.signal_count,
            "companies": list(self.companies),
            "per_type": dict(sorted(self.per_type.items())),
            "skipped": list(self.skipped),
        }


@dataclass
class SignalBuildResult:
    """Outcome of a signal build pass."""

    timelines: dict[str, CompanyTimeline]
    report: SignalBuildReport

    def to_dict(self) -> dict[str, object]:
        return {
            "timelines": {
                cid: timeline.to_dict()
                for cid, timeline in sorted(self.timelines.items())
            },
            "report": self.report.to_dict(),
        }


def record_to_company_ids(records: list[DatasetRecord]) -> dict[str, str]:
    """Resolve records to canonical company ids (graph-compatible)."""
    from predictron_engine.dataset.entity_resolution import EntityResolver

    resolved = EntityResolver().resolve(records)
    mapping: dict[str, str] = {}
    for identity in resolved.identities:
        identity_id = company_node_id(identity)
        for record_id in identity.record_ids:
            mapping[record_id] = identity_id
    return mapping


class CompanySignalBuilder:
    """Builds deterministic per-company timelines from records + outcomes."""

    def __init__(self) -> None:
        self._extractor = OutcomeSignalExtractor()

    def build(
        self,
        records: list[DatasetRecord],
        outcomes_by_record: dict[str, OutcomeRecord] | None = None,
        *,
        identity_map: dict[str, str] | None = None,
    ) -> SignalBuildResult:
        outcomes = outcomes_by_record or {}
        if identity_map is None:
            identity_map = record_to_company_ids(records)

        gathered: dict[str, list[CompanySignal]] = {}
        skipped: list[str] = []
        for record in sorted(records, key=lambda r: r.record_id):
            company_id = identity_map.get(record.record_id)
            if not company_id:
                company_id = f"company:{record.record_id}"
            outcome = outcomes.get(record.record_id)
            if outcome is None:
                continue
            signals, record_skips = self._extractor.extract(outcome, company_id)
            skipped.extend(record_skips)
            if not signals:
                continue
            gathered.setdefault(company_id, []).extend(signals)

        timelines: dict[str, CompanyTimeline] = {}
        counts: Counter[str] = Counter()
        total = 0
        for company_id in sorted(gathered):
            timeline = CompanyTimeline.build(company_id, gathered[company_id])
            timelines[company_id] = timeline
            total += timeline.signal_count
            counts.update(timeline.types())

        report = SignalBuildReport(
            timeline_count=len(timelines),
            signal_count=total,
            companies=sorted(timelines),
            per_type=dict(sorted(counts.items())),
            skipped=sorted(skipped),
        )
        return SignalBuildResult(timelines=timelines, report=report)
