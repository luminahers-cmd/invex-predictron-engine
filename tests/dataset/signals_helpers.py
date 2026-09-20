"""Shared fixtures and factories for Project E4 signal tests."""

from __future__ import annotations

from datetime import UTC, datetime

from predictron_engine.dataset.outcomes import (
    FundingEvent,
    OutcomeRecord,
    StartupOutcome,
)
from predictron_engine.dataset.signals.model import (
    CompanySignal,
    EvidenceReference,
    SignalType,
)
from predictron_engine.dataset.signals.timeline import CompanyTimeline


def dt(value: str) -> datetime:
    """Parse an ISO datetime string, assuming UTC when naive."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def make_signal(
    company_id: str = "company:a",
    signal_type: SignalType = SignalType.FUNDING_ROUND,
    at: str = "2024-01-15T00:00:00+00:00",
    source: str = "test",
    confidence: float = 1.0,
    provenance: str = "test",
    evidence_ref: str = "test",
    evidence_desc: str = "fixture evidence",
    **metadata: object,
) -> CompanySignal:
    return CompanySignal(
        company_id=company_id,
        signal_type=signal_type,
        timestamp=dt(at),
        source=source,
        confidence=confidence,
        provenance=provenance,
        evidence=EvidenceReference(reference=evidence_ref, description=evidence_desc),
        metadata=dict(metadata),
    )


def make_timeline(
    company_id: str,
    *signals: CompanySignal,
) -> CompanyTimeline:
    return CompanyTimeline.build(company_id, list(signals))


def make_funding_event(
    round_type: str = "seed",
    amount: float | None = 1_000_000.0,
    date: str | None = "2024-01-15T00:00:00+00:00",
    investors: list[str] | None = None,
    valuation: float | None = None,
    source: str = "crunchbase",
) -> FundingEvent:
    return FundingEvent(
        round_type=round_type,
        amount_usd=amount,
        date=dt(date) if date else None,
        investors=investors or [],
        valuation_usd=valuation,
        source=source,
    )


def make_outcome(
    record_id: str = "rec-1",
    *,
    funding_rounds: list[FundingEvent] | None = None,
    acquisition: str | None = None,
    exit_date: str | None = None,
    exit_type: str | None = None,
    shutdown: bool | None = None,
    shutdown_date: str | None = None,
    bankruptcy: bool | None = None,
    bankruptcy_date: str | None = None,
    arr: list[tuple[str, float]] | None = None,
    employees: list[tuple[str, float]] | None = None,
    valuations: list[tuple[str, float]] | None = None,
) -> OutcomeRecord:
    arr_snapshots = [
        {"date": dt(date), "arr_usd": amount, "source": "test"}
        for date, amount in (arr or [])
    ]
    employee_history = [
        {"date": dt(date), "count": count, "source": "test"}
        for date, count in (employees or [])
    ]
    valuation_history = [
        {"date": dt(date), "valuation_usd": amount, "source": "test"}
        for date, amount in (valuations or [])
    ]
    outcome = StartupOutcome(
        funding_rounds=list(funding_rounds or []),
        acquisition=acquisition,
        exit_date=dt(exit_date) if exit_date else None,
        exit_type=exit_type,
        shutdown=shutdown,
        shutdown_date=dt(shutdown_date) if shutdown_date else None,
        bankruptcy=bankruptcy,
        bankruptcy_date=dt(bankruptcy_date) if bankruptcy_date else None,
        arr_milestones=arr_snapshots,
        employee_count_history=employee_history,
        valuation_history=valuation_history,
    )
    return OutcomeRecord(record_id=record_id, outcome=outcome)
