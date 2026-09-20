"""CompanyOutcomeService — recording and listing real-world outcomes (Phase 4).

Outcomes are observed facts, never predictions. This module owns:

* deterministic outcome identity (:func:`compute_outcome_id`) so
  re-submitting the *same* observation is an idempotent no-op, mirroring the
  registry's snapshot-append semantics;
* :func:`outcome_verdict` — the verdict is derived through the engine's own
  :meth:`OutcomeRecord.derive_verdict`, never invented here;
* :func:`outcome_record_from_row` — rebuilds the engine's
  :class:`OutcomeRecord` view from a stored row so the existing evaluation
  machinery can be reused verbatim.

Recording also triggers :class:`CompanyEvaluationService` reconciliation so
newly applicable prediction evaluations are generated immediately.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import desc, func, select

from app.models.company import CompanySnapshot
from app.models.company_outcome import CompanyOutcome
from app.services.company_postgres import PostgresCompanyStore
from predictron_engine.dataset.outcomes import (
    OutcomeRecord,
    OutcomeVerdict,
    StartupOutcome,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.schemas.outcome import OutcomeCreateRequest

_OUTCOME_ID_SEPARATOR = "\x00"


class OutcomeSnapshotNotFoundError(LookupError):
    """Raised when a requested ``snapshot_id`` does not belong to the company."""


@dataclass(frozen=True)
class OutcomeWriteResult:
    """Result of recording one outcome plus its idempotency/evaluation info."""

    outcome: CompanyOutcome
    created: bool
    evaluations_created: int


@dataclass(frozen=True)
class OutcomePageResult:
    """One page of stored outcomes (newest-first) and the total count."""

    outcomes: list[CompanyOutcome]
    total: int


def _outcome_signature(outcome: StartupOutcome) -> str:
    """Canonical JSON signature of an observed outcome payload."""
    return json.dumps(outcome.model_dump(mode="json"), sort_keys=True, default=str)


def compute_outcome_id(
    *,
    company_id: str,
    snapshot_id: str | None,
    source: str,
    occurred_at: datetime,
    outcome: StartupOutcome,
) -> str:
    """Deterministic outcome ID derived from the observed fact.

    Identity is ``sha256(company_id, snapshot_id, source, occurred_at,
    canonical-json(outcome))``. Two submissions describing the same
    observation of the same company therefore collapse onto one row.
    """
    material = _OUTCOME_ID_SEPARATOR.join(
        [
            company_id,
            snapshot_id or "",
            source,
            occurred_at.isoformat(),
            _outcome_signature(outcome),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def outcome_verdict(outcome: StartupOutcome) -> OutcomeVerdict:
    """Derive the categorical outcome verdict via the engine's own logic."""
    return OutcomeRecord(record_id="", outcome=outcome).derive_verdict()


def outcome_record_from_row(row: CompanyOutcome) -> OutcomeRecord:
    """Rebuild the engine's :class:`OutcomeRecord` view from a stored row."""
    return OutcomeRecord(
        outcome_id=row.id,
        record_id=row.snapshot_id or row.company_id,
        outcome=StartupOutcome.model_validate(row.outcome_data),
        verdict=OutcomeVerdict(row.verdict),
        verdict_reasoning=row.verdict_reasoning,
        time_horizon_days=row.time_horizon_days,
        created_at=row.created_at,
        updated_at=row.updated_at,
        notes=row.notes,
    )


def as_utc(value: datetime | None) -> datetime | None:
    """Normalize a possibly-naive datetime to an aware UTC datetime."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class CompanyOutcomeService:
    """Persist and list company outcomes (stateless; session per call)."""

    def __init__(self, *, store: PostgresCompanyStore | None = None) -> None:
        self._store = store or PostgresCompanyStore()

    async def record(
        self,
        session: AsyncSession,
        company_id: str,
        request: OutcomeCreateRequest,
        *,
        user_id: str | None = None,
    ) -> OutcomeWriteResult | None:
        """Record one observed outcome, idempotently.

        Returns ``None`` when the company is not found in the caller's scope;
        raises :class:`OutcomeSnapshotNotFoundError` when ``snapshot_id`` is
        given but does not belong to the company. A duplicate observation
        (same deterministic id) is returned with ``created=False``.
        """
        company = await self._store.get_company(session, company_id, user_id=user_id)
        if company is None:
            return None

        snapshot: CompanySnapshot | None = None
        if request.snapshot_id is not None:
            snapshot = await self._load_snapshot(
                session, company_id, request.snapshot_id
            )
            if snapshot is None:
                raise OutcomeSnapshotNotFoundError(
                    f"Snapshot {request.snapshot_id} not found for company {company_id}"
                )

        occurred_at = as_utc(request.occurred_at) or datetime.now(UTC)
        outcome = request.outcome
        verdict = outcome_verdict(outcome)

        time_horizon_days: int | None = None
        if snapshot is not None:
            snapshot_time = as_utc(snapshot.created_at)
            if snapshot_time is not None and occurred_at >= snapshot_time:
                time_horizon_days = (occurred_at - snapshot_time).days

        outcome_id = compute_outcome_id(
            company_id=company_id,
            snapshot_id=request.snapshot_id,
            source=request.source,
            occurred_at=occurred_at,
            outcome=outcome,
        )

        existing = await session.get(CompanyOutcome, outcome_id)
        if existing is not None:
            row = existing
            created = False
        else:
            row = CompanyOutcome(
                id=outcome_id,
                company_id=company_id,
                snapshot_id=request.snapshot_id,
                source=request.source,
                occurred_at=occurred_at,
                time_horizon_days=time_horizon_days,
                outcome_data=outcome.model_dump(mode="json"),
                verdict=verdict.value,
                verdict_reasoning="",
                status=outcome.status.value,
                notes=request.notes,
            )
            session.add(row)
            await session.flush()
            created = True

        from app.services.company_evaluation import CompanyEvaluationService

        evaluations_created = await CompanyEvaluationService(
            store=self._store
        ).reconcile(session, company_id, user_id=user_id)

        return OutcomeWriteResult(
            outcome=row, created=created, evaluations_created=evaluations_created
        )

    async def list(
        self,
        session: AsyncSession,
        company_id: str,
        *,
        user_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> OutcomePageResult | None:
        """List a company's outcomes, newest-first, scoped to the caller."""
        company = await self._store.get_company(session, company_id, user_id=user_id)
        if company is None:
            return None

        count_over = func.count().over().label("total_count")
        stmt = (
            select(CompanyOutcome, count_over)
            .where(CompanyOutcome.company_id == company_id)
            .order_by(desc(CompanyOutcome.created_at), desc(CompanyOutcome.id))
            .offset(offset)
            .limit(limit)
        )
        rows = (await session.execute(stmt)).all()
        total = rows[0].total_count if rows else 0
        return OutcomePageResult(outcomes=[row for row, _ in rows], total=total)

    async def _load_snapshot(
        self, session: AsyncSession, company_id: str, snapshot_id: str
    ) -> CompanySnapshot | None:
        stmt = select(CompanySnapshot).where(
            CompanySnapshot.id == snapshot_id,
            CompanySnapshot.company_id == company_id,
        )
        return (await session.execute(stmt)).scalar_one_or_none()


__all__ = [
    "CompanyOutcomeService",
    "OutcomePageResult",
    "OutcomeSnapshotNotFoundError",
    "OutcomeWriteResult",
    "as_utc",
    "compute_outcome_id",
    "outcome_record_from_row",
    "outcome_verdict",
]
