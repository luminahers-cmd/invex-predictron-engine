"""CompanyDeltaService and the deterministic delta engine (Phase 3).

The delta engine compares two chronologically ordered snapshots and produces
structured, field-level :class:`~app.schemas.company_delta.CompanyDelta`
objects. Comparisons are pure and deterministic:

* numeric fields (confidence, readiness, composite score, benchmark
  percentile / z-score, evidence count) report ``change = current - previous``,
* nominal fields (decision) report ``"same"`` / ``"different"``,
* collection fields (decision explanation, key facts) compare their
  deterministic value sets.

Missing values are first-class: a field absent from either snapshot is
reported as ``added`` / ``removed`` / ``missing`` instead of inventing a
number. Nothing here predicts or extrapolates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.schemas.company_delta import (
    CompanyDelta,
    DeltaField,
    DeltaKind,
    DeltaStatus,
)
from app.schemas.company_history import CompanyTrendSummary
from app.services.company_history import (
    CHANGE_EPSILON,
    CompanyHistoryService,
    SnapshotView,
    trend_summary,
)
from app.services.company_protocols import CompanyRecord, CompanyStore

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

#: Numeric fields compared as ``current - previous``.
_NUMERIC_FIELDS: tuple[tuple[str, Callable[[SnapshotView], float | None]], ...] = (
    ("confidence", lambda view: view.snapshot.confidence),
    ("readiness", lambda view: view.snapshot.readiness_score),
    ("composite_score", lambda view: view.snapshot.composite_score),
    (
        "benchmark_percentile",
        lambda view: (
            view.benchmark.percentile_rank if view.benchmark is not None else None
        ),
    ),
    (
        "benchmark_z_score",
        lambda view: view.benchmark.z_score if view.benchmark is not None else None,
    ),
    (
        "evidence_count",
        lambda view: _optional_float(view.evidence_count),
    ),
)

#: Nominal fields compared for equality.
_NOMINAL_FIELDS: tuple[tuple[str, Callable[[SnapshotView], str | None]], ...] = (
    ("decision", lambda view: view.snapshot.decision),
)

#: Collection fields compared as deterministic value sets.
_COLLECTION_FIELDS: tuple[
    tuple[str, Callable[[SnapshotView], list[str] | None]], ...
] = (
    (
        "decision_explanation",
        lambda view: (
            list(view.decision_explanation) if view.decision_explanation else None
        ),
    ),
    ("key_facts", lambda view: list(view.key_facts) if view.key_facts else None),
)


def _optional_float(value: int | None) -> float | None:
    return float(value) if value is not None else None


def compare_snapshots(previous: SnapshotView, current: SnapshotView) -> CompanyDelta:
    """Compare an earlier snapshot against a later snapshot.

    ``change`` values always read ``earlier -> later`` regardless of the
    order the resulting delta is served in.
    """
    fields: list[DeltaField] = []
    for name, numeric_getter in _NUMERIC_FIELDS:
        fields.append(
            _compare_numeric(name, numeric_getter(previous), numeric_getter(current))
        )
    for name, nominal_getter in _NOMINAL_FIELDS:
        fields.append(
            _compare_nominal(name, nominal_getter(previous), nominal_getter(current))
        )
    for name, collection_getter in _COLLECTION_FIELDS:
        fields.append(
            _compare_collection(
                name, collection_getter(previous), collection_getter(current)
            )
        )

    changed = sum(1 for field in fields if field.status == DeltaStatus.CHANGED)
    unchanged = sum(1 for field in fields if field.status == DeltaStatus.UNCHANGED)
    return CompanyDelta(
        previous_snapshot_id=previous.snapshot.id,
        current_snapshot_id=current.snapshot.id,
        previous_created_at=previous.snapshot.created_at,
        current_created_at=current.snapshot.created_at,
        fields=fields,
        changed_fields=changed,
        unchanged_fields=unchanged,
    )


def build_delta_sequence(views: Sequence[SnapshotView]) -> list[CompanyDelta]:
    """Compare every adjacent pair of a chronological snapshot series."""
    return [
        compare_snapshots(previous, current)
        for previous, current in zip(views, views[1:])
    ]


def _compare_numeric(
    name: str, previous: float | None, current: float | None
) -> DeltaField:
    if previous is None and current is None:
        return DeltaField(
            field=name, delta_type=DeltaKind.NUMERIC, status=DeltaStatus.MISSING
        )
    if previous is None:
        return DeltaField(
            field=name,
            delta_type=DeltaKind.NUMERIC,
            status=DeltaStatus.ADDED,
            current=current,
        )
    if current is None:
        return DeltaField(
            field=name,
            delta_type=DeltaKind.NUMERIC,
            status=DeltaStatus.REMOVED,
            previous=previous,
        )
    change = round(current - previous, 6)
    status = (
        DeltaStatus.UNCHANGED
        if abs(change) <= CHANGE_EPSILON
        else DeltaStatus.CHANGED
    )
    return DeltaField(
        field=name,
        delta_type=DeltaKind.NUMERIC,
        status=status,
        previous=previous,
        current=current,
        change=change,
    )


def _compare_nominal(
    name: str, previous: str | None, current: str | None
) -> DeltaField:
    return _compare_values(name, DeltaKind.NOMINAL, previous, current)


def _compare_collection(
    name: str, previous: list[str] | None, current: list[str] | None
) -> DeltaField:
    return _compare_values(name, DeltaKind.COLLECTION, previous, current)


def _compare_values(
    name: str,
    kind: DeltaKind,
    previous: object | None,
    current: object | None,
) -> DeltaField:
    if previous is None and current is None:
        return DeltaField(
            field=name, delta_type=kind, status=DeltaStatus.MISSING
        )
    if previous is None:
        return DeltaField(
            field=name,
            delta_type=kind,
            status=DeltaStatus.ADDED,
            current=current,
        )
    if current is None:
        return DeltaField(
            field=name,
            delta_type=kind,
            status=DeltaStatus.REMOVED,
            previous=previous,
        )
    status = (
        DeltaStatus.UNCHANGED if previous == current else DeltaStatus.CHANGED
    )
    marker = "same" if status == DeltaStatus.UNCHANGED else "different"
    return DeltaField(
        field=name,
        delta_type=kind,
        status=status,
        previous=previous,
        current=current,
        change=marker,
    )


@dataclass
class CompanyDeltaResult:
    """Company delta page plus the deterministic trend summary."""

    company: CompanyRecord
    deltas: list[CompanyDelta]
    total: int
    trend: CompanyTrendSummary


class CompanyDeltaService:
    """Derives the deterministic delta sequence for one company.

    Reuses :class:`CompanyHistoryService` for store access, chronological
    ordering, report linkage, and enrichment so the delta and timeline views
    always agree about the underlying snapshots.
    """

    def __init__(
        self,
        *,
        history: CompanyHistoryService | None = None,
        store: CompanyStore | None = None,
    ) -> None:
        if history is not None:
            self._history = history
        else:
            self._history = CompanyHistoryService(store=store)

    async def deltas(
        self,
        session: AsyncSession | None,
        company_id: str,
        *,
        user_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> CompanyDeltaResult | None:
        """Delta sequence for one company, newest-first, with pagination."""
        company, views = await self._history.fetch_all_views(
            session, company_id, user_id=user_id
        )
        if company is None:
            return None
        ordered = build_delta_sequence(views)
        ordered.sort(
            key=lambda delta: (delta.current_created_at, delta.previous_snapshot_id),
            reverse=True,
        )
        page = ordered[offset : offset + limit]
        return CompanyDeltaResult(
            company=company,
            deltas=page,
            total=len(ordered),
            trend=trend_summary(views),
        )


__all__ = [
    "CompanyDeltaResult",
    "CompanyDeltaService",
    "build_delta_sequence",
    "compare_snapshots",
]
