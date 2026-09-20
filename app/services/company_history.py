"""CompanyHistoryService — the Phase 3 temporal read layer over the registry.

Phase 3 adds deterministic temporal intelligence on top of the Phase 1
registry and the Phase 2 offline linkage. This module owns:

* :class:`SnapshotView` — one stored snapshot enriched with the derived
  temporal fields (benchmark placement, decision explanation, key facts,
  evidence count) that the delta engine compares.
* :class:`CompanyHistoryService` — read-side orchestration: chronological
  ordering, persisted-report linkage, benchmark derivation, pagination, and
  the lightweight deterministic trend engine.

Everything here derives from stored data only: the immutable snapshot rows
and the persisted analysis report JSON (``analysis_reports.full_report``)
they reference, plus the benchmark population from the offline dataset
adapter. No prediction, no ML, no generated prose.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.schemas.company_history import CompanyTrendSummary, TrendDirection
from app.schemas.company_profile import CompanyBenchmarkSummary
from app.services.company_postgres import PostgresCompanyStore
from app.services.company_protocols import (
    CompanyRecord,
    CompanySnapshotRecord,
    CompanyStore,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.services.company_dataset import DatasetCompanyStore

    #: A report resolver maps a persisted report id to its stored JSON payload.
    ReportResolverCallable = Callable[
        [AsyncSession | None, str], Awaitable[dict[str, object] | None]
    ]


#: Float comparison tolerance used by the deterministic trend engine.
CHANGE_EPSILON = 1e-9

#: Keys inside the persisted analysis report JSON consumed for enrichment.
_REPORT_EVIDENCE_KEY = "evidence"
_REPORT_RECOMMENDATIONS_KEY = "recommendations"
_REPORT_INVESTMENT_DECISION_KEY = "investment_decision"
_REPORT_RATIONALE_KEY = "rationale"
_REPORT_INVESTMENT_READINESS_KEY = "investment_readiness"
_REPORT_READINESS_LEVEL_KEY = "readiness_level"
_REPORT_READINESS_STRENGTHS_KEY = "key_strengths"
_REPORT_READINESS_CONCERNS_KEY = "key_concerns"

_RATIONALE_STRING_KEYS: tuple[str, ...] = (
    "key_evidence_summary",
    "confidence_explanation",
    "why_not_higher",
)
_RATIONALE_LIST_KEYS: tuple[str, ...] = (
    "primary_reasons_for",
    "primary_reasons_against",
    "highest_impact_positive",
    "highest_impact_negative",
    "missing_information",
    "information_that_could_change_decision",
)


def default_dataset_root() -> Path:
    """Default offline dataset root, mirroring the Phase 2 profile layer."""
    from pathlib import Path

    return Path("data") / "dataset"


def default_dataset() -> DatasetCompanyStore:
    """Build the default offline dataset adapter for benchmark enrichment."""
    from app.services.company_dataset import DatasetCompanyStore
    from predictron_engine.dataset.store import DatasetStore

    return DatasetCompanyStore(DatasetStore(str(default_dataset_root())))


async def default_report_resolver(
    session: AsyncSession | None, report_id: str
) -> dict[str, object] | None:
    """Resolve the persisted analysis report JSON for one report id."""
    if session is None:
        return None
    from sqlalchemy import select

    from app.models.analysis import AnalysisReport

    stmt = select(AnalysisReport.full_report).where(AnalysisReport.id == report_id)
    result = await session.execute(stmt)
    payload = result.scalar_one_or_none()
    if not isinstance(payload, dict):
        return None
    return payload


@dataclass(frozen=True)
class SnapshotView:
    """One stored snapshot plus its deterministically derived temporal fields."""

    snapshot: CompanySnapshotRecord
    benchmark: CompanyBenchmarkSummary | None = None
    decision_explanation: tuple[str, ...] = ()
    key_facts: tuple[str, ...] = ()
    evidence_count: int | None = None


@dataclass
class CompanyHistoryResult:
    """Company history page plus its deterministic trend summary."""

    company: CompanyRecord
    entries: list[SnapshotView]
    total: int
    trend: CompanyTrendSummary = field(
        default_factory=lambda: trend_summary([])
    )


class CompanyHistoryService:
    """Read-side orchestration of the temporal registry view.

    Instances are stateless and accept an explicit ``AsyncSession`` per
    call, mirroring the store and profile conventions. ``dataset`` is
    optional — when absent, benchmark enrichment is skipped (derived
    ``benchmark`` fields are ``None``).
    """

    def __init__(
        self,
        *,
        store: CompanyStore | None = None,
        dataset: DatasetCompanyStore | None = None,
        report_resolver: ReportResolverCallable | None = None,
    ) -> None:
        self._store = store or PostgresCompanyStore()
        self._dataset = dataset
        self._report_resolver = report_resolver or default_report_resolver
        self._benchmark_cache: dict[float, CompanyBenchmarkSummary] = {}

    async def timeline(
        self,
        session: AsyncSession | None,
        company_id: str,
        *,
        user_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> CompanyHistoryResult | None:
        """Paginated enriched timeline, newest-first, scoped to the user."""
        company, views = await self.fetch_all_views(session, company_id, user_id=user_id)
        if company is None:
            return None
        page = list(reversed(views))[offset : offset + limit]
        return CompanyHistoryResult(
            company=company,
            entries=page,
            total=len(views),
            trend=trend_summary(views),
        )

    async def latest(
        self,
        session: AsyncSession | None,
        company_id: str,
        *,
        user_id: str | None = None,
    ) -> CompanyHistoryResult | None:
        """The most recent enriched snapshot plus the company-level trend."""
        company, views = await self.fetch_all_views(session, company_id, user_id=user_id)
        if company is None:
            return None
        latest_view = views[-1] if views else None
        return CompanyHistoryResult(
            company=company,
            entries=[latest_view] if latest_view is not None else [],
            total=len(views),
            trend=trend_summary(views),
        )

    async def fetch_all_views(
        self,
        session: AsyncSession | None,
        company_id: str,
        *,
        user_id: str | None = None,
    ) -> tuple[CompanyRecord | None, list[SnapshotView]]:
        """Resolve one company and its full enriched history (chronological).

        Returns ``(company, views)`` where ``company`` is ``None`` when the
        company is unknown to (or out of scope for) the caller, mirroring the
        store's not-found semantics.
        """
        assert session is not None
        company = await self._store.get_company(session, company_id, user_id=user_id)
        if company is None:
            return None, []
        snapshots = await self.fetch_all_snapshots(session, company_id, user_id=user_id)
        views = await self.enrich_views(session, snapshots)
        return company, views

    async def fetch_all_snapshots(
        self,
        session: AsyncSession | None,
        company_id: str,
        *,
        user_id: str | None = None,
    ) -> list[CompanySnapshotRecord]:
        """Fetch the complete snapshot history in chronological order."""
        assert session is not None
        collected: list[CompanySnapshotRecord] = []
        offset = 0
        page_size = 100
        while True:
            page = await self._store.list_snapshots(
                session, company_id, user_id=user_id, offset=offset, limit=page_size
            )
            collected.extend(page.snapshots)
            if len(page.snapshots) < page_size:
                break
            offset += page_size
        collected.sort(key=lambda snap: (snap.created_at, snap.id))
        return collected

    async def enrich_views(
        self,
        session: AsyncSession | None,
        snapshots: Sequence[CompanySnapshotRecord],
    ) -> list[SnapshotView]:
        """Enrich a batch of snapshots, resolving report payloads once each."""
        reports = await self._load_reports(session, snapshots)
        return [
            self.enrich_snapshot(snapshot, reports.get(snapshot.report_id))
            for snapshot in snapshots
        ]

    def enrich_snapshot(
        self,
        snapshot: CompanySnapshotRecord,
        report: dict[str, object] | None = None,
    ) -> SnapshotView:
        """Enrich one snapshot from its stored report JSON (if any).

        When the report is unavailable the report-derived fields stay empty
        and ``evidence_count`` stays ``None`` — derived fields never
        fabricate values.
        """
        benchmark = self._benchmark_for(snapshot.composite_score)
        key_facts = snapshot_key_facts(snapshot)
        if report is None:
            return SnapshotView(
                snapshot=snapshot,
                benchmark=benchmark,
                decision_explanation=(),
                key_facts=key_facts,
                evidence_count=None,
            )
        return SnapshotView(
            snapshot=snapshot,
            benchmark=benchmark,
            decision_explanation=extract_decision_explanation(report),
            key_facts=extract_key_facts(report, snapshot),
            evidence_count=extract_evidence_count(report),
        )

    def _benchmark_for(self, score: float | None) -> CompanyBenchmarkSummary | None:
        """Benchmark placement of one composite score, cached per value."""
        if score is None or self._dataset is None:
            return None
        cached = self._benchmark_cache.get(score)
        if cached is not None:
            return cached
        summary = CompanyBenchmarkSummary(**self._dataset.benchmark_summary(score))
        self._benchmark_cache[score] = summary
        return summary

    async def _load_reports(
        self,
        session: AsyncSession | None,
        snapshots: Sequence[CompanySnapshotRecord],
    ) -> dict[str, dict[str, object]]:
        report_ids = sorted({snapshot.report_id for snapshot in snapshots})
        reports: dict[str, dict[str, object]] = {}
        for report_id in report_ids:
            payload = await self._report_resolver(session, report_id)
            if payload is not None:
                reports[report_id] = payload
        return reports


def extract_evidence_count(report: dict[str, object]) -> int | None:
    """Deterministic evidence count from a stored analysis report payload."""
    items = report.get(_REPORT_EVIDENCE_KEY)
    if isinstance(items, list):
        return len(items)
    return None


def extract_decision_explanation(report: dict[str, object]) -> tuple[str, ...]:
    """Deterministic explanation strings from the stored decision rationale."""
    parts: list[str] = []
    decision = report.get(_REPORT_INVESTMENT_DECISION_KEY)
    rationale = (
        decision.get(_REPORT_RATIONALE_KEY) if isinstance(decision, dict) else None
    )
    if not isinstance(rationale, dict):
        return ()
    for key in _RATIONALE_STRING_KEYS:
        value = rationale.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    for key in _RATIONALE_LIST_KEYS:
        values = rationale.get(key)
        if isinstance(values, list):
            for item in values:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
    return tuple(parts)


def snapshot_key_facts(snapshot: CompanySnapshotRecord) -> tuple[str, ...]:
    """Deterministic key facts derived directly from a stored snapshot."""
    facts: list[str] = []
    if snapshot.decision is not None:
        facts.append(f"decision={snapshot.decision}")
    if snapshot.confidence is not None:
        facts.append(f"confidence={snapshot.confidence:g}")
    if snapshot.readiness_score is not None:
        facts.append(f"readiness_score={snapshot.readiness_score:g}")
    if snapshot.composite_score is not None:
        facts.append(f"composite_score={snapshot.composite_score:g}")
    for dimension, score in sorted(snapshot.dimension_scores.items()):
        facts.append(f"dimension.{dimension}={score:g}")
    return tuple(sorted(facts))


def extract_key_facts(
    report: dict[str, object], snapshot: CompanySnapshotRecord
) -> tuple[str, ...]:
    """Key facts derived from the stored snapshot plus its stored report."""
    facts = list(snapshot_key_facts(snapshot))
    recommendations = report.get(_REPORT_RECOMMENDATIONS_KEY)
    if isinstance(recommendations, list):
        for item in recommendations:
            if isinstance(item, str) and item.strip():
                facts.append(f"recommendation={item.strip()}")
            elif isinstance(item, dict):
                action = item.get("action")
                if isinstance(action, str) and action.strip():
                    facts.append(f"recommendation={action.strip()}")
    readiness = report.get(_REPORT_INVESTMENT_READINESS_KEY)
    if isinstance(readiness, dict):
        level = readiness.get(_REPORT_READINESS_LEVEL_KEY)
        if isinstance(level, str) and level.strip():
            facts.append(f"readiness_level={level.strip()}")
        for key, fact_prefix in (
            (_REPORT_READINESS_STRENGTHS_KEY, "key_strength"),
            (_REPORT_READINESS_CONCERNS_KEY, "key_concern"),
        ):
            values = readiness.get(key)
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, str) and item.strip():
                        facts.append(f"{fact_prefix}={item.strip()}")
    return tuple(sorted(set(facts)))


def classify_direction(values: Sequence[float | None]) -> TrendDirection:
    """Classify one chronological series into a deterministic trend direction.

    Requires at least two present values; every usable consecutive change
    must point the same way for ``increasing``/``decreasing``. This is a pure
    description of recorded history — never a prediction or extrapolation.
    """
    present = [value for value in values if value is not None]
    if len(present) < 2:
        return TrendDirection.INSUFFICIENT
    changes = [current - previous for previous, current in zip(present, present[1:])]
    has_up = any(change > CHANGE_EPSILON for change in changes)
    has_down = any(change < -CHANGE_EPSILON for change in changes)
    if has_up and has_down:
        return TrendDirection.MIXED
    if has_up:
        return TrendDirection.INCREASING
    if has_down:
        return TrendDirection.DECREASING
    return TrendDirection.STABLE


def trend_summary(views: Sequence[SnapshotView]) -> CompanyTrendSummary:
    """Deterministic trend directions across the tracked numeric series."""
    return CompanyTrendSummary(
        confidence=classify_direction(
            [view.snapshot.confidence for view in views]
        ),
        readiness=classify_direction(
            [view.snapshot.readiness_score for view in views]
        ),
        composite_score=classify_direction(
            [view.snapshot.composite_score for view in views]
        ),
        benchmark_percentile=classify_direction(
            [
                view.benchmark.percentile_rank if view.benchmark is not None else None
                for view in views
            ]
        ),
        snapshot_count=len(views),
    )


__all__ = [
    "CHANGE_EPSILON",
    "CompanyHistoryResult",
    "CompanyHistoryService",
    "ReportResolverCallable",
    "SnapshotView",
    "classify_direction",
    "default_dataset",
    "default_dataset_root",
    "default_report_resolver",
    "extract_decision_explanation",
    "extract_evidence_count",
    "extract_key_facts",
    "snapshot_key_facts",
    "trend_summary",
]
