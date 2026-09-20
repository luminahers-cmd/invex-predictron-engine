"""CompanyReadService — deterministic merge of live and offline company data.

Phase 2 read layer: builds a unified, deterministic :class:`CompanyProfileResponse`
for one company by merging

* the **live** Phase 1 registry (post-analysis snapshots), and
* the **offline** historical dataset (records, graph, signals, features) via
  :class:`~app.services.company_dataset.DatasetCompanyStore`.

Coverage gating
---------------
Live snapshots come from completed analyses and are grounded by construction.
Offline records are classified with deterministic quality gates; a company
whose only offline records are placeholders is reported as
``coverage = "offline_placeholder"`` and its intelligence summaries are
omitted — placeholders are never presented as live intelligence.

The service never mutates any store and returns identical output for
identical inputs pinned to the same ``as_of`` reference.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.schemas.company import CompanySnapshotResponse
from app.schemas.company_profile import (
    CompanyBenchmarkSummary,
    CompanyDecisionSummary,
    CompanyFeatureItem,
    CompanyFeatureSummary,
    CompanyGraphSummary,
    CompanyHistorySummary,
    CompanyProfileOfflineSummary,
    CompanyProfileResponse,
    CompanySignalSummary,
    CoverageStatus,
    ProfileCompleteness,
)
from app.services.companies import CompanyIdentityResolver
from app.services.company_dataset import DatasetCompanyStore

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession

    from app.services.company_protocols import (
        CompanyRecord,
        CompanySnapshotRecord,
        CompanyStore,
    )
    from predictron_engine.dataset.models import DatasetRecord
    from predictron_engine.dataset.signals.timeline import CompanyTimeline

#: Canonical identity / prediction fields tracked for profile completeness.
_COMPLETENESS_FIELDS: tuple[str, ...] = (
    "canonical_domain",
    "website",
    "latest_decision",
    "latest_confidence",
    "latest_composite_score",
)


def _default_dataset_root() -> Path:
    from pathlib import Path

    return Path("data") / "dataset"


def _default_dataset() -> DatasetCompanyStore:
    from predictron_engine.dataset.store import DatasetStore

    return DatasetCompanyStore(DatasetStore(str(_default_dataset_root())))


class CompanyReadService:
    """Assembles the unified company profile from live + offline stores."""

    def __init__(
        self,
        *,
        store: CompanyStore | None = None,
        dataset: DatasetCompanyStore | None = None,
        as_of: datetime | None = None,
    ) -> None:
        from app.services.company_postgres import PostgresCompanyStore

        self._store = store or PostgresCompanyStore()
        self._dataset = dataset or _default_dataset()
        self._as_of = as_of

    async def profile(
        self,
        session: AsyncSession,
        company_id: str,
        *,
        user_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> CompanyProfileResponse | None:
        """Build the unified profile for one company.

        Returns ``None`` when the company is unknown to the live registry
        (same not-found semantics as the Phase 1 endpoints).
        """
        company = await self._store.get_company(session, company_id, user_id=user_id)
        if company is None:
            return None
        page = await self._store.list_snapshots(
            session, company_id, user_id=user_id, offset=offset, limit=limit
        )
        return self._assemble(company, page.snapshots, page.total)

    async def resolve(
        self,
        session: AsyncSession,
        *,
        name: str,
        website: str | None = None,
        domain: str | None = None,
        user_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> CompanyProfileResponse | None:
        """Build a profile from identity signals, falling back to offline only.

        Resolves the deterministic company identity from the supplied signals
        and prefers the live registry when that identity is registered
        (identical to :meth:`profile`). Otherwise it answers entirely from the
        grounded offline dataset, so companies that predate the registry still
        surface a profile. Returns ``None`` when neither source has the company.
        """
        identity = CompanyIdentityResolver().resolve(name, website or domain or "")
        live = await self.profile(
            session, identity.company_id, user_id=user_id, offset=offset, limit=limit
        )
        if live is not None:
            return live

        records = self._dataset.records_for(name=name, website=website, domain=domain)
        offline_company = self._dataset.record_for(records)
        if offline_company is None:
            return None
        return self._assemble(offline_company, [], 0)

    def _assemble(
        self,
        company: CompanyRecord,
        snapshots: list[CompanySnapshotRecord],
        total_snapshots: int,
    ) -> CompanyProfileResponse:
        generated_at = self._reference_time()
        live_has_data = bool(snapshots)

        records = self._dataset.records_for(
            name=company.primary_name or company.canonical_name,
            website=company.website,
            domain=company.canonical_domain,
        )
        offline = self._dataset.offline_summary(records)
        is_placeholder = bool(offline["placeholder"])
        node_id = self._dataset.company_node_id(records) if records else None

        coverage, coverage_reasons = _classify_coverage(
            live_has_data=live_has_data,
            has_offline=bool(records),
            offline_placeholder=is_placeholder,
        )
        grounded_offline = bool(records) and not is_placeholder

        composite_score = (
            snapshots[0].composite_score
            if snapshots
            else (
                records[-1].prediction.composite_score
                if grounded_offline and records
                else None
            )
        )
        latest_record, latest_timeline = _latest_offline_snapshot(
            self._dataset, records, node_id, grounded_offline, generated_at
        )

        return CompanyProfileResponse(
            company_id=company.company_id,
            canonical_name=company.canonical_name,
            primary_name=company.primary_name,
            canonical_domain=company.canonical_domain,
            website=company.website,
            coverage=coverage,
            coverage_reasons=coverage_reasons,
            completeness=_completeness(company),
            latest_snapshot=(
                _snapshot_response(snapshots[0]) if snapshots else None
            ),
            history=_history_summary(company, snapshots, total_snapshots),
            offline=CompanyProfileOfflineSummary(**offline) if records else None,
            graph_summary=_graph_summary(self._dataset, node_id, grounded_offline),
            feature_summary=_feature_summary(
                self._dataset,
                node_id,
                grounded_offline,
                generated_at,
                latest_record,
                latest_timeline,
            ),
            signal_summary=_signal_summary(
                self._dataset, node_id, grounded_offline, generated_at
            ),
            benchmark_summary=(
                CompanyBenchmarkSummary(**self._dataset.benchmark_summary(composite_score))
                if composite_score is not None
                else None
            ),
            decision_summary=(
                CompanyDecisionSummary(
                    **self._dataset.decision_summary(
                        latest_record,
                        company_node_id=node_id,
                        timeline=latest_timeline,
                        as_of=generated_at,
                    )
                )
                if latest_record is not None
                else None
            ),
            generated_at=generated_at,
        )

    def _reference_time(self) -> datetime:
        return self._as_of or datetime.now(UTC)


def _classify_coverage(
    *,
    live_has_data: bool,
    has_offline: bool,
    offline_placeholder: bool,
) -> tuple[CoverageStatus, list[str]]:
    """Deterministically classify profile coverage from the data available."""
    if live_has_data:
        return CoverageStatus.LIVE, ["live analysis snapshots found"]
    if has_offline and offline_placeholder:
        return (
            CoverageStatus.OFFLINE_PLACEHOLDER,
            ["offline records are placeholders"],
        )
    if has_offline:
        return (
            CoverageStatus.OFFLINE,
            ["no live snapshots; grounded offline records found"],
        )
    return CoverageStatus.NONE, ["no company data found"]


def _completeness(company: CompanyRecord) -> ProfileCompleteness:
    """Deterministic identity-field completeness of the live registry row."""
    present: list[str] = []
    for field in _COMPLETENESS_FIELDS:
        value = getattr(company, field)
        if value not in (None, ""):
            present.append(field)
    total = len(_COMPLETENESS_FIELDS)
    populated = len(present)
    fraction = round(populated / total, 6) if total else 0.0
    missing = [field for field in _COMPLETENESS_FIELDS if field not in present]
    return ProfileCompleteness(
        total_fields=total,
        populated_fields=populated,
        fraction=fraction,
        missing_fields=missing,
    )


def _history_summary(
    company: CompanyRecord,
    snapshots: list[CompanySnapshotRecord],
    total: int,
) -> CompanyHistorySummary:
    """Deterministic summary of the live snapshot history."""
    return CompanyHistorySummary(
        snapshot_count=total,
        first_seen=company.first_seen,
        last_seen=company.last_seen,
        latest_decision=snapshots[0].decision if snapshots else company.latest_decision,
        latest_confidence=(
            snapshots[0].confidence if snapshots else company.latest_confidence
        ),
        latest_composite_score=(
            snapshots[0].composite_score
            if snapshots
            else company.latest_composite_score
        ),
    )


def _snapshot_response(snapshot: CompanySnapshotRecord) -> CompanySnapshotResponse:
    return CompanySnapshotResponse(
        id=snapshot.id,
        company_id=snapshot.company_id,
        analysis_id=snapshot.analysis_id,
        report_id=snapshot.report_id,
        decision=snapshot.decision,
        confidence=snapshot.confidence,
        composite_score=snapshot.composite_score,
        readiness_score=snapshot.readiness_score,
        dimension_scores=snapshot.dimension_scores,
        created_at=snapshot.created_at,
    )


def _graph_summary(
    dataset: DatasetCompanyStore,
    node_id: str | None,
    grounded: bool,
) -> CompanyGraphSummary | None:
    """Offline graph summary only when grounded data exists.

    Placeholder offline data is never surfaced as intelligence, so the
    summary is omitted when the record set is a placeholder.
    """
    if not grounded or node_id is None:
        return None
    return CompanyGraphSummary(**dataset.graph_summary(node_id))


def _latest_offline_snapshot(
    dataset: DatasetCompanyStore,
    records: list[DatasetRecord],
    node_id: str | None,
    grounded: bool,
    as_of: datetime,
) -> tuple[DatasetRecord | None, CompanyTimeline | None]:
    """Resolve the latest grounded offline record and its signal timeline.

    Both the feature summary and the decision summary are computed from this
    single deterministic snapshot, so they always agree with each other.
    """
    from predictron_engine.dataset.models import DatasetRecord

    if not grounded:
        return None, None
    latest = next((r for r in reversed(records) if isinstance(r, DatasetRecord)), None)
    if latest is None:
        return None, None
    timeline = dataset.signals().timeline(node_id) if node_id else None
    return latest, timeline


def _feature_summary(
    dataset: DatasetCompanyStore,
    node_id: str | None,
    grounded: bool,
    as_of: datetime,
    latest: DatasetRecord | None = None,
    timeline: CompanyTimeline | None = None,
) -> CompanyFeatureSummary | None:
    if not grounded or latest is None:
        return None
    summary = dataset.feature_summary(
        latest,
        company_node_id=node_id,
        timeline=timeline,
        as_of=as_of,
    )
    features = [
        CompanyFeatureItem(
            feature_id=item["feature_id"],
            feature_name=item["feature_name"],
            category=item["category"],
            value=item["value"],
            value_type=item["value_type"],
            status=item["status"],
        )
        for item in summary["features"]
    ]
    return CompanyFeatureSummary(
        feature_count=summary["feature_count"],
        computed_count=summary["computed_count"],
        failed_count=summary["failed_count"],
        by_category=summary["by_category"],
        features=features,
    )


def _signal_summary(
    dataset: DatasetCompanyStore,
    node_id: str | None,
    grounded: bool,
    as_of: datetime,
) -> CompanySignalSummary | None:
    if not grounded:
        return None
    summary = dataset.signal_summary(node_id, as_of=as_of)
    if summary is None:
        return None
    return CompanySignalSummary(**summary)


__all__ = ["CompanyReadService", "classify_coverage"]

classify_coverage = _classify_coverage
