"""Company registry backfill — ingest legacy analyses into the CIH registry.

Roadmap item 6: companies analyzed *before* the registry milestone are
backfilled from the existing ``analysis_reports`` / ``analysis_requests``
tables so their history becomes first-class registry data.

Two deterministic entry points share the same pure mapping
(:func:`build_backfill_inputs`) and the same identity resolver:

* :meth:`CompanyBackfillService.backfill_all` — the canonical async path,
  used by the application. It reuses ``CompanyIngestService`` so backfilled
  rows go through the exact same resolve -> upsert/adopt -> append-snapshot
  code as live analyses (including atomic ``ON CONFLICT`` appends).

* :func:`backfill_sync` — the same semantics implemented against a
  synchronous SQLAlchemy connection, invoked by the ``0006`` Alembic data
  migration. Migrations run on a sync engine, so they cannot await the async
  store; both paths produce identical rows for identical inputs.

Both paths are idempotent: re-running yields no additional snapshots and
never double-increments ``snapshot_count``. A company row created by the
migration is recorded in the ``company_registry_backfill`` tracking table so
``downgrade`` can remove only the data this milestone introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.models.analysis import AnalysisReport
from app.models.company import Company
from app.services.companies import (
    AnalysisSnapshotInputs,
    CompanyIdentityResolver,
    CompanyIngestService,
)
from app.services.company_postgres import compute_snapshot_id
from app.services.company_protocols import CompanyIdentity

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class BackfillSummary:
    """Outcome counters of one backfill pass.

    ``created_company_ids`` feeds the migration's tracking table so a
    downgrade can cleanly remove only the registry rows this backfill
    introduced.
    """

    reports_scanned: int = 0
    snapshots_created: int = 0
    snapshots_skipped: int = 0
    companies_created: int = 0
    companies_adopted: int = 0
    created_company_ids: list[str] = field(default_factory=list)


def build_backfill_inputs(
    report: AnalysisReport,
    request: object | None,
) -> AnalysisSnapshotInputs:
    """Map one legacy (report, request) pair to CIH snapshot inputs.

    Pure and deterministic. Derived fields follow the modern ingest path:

    * ``decision`` / ``composite_score`` come from the stored ``full_report``
      JSON (investment decision + overall score) when present; otherwise the
      composite falls back to the highest of the four legacy dimension
      scores and the decision is omitted.
    * ``dimension_scores`` mirror the legacy venture/market/founder/traction
      breakdown.
    """
    website = None
    request_name = ""
    request_created_at = None
    if request is not None:
        request_name = str(getattr(request, "startup_name", "") or "")
        request_website = getattr(request, "website", None)
        if request_website:
            website = str(request_website)
        request_created_at = getattr(request, "created_at", None)

    name = (report.startup_name or "").strip() or request_name.strip()

    decision, composite = _legacy_decision_and_composite(
        report.full_report,
        report.venture_score,
        report.market_score,
        report.founder_score,
        report.traction_score,
    )

    return AnalysisSnapshotInputs(
        startup_name=name,
        website=website or None,
        decision=decision,
        confidence=report.confidence,
        composite_score=composite,
        readiness_score=None,
        dimension_scores=_dimension_scores(report),
        created_at=request_created_at or report.created_at,
    )


def _dimension_scores(report: AnalysisReport) -> dict[str, float]:
    return {
        "venture": report.venture_score,
        "market": report.market_score,
        "founder": report.founder_score,
        "traction": report.traction_score,
    }


def _legacy_decision_and_composite(
    full_report: dict[str, object],
    venture: float,
    market: float,
    founder: float,
    traction: float,
) -> tuple[str | None, float]:
    """Extract decision + composite from a legacy report dump (deterministic)."""
    composite: float | None = None
    decision: str | None = None

    if isinstance(full_report, dict):
        inv = full_report.get("investment_decision")
        if isinstance(inv, dict):
            composite = _to_float(inv.get("composite_score"))
            decision = _read_str(inv.get("category"))
        if composite is None:
            composite = _to_float(full_report.get("overall_score"))

    if composite is None:
        composite = float(max(venture, market, founder, traction))
    return decision, composite


def _to_float(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _read_str(value: object) -> str | None:
    """Coerce an enum/enum-dump/str decision value into a plain string."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        return _read_str(value.get("value"))
    if value is not None and hasattr(value, "value"):
        return _read_str(value.value)
    return None


class CompanyBackfillService:
    """Application-side backfill that reuses the live ingest pipeline.

    Deterministic and idempotent: identical inputs produce identical rows,
    and duplicate analysis IDs never create duplicate snapshots.
    """

    def __init__(
        self,
        *,
        store=None,
        resolver: CompanyIdentityResolver | None = None,
        ingest: CompanyIngestService | None = None,
    ) -> None:
        self._resolver = resolver or CompanyIdentityResolver()
        self._ingest = ingest or CompanyIngestService(
            store=store, resolver=self._resolver
        )

    async def backfill_all(
        self,
        session: AsyncSession,
        *,
        user_id: str | None = None,
    ) -> BackfillSummary:
        """Backfill every stored legacy analysis into the registry."""
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        reports = (
            await session.execute(
                select(AnalysisReport)
                .options(selectinload(AnalysisReport.request))
                .order_by(AnalysisReport.created_at)
            )
        ).scalars().all()

        summary = BackfillSummary(reports_scanned=len(reports))
        seen_company_ids: set[str] = set()

        for report in reports:
            request = report.request
            inputs = build_backfill_inputs(report, request)
            identity = self._resolver.resolve(inputs.startup_name, inputs.website)

            if identity.company_id not in seen_company_ids:
                seen_company_ids.add(identity.company_id)
                if await self._company_known(session, identity):
                    summary.companies_adopted += 1
                else:
                    summary.companies_created += 1
                    summary.created_company_ids.append(identity.company_id)

            owner = (
                request.user_id if request is not None and request.user_id else user_id
            )
            result = await self._ingest.ingest_analysis(
                session,
                startup_name=inputs.startup_name,
                website=inputs.website,
                analysis_id=report.id,
                report_id=report.id,
                decision=inputs.decision,
                confidence=inputs.confidence,
                composite_score=inputs.composite_score,
                readiness_score=inputs.readiness_score,
                dimension_scores=inputs.dimension_scores,
                user_id=owner,
                created_at=inputs.created_at,
            )
            if result.created_snapshot:
                summary.snapshots_created += 1
            else:
                summary.snapshots_skipped += 1

        return summary

    async def _company_known(
        self, session: AsyncSession, identity: CompanyIdentity
    ) -> bool:
        """Whether any registry row already represents this identity.

        Checks the deterministic company id and the canonical domain — the
        two adoption signals that decide between "created" and "adopted".
        """
        from sqlalchemy import select

        if (await session.get(Company, identity.company_id)) is not None:
            return True
        if identity.canonical_domain:
            existing = (
                await session.execute(
                    select(Company).where(
                        Company.canonical_domain == identity.canonical_domain
                    )
                )
            ).scalar_one_or_none()
            if existing is not None:
                return True
        return False


def backfill_sync(connection, *, user_id: str | None = None) -> BackfillSummary:
    """Idempotent legacy ingest executed by the migration on a sync engine.

    Mirrors :meth:`CompanyBackfillService.backfill_all` using portable
    Core/ORM statements so it can run from an Alembic migration.
    """
    from sqlalchemy import select, update
    from sqlalchemy.orm import Session

    session = Session(bind=connection)
    resolver = CompanyIdentityResolver()
    reports = (
        session.execute(select(AnalysisReport).order_by(AnalysisReport.created_at))
        .scalars()
        .all()
    )

    summary = BackfillSummary(reports_scanned=len(reports))

    for report in reports:
        request = report.request
        inputs = build_backfill_inputs(report, request)
        identity = resolver.resolve(inputs.startup_name, inputs.website)
        existing = _find_existing_sync(session, identity)

        if existing is not None:
            summary.companies_adopted += 1
            _adopt_existing_sync(existing, identity, inputs)
            company_id = existing.company_id
        else:
            owner = (
                request.user_id if request is not None and request.user_id else user_id
            )
            session.add(
                Company(
                    company_id=identity.company_id,
                    canonical_name=identity.canonical_name,
                    canonical_name_key=identity.canonical_name_key,
                    canonical_domain=identity.canonical_domain,
                    fallback_slug=identity.fallback_slug,
                    primary_name=identity.primary_name,
                    website=identity.website,
                    latest_decision=inputs.decision,
                    latest_confidence=inputs.confidence,
                    latest_composite_score=inputs.composite_score,
                    snapshot_count=0,
                    first_seen=inputs.created_at,
                    last_seen=inputs.created_at,
                    user_id=owner,
                )
            )
            session.flush()
            summary.companies_created += 1
            summary.created_company_ids.append(identity.company_id)
            company_id = identity.company_id

        snapshot_values = {
            "id": compute_snapshot_id(company_id, report.id),
            "company_id": company_id,
            "analysis_id": report.id,
            "report_id": report.id,
            "decision": inputs.decision,
            "confidence": inputs.confidence,
            "composite_score": inputs.composite_score,
            "readiness_score": inputs.readiness_score,
            "dimension_scores": inputs.dimension_scores,
            "created_at": inputs.created_at,
        }
        if _insert_snapshot_sync(session, snapshot_values):
            summary.snapshots_created += 1
            session.execute(
                update(Company)
                .where(Company.company_id == company_id)
                .values(
                    snapshot_count=Company.snapshot_count + 1,
                    last_seen=inputs.created_at,
                )
            )
        else:
            summary.snapshots_skipped += 1

    session.flush()
    return summary


def _find_existing_sync(session, identity: CompanyIdentity):
    """Locate an existing registry row in the store's priority order."""
    from sqlalchemy import select

    row = session.get(Company, identity.company_id)
    if row is not None:
        return row
    if identity.canonical_domain:
        row = (
            session.execute(
                select(Company).where(
                    Company.canonical_domain == identity.canonical_domain
                )
            )
            .scalar_one_or_none()
        )
        if row is not None:
            return row
    if identity.canonical_name_key:
        row = (
            session.execute(
                select(Company)
                .where(Company.canonical_name_key == identity.canonical_name_key)
                .order_by(Company.first_seen.asc())
            )
            .scalars()
            .first()
        )
        if row is not None:
            return row
    row = (
        session.execute(
            select(Company)
            .where(Company.fallback_slug == identity.fallback_slug)
            .order_by(Company.first_seen.asc())
        )
        .scalars()
        .first()
    )
    return row


def _adopt_existing_sync(existing, identity: CompanyIdentity, inputs) -> None:
    """Adopt an existing row: refresh identity material and latest values."""
    existing.canonical_name = identity.canonical_name
    existing.canonical_name_key = identity.canonical_name_key
    existing.fallback_slug = identity.fallback_slug
    existing.primary_name = identity.primary_name
    existing.website = identity.website
    if identity.canonical_domain:
        existing.canonical_domain = identity.canonical_domain
    existing.latest_decision = inputs.decision
    existing.latest_confidence = inputs.confidence
    existing.latest_composite_score = inputs.composite_score
    existing.last_seen = inputs.created_at


def _insert_snapshot_sync(session, values: dict[str, Any]) -> bool:
    """Append one snapshot with dialect-aware atomic conflict handling."""
    from sqlalchemy import select

    from app.models.company import CompanySnapshot

    dialect = session.get_bind().dialect.name

    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = pg_insert(CompanySnapshot).values(**values).on_conflict_do_nothing(
            index_elements=[CompanySnapshot.analysis_id]
        )
        return (session.execute(stmt).rowcount or 0) > 0

    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        sqlite_stmt = (
            sqlite_insert(CompanySnapshot).values(**values).on_conflict_do_nothing()
        )
        return (session.execute(sqlite_stmt).rowcount or 0) > 0

    existing = (
        session.execute(
            select(CompanySnapshot).where(
                CompanySnapshot.analysis_id == values["analysis_id"]
            )
        )
        .scalar_one_or_none()
    )
    if existing is not None:
        return False
    session.add(CompanySnapshot(**values))
    session.flush()
    return True


__all__ = [
    "BackfillSummary",
    "CompanyBackfillService",
    "backfill_sync",
    "build_backfill_inputs",
]
