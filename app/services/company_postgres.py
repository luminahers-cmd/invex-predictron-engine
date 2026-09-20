"""PostgresCompanyStore — the Phase 1 implementation of ``CompanyStore``.

This store maps the persistence-agnostic protocol value objects onto the
SQLAlchemy ORM rows defined in :mod:`app.models.company`. It is written with
portable SQLAlchemy so the same code runs against PostgreSQL in production
and SQLite in the test-suite integration tests.

Identity resolution happens *inside* the store in priority order when the
deterministic ``company_id`` does not yet exist:

1. canonical domain
2. canonical company name (maximal-compression key)
3. normalized fallback slug

An existing company found through any of these lookups is *adopted* — its
stable ``company_id`` is kept so analysis history stays on one row even when
a later analysis adds richer identity material (e.g. a website).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Select, desc, func, select

from app.models.company import Company, CompanySnapshot
from app.services.company_protocols import (
    CompanyIdentity,
    CompanyPage,
    CompanyRecord,
    CompanySnapshotPayload,
    CompanySnapshotRecord,
    SnapshotAppendResult,
    SnapshotPage,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_SNAPSHOT_ID_SEPARATOR = "\x00"


def compute_snapshot_id(company_id: str, analysis_id: str) -> str:
    """Deterministic snapshot ID derived from (company_id, analysis_id).

    Re-ingesting the same analysis for the same company always yields the
    same primary key, which is what makes snapshot appends idempotent.
    """
    import hashlib

    material = f"{company_id}{_SNAPSHOT_ID_SEPARATOR}{analysis_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class PostgresCompanyStore:
    """SQLAlchemy-backed company registry store.

    Instances are stateless; every method receives the ``AsyncSession``.
    This mirrors the explicit-session convention used throughout
    ``app/services/persistence.py`` and makes the store trivially testable.
    """

    async def upsert_company(
        self,
        session: AsyncSession,
        identity: CompanyIdentity,
        *,
        user_id: str | None = None,
        seen_at: datetime | None = None,
        latest_decision: str | None = None,
        latest_confidence: float | None = None,
        latest_composite_score: float | None = None,
    ) -> CompanyRecord:
        now = seen_at or datetime.now()
        existing = await self._find_existing(session, identity)

        if existing is not None:
            existing.canonical_name = identity.canonical_name
            existing.canonical_name_key = identity.canonical_name_key
            existing.fallback_slug = identity.fallback_slug
            existing.primary_name = identity.primary_name
            existing.website = identity.website
            # Backfill identity material so previously name-only companies
            # become discoverable by domain on later passes.
            if identity.canonical_domain:
                existing.canonical_domain = identity.canonical_domain
            existing.latest_decision = latest_decision
            existing.latest_confidence = latest_confidence
            existing.latest_composite_score = latest_composite_score
            existing.last_seen = now
            session.add(existing)
            await session.flush()
            return _company_record(existing)

        row = Company(
            company_id=identity.company_id,
            canonical_name=identity.canonical_name,
            canonical_name_key=identity.canonical_name_key,
            canonical_domain=identity.canonical_domain,
            fallback_slug=identity.fallback_slug,
            primary_name=identity.primary_name,
            website=identity.website,
            latest_decision=latest_decision,
            latest_confidence=latest_confidence,
            latest_composite_score=latest_composite_score,
            snapshot_count=0,
            first_seen=now,
            last_seen=now,
            user_id=user_id,
        )
        session.add(row)
        await session.flush()
        return _company_record(row)

    async def append_snapshot(
        self,
        session: AsyncSession,
        payload: CompanySnapshotPayload,
    ) -> SnapshotAppendResult:
        """Append one immutable snapshot, atomically.

        Idempotent by design: if ``payload.analysis_id`` is already stored,
        the existing row is returned with ``created=False`` and neither a new
        snapshot nor a snapshot-count increment happens.

        Phase 2 atomicity: on PostgreSQL and SQLite the insert is emitted as
        ``INSERT ... ON CONFLICT (analysis_id) DO NOTHING`` (SQLite emits
        ``INSERT OR IGNORE``), so a racing duplicate append is resolved by the
        database instead of a check-then-insert gap. The surviving row is
        always re-read from the store (read-after-write) before returning.
        """
        created_at = payload.created_at or datetime.now()
        dialect = session.get_bind().dialect.name
        values: dict[str, Any] = {
            "id": compute_snapshot_id(payload.company_id, payload.analysis_id),
            "company_id": payload.company_id,
            "analysis_id": payload.analysis_id,
            "report_id": payload.report_id,
            "decision": payload.decision,
            "confidence": payload.confidence,
            "composite_score": payload.composite_score,
            "readiness_score": payload.readiness_score,
            "dimension_scores": payload.dimension_scores or {},
            "created_at": created_at,
        }

        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            stmt = pg_insert(CompanySnapshot).values(**values).on_conflict_do_nothing(
                index_elements=[CompanySnapshot.analysis_id]
            )
            result = await session.execute(stmt)
            created = result.rowcount > 0
        elif dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as _sqlite_insert

            # INSERT OR IGNORE — any unique violation is treated as a no-op.
            sqlite_stmt = (
                _sqlite_insert(CompanySnapshot)
                .values(**values)
                .on_conflict_do_nothing()
            )
            result = await session.execute(sqlite_stmt)
            created = result.rowcount > 0
        else:
            # Unknown dialect: keep the portable check-then-insert fallback.
            existing = (
                await session.execute(
                    select(CompanySnapshot).where(
                        CompanySnapshot.analysis_id == payload.analysis_id
                    )
                )
            ).scalar_one_or_none()
            if existing is not None:
                return SnapshotAppendResult(
                    snapshot=_snapshot_record(existing), created=False
                )
            session.add(CompanySnapshot(**values))
            await session.flush()
            created = True

        # Read-after-write: fetch whichever row now exists (inserted or the
        # pre-existing duplicate) so the result is always consistent.
        existing = (
            await session.execute(
                select(CompanySnapshot).where(
                    CompanySnapshot.analysis_id == payload.analysis_id
                )
            )
        ).scalar_one()

        if created:
            company = (
                await session.execute(
                    select(Company).where(Company.company_id == payload.company_id)
                )
            ).scalar_one_or_none()
            if company is not None:
                company.snapshot_count = (company.snapshot_count or 0) + 1
                company.last_seen = created_at
                session.add(company)
                await session.flush()

        return SnapshotAppendResult(
            snapshot=_snapshot_record(existing), created=created
        )

    async def get_company(
        self,
        session: AsyncSession,
        company_id: str,
        *,
        user_id: str | None = None,
    ) -> CompanyRecord | None:
        stmt = select(Company).where(Company.company_id == company_id)
        stmt = _apply_ownership_scope(stmt, user_id)
        row = (await session.execute(stmt)).scalar_one_or_none()
        if row is None:
            return None
        return _company_record(row)

    async def list_companies(
        self,
        session: AsyncSession,
        *,
        user_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> CompanyPage:
        count_over = func.count().over().label("total_count")
        stmt = (
            select(Company, count_over)
            .order_by(desc(Company.last_seen), desc(Company.company_id))
            .offset(offset)
            .limit(limit)
        )
        stmt = _apply_ownership_scope(stmt, user_id)

        rows = (await session.execute(stmt)).all()
        total = rows[0].total_count if rows else 0
        records = [_company_record(row) for row, _ in rows]
        return CompanyPage(companies=records, total=total)

    async def list_snapshots(
        self,
        session: AsyncSession,
        company_id: str,
        *,
        user_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> SnapshotPage:
        count_over = func.count().over().label("total_count")
        stmt = (
            select(CompanySnapshot, count_over)
            .join(Company, Company.company_id == CompanySnapshot.company_id)
            .where(CompanySnapshot.company_id == company_id)
            .order_by(desc(CompanySnapshot.created_at), desc(CompanySnapshot.id))
            .offset(offset)
            .limit(limit)
        )
        stmt = _apply_ownership_scope(stmt, user_id)

        rows = (await session.execute(stmt)).all()
        total = rows[0].total_count if rows else 0
        snapshots = [_snapshot_record(row) for row, _ in rows]
        return SnapshotPage(snapshots=snapshots, total=total)

    # ---- Internal helpers ----

    async def _find_existing(self, session: AsyncSession, identity: CompanyIdentity):
        """Locate an existing row for this identity, in priority order.

        Returns ``None`` when no company matches any identity signal.
        """
        stmt = select(Company).where(Company.company_id == identity.company_id)
        row = (await session.execute(stmt)).scalar_one_or_none()
        if row is not None:
            return row

        if identity.canonical_domain:
            stmt = select(Company).where(
                Company.canonical_domain == identity.canonical_domain
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is not None:
                return row

        if identity.canonical_name_key:
            stmt = (
                select(Company)
                .where(Company.canonical_name_key == identity.canonical_name_key)
                .order_by(Company.first_seen.asc())
            )
            row = (await session.execute(stmt)).scalars().first()
            if row is not None:
                return row

        stmt = (
            select(Company)
            .where(Company.fallback_slug == identity.fallback_slug)
            .order_by(Company.first_seen.asc())
        )
        row = (await session.execute(stmt)).scalars().first()
        if row is not None:
            return row

        return None


def _apply_ownership_scope(stmt: Select[Any], user_id: str | None) -> Select[Any]:
    """Scope a statement to the anonymous (NULL) namespace or one user."""
    if user_id is None:
        return stmt.where(Company.user_id.is_(None))
    return stmt.where(Company.user_id == user_id)


def _company_record(row: Company) -> CompanyRecord:
    return CompanyRecord(
        company_id=row.company_id,
        canonical_name=row.canonical_name,
        canonical_domain=row.canonical_domain,
        primary_name=row.primary_name,
        website=row.website,
        latest_decision=row.latest_decision,
        latest_confidence=row.latest_confidence,
        latest_composite_score=row.latest_composite_score,
        snapshot_count=row.snapshot_count or 0,
        first_seen=row.first_seen,
        last_seen=row.last_seen,
        user_id=row.user_id,
        canonical_name_key=row.canonical_name_key,
        fallback_slug=row.fallback_slug,
    )


def _snapshot_record(row: CompanySnapshot) -> CompanySnapshotRecord:
    return CompanySnapshotRecord(
        id=row.id,
        company_id=row.company_id,
        analysis_id=row.analysis_id,
        report_id=row.report_id,
        decision=row.decision,
        confidence=row.confidence,
        composite_score=row.composite_score,
        readiness_score=row.readiness_score,
        dimension_scores=row.dimension_scores or {},
        created_at=row.created_at,
    )


__all__ = ["PostgresCompanyStore", "compute_snapshot_id"]
