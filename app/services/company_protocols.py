"""CompanyStore protocol and value objects for the Company Intelligence Hub.

The protocol is the stable contract between the ingest/read services and the
persistent store. It uses structural subtyping (``typing.Protocol``) exactly
like the engine's ``predictron_engine/interfaces/protocols.py`` so that any
store implementing the shape can be swapped in — no inheritance required.

Phase 1 ships exactly one implementation (:mod:`app.services.company_postgres`).
A future dataset-store adapter would implement the same protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class CompanyIdentity:
    """Resolved, deterministic identity of one company.

    ``company_id`` is a SHA-256 digest of the strongest available signal:
    canonical domain, canonical company name, or normalized fallback slug.
    """

    company_id: str
    canonical_name: str
    canonical_domain: str | None
    canonical_name_key: str
    fallback_slug: str
    primary_name: str
    website: str | None
    match_source: str


@dataclass
class CompanyRecord:
    """A stored company row as returned by the store (persistence-agnostic)."""

    company_id: str
    canonical_name: str
    canonical_domain: str | None
    primary_name: str
    website: str | None
    latest_decision: str | None
    latest_confidence: float | None
    latest_composite_score: float | None
    snapshot_count: int
    first_seen: datetime
    last_seen: datetime
    user_id: str | None
    canonical_name_key: str
    fallback_slug: str


@dataclass
class CompanySnapshotRecord:
    """One immutable, stored company snapshot."""

    id: str
    company_id: str
    analysis_id: str
    report_id: str
    decision: str | None
    confidence: float | None
    composite_score: float | None
    readiness_score: float | None
    dimension_scores: dict[str, float]
    created_at: datetime


@dataclass
class CompanySnapshotPayload:
    """Input payload for appending one snapshot."""

    company_id: str
    analysis_id: str
    report_id: str
    decision: str | None = None
    confidence: float | None = None
    composite_score: float | None = None
    readiness_score: float | None = None
    dimension_scores: dict[str, float] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass
class SnapshotAppendResult:
    """Result of trying to append a snapshot.

    ``created`` is ``False`` when the analysis ID was already ingested, in
    which case ``snapshot`` is the pre-existing row — duplicates never
    produce a second snapshot.
    """

    snapshot: CompanySnapshotRecord
    created: bool


@dataclass
class CompanyPage:
    """Paginated company listing."""

    companies: list[CompanyRecord]
    total: int


@dataclass
class SnapshotPage:
    """Paginated snapshot history."""

    snapshots: list[CompanySnapshotRecord]
    total: int


@runtime_checkable
class CompanyStore(Protocol):
    """Contract implemented by any company registry store.

    All methods take an explicit ``AsyncSession`` so the store stays
    transport-agnostic, mirroring the persistence service conventions in
    ``app/services/persistence.py``.
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
        """Insert or update one company, resolving identity by priority."""
        ...

    async def append_snapshot(
        self,
        session: AsyncSession,
        payload: CompanySnapshotPayload,
    ) -> SnapshotAppendResult:
        """Append one immutable snapshot, guarding against duplicates."""
        ...

    async def get_company(
        self,
        session: AsyncSession,
        company_id: str,
        *,
        user_id: str | None = None,
    ) -> CompanyRecord | None:
        """Fetch one company by ID (optionally ownership-scoped)."""
        ...

    async def list_companies(
        self,
        session: AsyncSession,
        *,
        user_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> CompanyPage:
        """List companies, newest first, with a total count."""
        ...

    async def list_snapshots(
        self,
        session: AsyncSession,
        company_id: str,
        *,
        user_id: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> SnapshotPage:
        """List a company's snapshots, newest first, with a total count."""
        ...
