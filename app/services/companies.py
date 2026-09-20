"""Company Intelligence Hub services — identity resolution and ingest.

This module owns:

* ``CompanyIdentityResolver`` — a lightweight, fully deterministic resolver
  that turns a (name, website) observation into one canonical identity.
  Priority: canonical domain > canonical company name > normalized fallback
  slug. No fuzzy matching, no ML — it deliberately reuses the deterministic
  normalization utilities from Project E2 (``company_name`` / ``enrichment``)
  instead of duplicating Project E2's entity-resolution pipeline.

* ``CompanyIngestService`` — the post-analysis hook that, after an analysis
  is persisted, resolves identity, upserts the company, and appends one
  immutable snapshot. Ingest is idempotent: duplicate analysis IDs never
  produce duplicate snapshots.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.services.company_postgres import PostgresCompanyStore
from app.services.company_protocols import (
    CompanyIdentity,
    CompanySnapshotPayload,
    CompanyStore,
)
from predictron_engine.dataset.company_name import canonical_name_key as _name_key
from predictron_engine.dataset.company_name import core_name as _core_name
from predictron_engine.dataset.company_name import fold_name as _fold_name
from predictron_engine.dataset.enrichment import extract_domain as _extract_domain

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.models.analysis import AnalysisRequest
    from app.schemas.analysis import (
        StartupAnalysisRequest,
        StartupAnalysisResponse,
    )
    from predictron_engine.models.report import Report

logger = logging.getLogger(__name__)

_SLUG_RE: re.Pattern[str] = re.compile(r"[^a-z0-9]+")


def utc_now() -> datetime:
    """Current UTC timestamp (naive UTC-agnostic default for this module)."""
    return datetime.now(UTC)


def compute_company_id(material: str) -> str:
    """Deterministic company ID: SHA-256 digest of the identity material."""
    if not material:
        material = "unknown"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def normalize_fallback_slug(name: str) -> str:
    """Normalize a display name into a safe deterministic fallback slug.

    Falls back to ``"unknown"`` when nothing usable remains.
    """
    folded = _fold_name(name)
    slug = _SLUG_RE.sub("-", folded).strip("-")
    return slug[:128] or "unknown"


@dataclass(frozen=True)
class CompanyIdentityResolver:
    """Resolves (name, website) observations into deterministic identities.

    The resolver itself holds no database state. Adopting an already-stored
    company (found through a lower-priority signal) is a store concern, not a
    resolver concern — see ``PostgresCompanyStore.upsert_company``.
    """

    def resolve(self, primary_name: str, website: str | None) -> CompanyIdentity:
        """Resolve one observation into a canonical, deterministic identity."""
        name = (primary_name or "").strip()
        canonical_name = _core_name(name) if name else ""
        canonical_name_key = _name_key(name) if name else ""
        canonical_domain = _extract_domain(website) or None
        fallback_slug = normalize_fallback_slug(name)

        if canonical_domain:
            material = canonical_domain
            match_source = "canonical_domain"
        elif canonical_name:
            material = canonical_name
            match_source = "canonical_name"
        else:
            material = fallback_slug
            match_source = "fallback_slug"

        return CompanyIdentity(
            company_id=compute_company_id(material),
            canonical_name=canonical_name,
            canonical_domain=canonical_domain,
            canonical_name_key=canonical_name_key,
            fallback_slug=fallback_slug,
            primary_name=name,
            website=website or None,
            match_source=match_source,
        )


@dataclass
class AnalysisSnapshotInputs:
    """Snapshot fields extracted from a completed analysis (pure mapping)."""

    startup_name: str
    website: str | None
    decision: str | None
    confidence: float | None
    composite_score: float | None
    readiness_score: float | None
    dimension_scores: dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


def build_snapshot_inputs(
    request: StartupAnalysisRequest,
    report: Report,
    response: StartupAnalysisResponse,
) -> AnalysisSnapshotInputs:
    """Extract CIH snapshot fields from a completed analysis.

    Pure function — safe to test with the engine's real ``Report`` models.
    """
    decision = (
        report.investment_decision.category.value
        if report.investment_decision is not None
        else None
    )
    composite_score = (
        report.investment_decision.composite_score
        if report.investment_decision is not None
        else report.overall_score
    )
    readiness_score = (
        report.investment_readiness.readiness_score
        if report.investment_readiness is not None
        else None
    )
    dimension_scores = {score.dimension: score.score for score in report.scores}

    website = str(request.website) if request.website else None
    if not website:
        website = report.startup.website or None

    created_at = report.analysis_metadata.timestamp or utc_now()

    return AnalysisSnapshotInputs(
        startup_name=request.startup_name,
        website=website,
        decision=decision,
        confidence=report.overall_confidence,
        composite_score=composite_score,
        readiness_score=readiness_score,
        dimension_scores=dimension_scores,
        created_at=created_at,
    )


@dataclass(frozen=True)
class IngestResult:
    """Outcome of one ingest operation."""

    company_id: str
    snapshot_id: str
    created_snapshot: bool


class CompanyIngestService:
    """Orchestrates resolve -> upsert company -> append snapshot.

    Used as the single hook after successful analysis persistence. It is
    deliberately additive: it never re-runs the engine, never touches the
    feature store / graph / signals, and never recomputes scores.
    """

    def __init__(
        self,
        store: CompanyStore | None = None,
        resolver: CompanyIdentityResolver | None = None,
    ) -> None:
        self._store = store or PostgresCompanyStore()
        self._resolver = resolver or CompanyIdentityResolver()

    async def ingest_analysis(
        self,
        session: AsyncSession,
        *,
        startup_name: str,
        website: str | None = None,
        analysis_id: str,
        report_id: str,
        decision: str | None = None,
        confidence: float | None = None,
        composite_score: float | None = None,
        readiness_score: float | None = None,
        dimension_scores: dict[str, float] | None = None,
        user_id: str | None = None,
        created_at: datetime | None = None,
    ) -> IngestResult:
        """Resolve identity, upsert the company, append one snapshot."""
        identity = self._resolver.resolve(startup_name, website)
        seen_at = created_at or utc_now()

        company = await self._store.upsert_company(
            session,
            identity,
            user_id=user_id,
            seen_at=seen_at,
            latest_decision=decision,
            latest_confidence=confidence,
            latest_composite_score=composite_score,
        )
        payload = CompanySnapshotPayload(
            company_id=company.company_id,
            analysis_id=analysis_id,
            report_id=report_id,
            decision=decision,
            confidence=confidence,
            composite_score=composite_score,
            readiness_score=readiness_score,
            dimension_scores=dimension_scores or {},
            created_at=seen_at,
        )
        append = await self._store.append_snapshot(session, payload)
        return IngestResult(
            company_id=company.company_id,
            snapshot_id=append.snapshot.id,
            created_snapshot=append.created,
        )

    async def ingest_after_persist(
        self,
        session: AsyncSession,
        *,
        analysis: AnalysisRequest,
        inputs: AnalysisSnapshotInputs,
        user_id: str | None = None,
    ) -> IngestResult | None:
        """Feed a just-persisted analysis into the company registry.

        Resolves the analysis' report ID from the persisted rows, then
        performs the ingest. Returns ``None`` when the report row cannot be
        located (should not happen after a successful persist).
        """
        report_id = await self._find_report_id(session, analysis.id)
        if report_id is None:
            logger.warning(
                "Company ingest skipped: no report row for analysis %s", analysis.id
            )
            return None
        return await self.ingest_analysis(
            session,
            startup_name=inputs.startup_name,
            website=inputs.website,
            analysis_id=analysis.id,
            report_id=report_id,
            decision=inputs.decision,
            confidence=inputs.confidence,
            composite_score=inputs.composite_score,
            readiness_score=inputs.readiness_score,
            dimension_scores=inputs.dimension_scores,
            user_id=user_id,
            created_at=inputs.created_at,
        )

    @staticmethod
    async def _find_report_id(session: AsyncSession, analysis_id: str) -> str | None:
        from sqlalchemy import select

        from app.models.analysis import AnalysisReport

        stmt = select(AnalysisReport.id).where(
            AnalysisReport.request_id == analysis_id
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


__all__ = [
    "AnalysisSnapshotInputs",
    "CompanyIdentityResolver",
    "CompanyIngestService",
    "IngestResult",
    "build_snapshot_inputs",
    "compute_company_id",
    "normalize_fallback_slug",
    "utc_now",
]
