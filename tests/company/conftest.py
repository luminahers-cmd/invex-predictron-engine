"""Shared fixtures for the CIH company test suite.

Store behavior is exercised against a real in-memory SQLite database
(portable SQLAlchemy); the ``sqlite_engine``/``sqlite_session`` fixtures live
in ``tests/conftest.py`` so the API tests can reuse them too.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.services.company_protocols import (
    CompanyPage,
    CompanyRecord,
    CompanySnapshotRecord,
    SnapshotAppendResult,
    SnapshotPage,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


class MemoryCompanyStore:
    """Minimal in-memory implementation of the CompanyStore protocol.

    Used by API tests as a deterministic stand-in for the SQL store. It does
    not re-implement identity adoption — the SQL store integration tests own
    that behavior.
    """

    def __init__(self) -> None:
        self._companies: dict[str, CompanyRecord] = {}
        self._snapshots: list[CompanySnapshotRecord] = []
        self._next_snapshot_seq = 0

    async def upsert_company(self, session, identity, **kwargs):
        record = CompanyRecord(
            company_id=identity.company_id,
            canonical_name=identity.canonical_name,
            canonical_domain=identity.canonical_domain,
            primary_name=identity.primary_name,
            website=identity.website,
            latest_decision=kwargs.get("latest_decision"),
            latest_confidence=kwargs.get("latest_confidence"),
            latest_composite_score=kwargs.get("latest_composite_score"),
            snapshot_count=0,
            first_seen=kwargs.get("seen_at") or utc_now(),
            last_seen=kwargs.get("seen_at") or utc_now(),
            user_id=kwargs.get("user_id"),
            canonical_name_key=identity.canonical_name_key,
            fallback_slug=identity.fallback_slug,
        )
        self._companies[identity.company_id] = record
        return record

    async def append_snapshot(self, session, payload):
        for snap in self._snapshots:
            if snap.analysis_id == payload.analysis_id:
                return SnapshotAppendResult(snapshot=snap, created=False)
        self._next_snapshot_seq += 1
        record = CompanySnapshotRecord(
            id=f"mem-snap-{self._next_snapshot_seq}",
            company_id=payload.company_id,
            analysis_id=payload.analysis_id,
            report_id=payload.report_id,
            decision=payload.decision,
            confidence=payload.confidence,
            composite_score=payload.composite_score,
            readiness_score=payload.readiness_score,
            dimension_scores=payload.dimension_scores,
            created_at=payload.created_at or utc_now(),
        )
        self._snapshots.append(record)
        return SnapshotAppendResult(snapshot=record, created=True)

    async def get_company(self, session, company_id, *, user_id=None):
        company = self._companies.get(company_id)
        if company is None:
            return None
        if user_id is None and company.user_id is not None:
            return None
        if user_id is not None and company.user_id != user_id:
            return None
        return company

    async def list_companies(self, session, *, user_id=None, offset=0, limit=20):
        rows = [
            c
            for c in self._companies.values()
            if (user_id is None and c.user_id is None)
            or (user_id is not None and c.user_id == user_id)
        ]
        rows.sort(key=lambda c: c.last_seen, reverse=True)
        total = len(rows)
        return CompanyPage(companies=rows[offset : offset + limit], total=total)

    async def list_snapshots(self, session, company_id, *, user_id=None, offset=0, limit=20):
        rows = [s for s in self._snapshots if s.company_id == company_id]
        rows.sort(key=lambda s: s.created_at, reverse=True)
        total = len(rows)
        return SnapshotPage(snapshots=rows[offset : offset + limit], total=total)


@pytest.fixture
def memory_store():
    return MemoryCompanyStore()
