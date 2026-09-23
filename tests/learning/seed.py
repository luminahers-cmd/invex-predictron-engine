"""Deterministic seeding helpers for the learning service/API test suites.

Builds the evaluated prediction ledger (company → analysis report with
``features`` → immutable snapshot → outcome → append-only evaluation)
through the real stores/services so the learning projections exercise their
exact query paths.  No randomness anywhere: every id is derived from the
seeded content.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

from app.models.analysis import AnalysisReport, AnalysisRequest
from app.models.company_outcome import CompanyEvaluation, CompanyOutcome


def utc_now() -> datetime:
    return datetime.now(UTC)


def _digest(*parts: object) -> str:
    material = "\x00".join(str(part) for part in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


async def seed_company(session, *, name: str, user_id: str | None = None) -> str:
    """Insert one company through the real store; returns its company id."""
    from app.services.companies import CompanyIdentityResolver
    from app.services.company_postgres import PostgresCompanyStore

    identity = CompanyIdentityResolver().resolve(name, None)
    record = await PostgresCompanyStore().upsert_company(
        session, identity, user_id=user_id
    )
    return record.company_id


async def seed_analysis(
    session,
    *,
    company_id: str,
    features: dict[str, object] | None = None,
    snapshot_created_at: datetime | None = None,
) -> str:
    """Seed analysis request/report + immutable snapshot; returns snapshot id."""
    from app.services.company_postgres import PostgresCompanyStore
    from app.services.company_protocols import CompanySnapshotPayload

    analysis_id = _digest("analysis", company_id, str(features or {}))
    report_id = _digest("report", analysis_id)
    request_id = _digest("request", analysis_id)
    session.add(
        AnalysisRequest(
            id=request_id,
            user_id=None,
            startup_name=company_id,
            website=f"https://{company_id}.example",
            description="seeded",
            founder_linkedin_urls=[],
        )
    )
    session.add(
        AnalysisReport(
            id=analysis_id,
            request_id=request_id,
            startup_name=company_id,
            venture_score=70.0,
            market_score=70.0,
            founder_score=70.0,
            traction_score=70.0,
            recommendations=[],
            confidence=0.8,
            engine_version="7.0.0",
            full_report={"features": features or {}},
        )
    )
    created_at = snapshot_created_at or (utc_now() - timedelta(days=30))
    payload = CompanySnapshotPayload(
        company_id=company_id,
        analysis_id=analysis_id,
        report_id=report_id,
        decision="invest",
        confidence=0.8,
        composite_score=70.0,
        readiness_score=60.0,
        dimension_scores={"market": 80.0},
        created_at=created_at,
    )
    append = await PostgresCompanyStore().append_snapshot(session, payload)
    return append.snapshot.id


async def seed_evaluation(
    session,
    *,
    company_id: str,
    snapshot_id: str,
    confidence: float,
    verdict: str,
    prediction: dict[str, object] | None = None,
    occurred_at: datetime | None = None,
) -> str:
    """Seed an outcome + append-only evaluation row; returns the evaluation id.

    ``verdict`` must be ``"correct"`` or ``"incorrect"`` (the two binary
    scoreable verdicts the learning engine consumes).
    """
    outcome_id = _digest("outcome", company_id, snapshot_id, verdict)
    session.add(
        CompanyOutcome(
            id=outcome_id,
            company_id=company_id,
            snapshot_id=snapshot_id,
            source="manual",
            occurred_at=occurred_at or (utc_now() - timedelta(days=10)),
            outcome_data={"status": "active"},
            verdict="active",
        )
    )
    await session.flush()
    evaluation_id = _digest("evaluation", company_id, snapshot_id, verdict)
    session.add(
        CompanyEvaluation(
            id=evaluation_id,
            company_id=company_id,
            snapshot_id=snapshot_id,
            outcome_id=outcome_id,
            prediction=prediction
            or {
                "confidence": confidence,
                "dimension_scores": {"market": 80.0},
            },
            verdict=verdict,
            alignment="strong_match",
            outcome_verdict="active",
            decision_match=verdict == "correct",
            snapshot_confidence=confidence,
            snapshot_composite_score=70.0,
            snapshot_created_at=utc_now() - timedelta(days=30),
            created_at=utc_now() - timedelta(days=5),
        )
    )
    await session.flush()
    return evaluation_id


async def seed_evaluated_company(
    session,
    *,
    name: str,
    user_id: str | None = None,
    features: dict[str, object] | None = None,
    confidence: float = 0.8,
    verdict: str = "correct",
) -> tuple[str, str]:
    """Full chain: company → report → snapshot → outcome → evaluation."""
    company_id = await seed_company(session, name=name, user_id=user_id)
    snapshot_id = await seed_analysis(
        session, company_id=company_id, features=features
    )
    await seed_evaluation(
        session,
        company_id=company_id,
        snapshot_id=snapshot_id,
        confidence=confidence,
        verdict=verdict,
    )
    return company_id, snapshot_id


async def count_rows(session, model) -> int:
    from sqlalchemy import func, select

    stmt = select(func.count()).select_from(model)
    result = await session.execute(stmt)
    return result.scalar_one()


__all__ = [
    "count_rows",
    "seed_analysis",
    "seed_company",
    "seed_evaluation",
    "seed_evaluated_company",
    "utc_now",
]
