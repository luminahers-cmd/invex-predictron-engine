"""Service tests for the Phase 7 learning service.

Covers the operator-facing ``snapshot`` recording (idempotent, append-only,
repository-wide) plus the namespace scoping of the live projections, which
the HTTP suite does not exercise directly.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.models.learning import (
    LearningObservationRecord,
    LearningPatternRecord,
    LearningReportRecord,
    LearningSnapshotRecord,
)
from app.services.learning import LearningService
from predictron_engine.learning.models import LearningPeriodKind
from tests.learning.seed import (
    count_rows,
    seed_analysis,
    seed_company,
    seed_evaluation,
)

_FEATURES = {
    "industry": "AI",
    "funding_stage": "Seed",
    "headquarters_region": "US",
    "primary_technology_domain": "Machine Learning",
    "business_model": "SaaS",
    "founder_team_type": "team",
}


async def _seed_evaluated(
    session, *, name: str, user_id: str | None = None, verdict: str = "correct"
) -> tuple[str, str]:
    company_id = await seed_company(session, name=name, user_id=user_id)
    snapshot_id = await seed_analysis(
        session, company_id=company_id, features=_FEATURES
    )
    await seed_evaluation(
        session,
        company_id=company_id,
        snapshot_id=snapshot_id,
        confidence=0.9,
        verdict=verdict,
    )
    return company_id, snapshot_id


@pytest.mark.asyncio
async def test_live_summary_scopes_to_user(sqlite_session) -> None:
    await _seed_evaluated(sqlite_session, name="cmp-u1", user_id="u1", verdict="correct")
    await _seed_evaluated(
        sqlite_session, name="cmp-u2", user_id="u2", verdict="incorrect"
    )
    await _seed_evaluated(sqlite_session, name="cmp-anon", verdict="correct")

    service = LearningService()
    scoped = await service.summary(sqlite_session, user_id="u1")
    other = await service.summary(sqlite_session, user_id="u2")
    anon = await service.summary(sqlite_session, user_id=None)

    assert scoped.counts["evaluations"] == 1
    assert scoped.scope == "user:u1"
    assert other.counts["evaluations"] == 1
    assert anon.counts["evaluations"] == 1
    assert anon.scope == "anonymous"
    assert scoped.digest.accuracy == 1.0
    assert other.digest.accuracy == 0.0


@pytest.mark.asyncio
async def test_repository_snapshot_ignores_namespace(sqlite_session) -> None:
    await _seed_evaluated(sqlite_session, name="cmp-u1", user_id="u1")
    await _seed_evaluated(sqlite_session, name="cmp-anon")

    service = LearningService()
    repo = await service.snapshot(sqlite_session, as_of=datetime.now(UTC))
    assert repo.scope == "repository"
    assert repo.counts["evaluations"] == 2


@pytest.mark.asyncio
async def test_snapshot_recording_is_idempotent_and_append_only(sqlite_session) -> None:
    await _seed_evaluated(sqlite_session, name="cmp-1")

    service = LearningService()
    as_of = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
    first = await service.snapshot(sqlite_session, as_of=as_of)
    second = await service.snapshot(sqlite_session, as_of=as_of)

    assert first.snapshot_id == second.snapshot_id
    assert first.verify() and second.verify()
    assert await count_rows(sqlite_session, LearningSnapshotRecord) == 1
    assert (
        await count_rows(sqlite_session, LearningObservationRecord)
        == len(first.observations)
    )
    assert (
        await count_rows(sqlite_session, LearningPatternRecord)
        == len(first.patterns)
    )
    assert await count_rows(sqlite_session, LearningReportRecord) == 1


@pytest.mark.asyncio
async def test_snapshot_records_distinct_periods(sqlite_session) -> None:
    await _seed_evaluated(sqlite_session, name="cmp-1")

    service = LearningService()
    day = await service.snapshot(
        sqlite_session,
        as_of=datetime(2026, 9, 21, tzinfo=UTC),
        period_kind=LearningPeriodKind.DAILY,
    )
    week = await service.snapshot(
        sqlite_session,
        as_of=datetime(2026, 9, 21, tzinfo=UTC),
        period_kind=LearningPeriodKind.WEEKLY,
    )
    assert day.snapshot_id != week.snapshot_id
    assert await count_rows(sqlite_session, LearningSnapshotRecord) == 2


@pytest.mark.asyncio
async def test_reports_list_and_detail(sqlite_session) -> None:
    await _seed_evaluated(sqlite_session, name="cmp-1")

    service = LearningService()
    as_of = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
    recorded = await service.snapshot(sqlite_session, as_of=as_of)

    listing = await service.reports(
        sqlite_session, period_kind=LearningPeriodKind.DAILY
    )
    assert listing.period_kind == LearningPeriodKind.DAILY
    assert len(listing.reports) == 1
    header = listing.reports[0]
    assert header["snapshot_id"] == recorded.snapshot_id
    assert header["content_hash"] == recorded.content_hash
    assert header["anchor_date"] == "2026-09-21"

    detail = await service.report_detail(
        sqlite_session, snapshot_id=recorded.snapshot_id
    )
    assert detail is not None
    assert detail.snapshot_id == recorded.snapshot_id
    assert detail.content_hash == recorded.content_hash
    assert detail.verify()

    missing = await service.report_detail(sqlite_session, snapshot_id="nope")
    assert missing is None


@pytest.mark.asyncio
async def test_patterns_and_observations_views(sqlite_session) -> None:
    await _seed_evaluated(sqlite_session, name="cmp-1")

    service = LearningService()
    patterns = await service.patterns(sqlite_session, user_id=None)
    observations = await service.observations(sqlite_session, user_id=None)

    assert len(patterns.patterns) == 6
    assert {p.dimension.value for p in patterns.patterns} == {
        "sector",
        "stage",
        "country",
        "technology",
        "business_model",
        "founder",
    }
    # one evaluation -> low-support observations per dimension
    assert len(observations.observations) == 6
    assert all(o.metric == "sample_size" for o in observations.observations)


@pytest.mark.asyncio
async def test_dimension_intelligence_resolves_and_errors(sqlite_session) -> None:
    await _seed_evaluated(sqlite_session, name="cmp-1")

    service = LearningService()
    knowledge = await service.dimension_intelligence(
        sqlite_session, "sector", user_id=None
    )
    assert knowledge.dimension == "sector"
    assert knowledge.entries[0].value == "ai"
    assert knowledge.entries[0].sample_size == 1

    with pytest.raises(ValueError, match="unknown learning dimension"):
        await service.dimension_intelligence(sqlite_session, "not-a-dim")


@pytest.mark.asyncio
async def test_bias_and_confidence_shares_metric_surface(sqlite_session) -> None:
    await _seed_evaluated(sqlite_session, name="cmp-1")

    service = LearningService()
    bias = await service.bias(sqlite_session, user_id=None)
    confidence = await service.confidence(sqlite_session, user_id=None)

    assert bias.metrics["accuracy"] == 1.0
    assert set(bias.knowledge) == {
        "sector",
        "stage",
        "country",
        "technology",
        "business_model",
        "founder",
    }
    assert confidence.confidence.count == 1
    assert confidence.confidence.mean == 0.9
