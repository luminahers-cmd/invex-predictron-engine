"""API tests for the Continuous Learning Intelligence (Phase 7).

Mirrors the forecasts suite: real routers against in-memory SQLite with
``get_db`` overridden.  The evaluated ledger is seeded through the real
stores/ORM so the learning endpoints exercise their exact query paths.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.db.session import get_db
from app.main import app
from tests.learning.seed import (
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


@pytest.fixture
def override_get_db(sqlite_session):
    """Point every router's ``get_db`` dependency at the test session."""

    async def _override():
        yield sqlite_session

    app.dependency_overrides[get_db] = _override
    yield sqlite_session
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def override_async_session(sqlite_engine, monkeypatch):
    """Bind the report endpoints' ``AsyncSessionLocal`` to the test engine.

    The ``/learning/reports`` routes record/read through
    ``app.db.session.AsyncSessionLocal`` (not ``get_db``), so this fixture
    swaps that factory for one bound to the isolated in-memory engine.
    """
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    import app.db.session as db_session

    factory = async_sessionmaker(
        sqlite_engine, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(db_session, "AsyncSessionLocal", factory)
    return factory


async def _seed_evaluated(session, *, name: str, verdict: str = "correct"):
    company_id = await seed_company(session, name=name)
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


def _headers(user_id: str) -> dict[str, str]:
    from app.auth.jwt import create_access_token

    return {"Authorization": f"Bearer {create_access_token(subject=user_id)}"}


class TestLearningSummaryApi:
    @pytest.mark.asyncio
    async def test_summary_empty_db(self, client, override_get_db) -> None:
        resp = await client.get("/api/v1/learning/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["counts"] == {"evaluations": 0, "samples": 0, "scoreable": 0}
        assert body["scope"] == "anonymous"
        assert body["digest"]["accuracy"] is None
        assert body["observations"] == []
        assert body["recommendations"] == []

    @pytest.mark.asyncio
    async def test_summary_over_seeded_ledger(self, client, override_get_db) -> None:
        await _seed_evaluated(override_get_db, name="cmp-1")

        resp = await client.get("/api/v1/learning/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["counts"]["evaluations"] == 1
        assert body["digest"]["accuracy"] == 1.0
        assert body["confidence"]["mean"] == 0.9
        assert body["distributions"]["sector"] == {"ai": 1}

    @pytest.mark.asyncio
    async def test_summary_namespace_scoping(self, client, override_get_db) -> None:
        company = await seed_company(override_get_db, name="cmp-u1", user_id="u1")
        snapshot = await seed_analysis(
            override_get_db, company_id=company, features=_FEATURES
        )
        await seed_evaluation(
            override_get_db,
            company_id=company,
            snapshot_id=snapshot,
            confidence=0.9,
            verdict="correct",
        )

        anon = await client.get("/api/v1/learning/summary")
        scoped = await client.get("/api/v1/learning/summary", headers=_headers("u1"))
        assert anon.json()["counts"]["evaluations"] == 0
        assert scoped.json()["counts"]["evaluations"] == 1
        assert scoped.json()["scope"] == "user:u1"


class TestLearningViewsApi:
    @pytest.mark.asyncio
    async def test_patterns(self, client, override_get_db) -> None:
        await _seed_evaluated(override_get_db, name="cmp-1")

        resp = await client.get("/api/v1/learning/patterns")
        assert resp.status_code == 200
        body = resp.json()
        assert {p["dimension"] for p in body["patterns"]} == {
            "sector",
            "stage",
            "country",
            "technology",
            "business_model",
            "founder",
        }

    @pytest.mark.asyncio
    async def test_observations(self, client, override_get_db) -> None:
        await _seed_evaluated(override_get_db, name="cmp-1")

        resp = await client.get("/api/v1/learning/observations")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["observations"]) == 6
        assert all(o["metric"] == "sample_size" for o in body["observations"])

    @pytest.mark.asyncio
    async def test_confidence(self, client, override_get_db) -> None:
        await _seed_evaluated(override_get_db, name="cmp-1")

        resp = await client.get("/api/v1/learning/confidence")
        assert resp.status_code == 200
        body = resp.json()
        assert body["confidence"]["count"] == 1
        assert body["calibration"]["overconfidence_detected"] is False

    @pytest.mark.asyncio
    async def test_bias(self, client, override_get_db) -> None:
        await _seed_evaluated(override_get_db, name="cmp-1")

        resp = await client.get("/api/v1/learning/bias")
        assert resp.status_code == 200
        body = resp.json()
        assert body["metrics"]["accuracy"] == 1.0
        assert "sector" in body["knowledge"]

    @pytest.mark.asyncio
    async def test_recommendations(self, client, override_get_db) -> None:
        await _seed_evaluated(override_get_db, name="cmp-1")

        resp = await client.get("/api/v1/learning/recommendations")
        assert resp.status_code == 200
        body = resp.json()
        # accuracy 1.0 >= 0.6 -> maintain_confidence_weighting
        assert {r["kind"] for r in body["recommendations"]} == {
            "maintain_confidence_weighting"
        }


class TestLearningKnowledgeApi:
    @pytest.mark.asyncio
    async def test_knowledge_by_dimension(self, client, override_get_db) -> None:
        await _seed_evaluated(override_get_db, name="cmp-1")

        resp = await client.get("/api/v1/learning/knowledge/sector")
        assert resp.status_code == 200
        body = resp.json()
        assert body["dimension"] == "sector"
        assert body["entries"][0]["value"] == "ai"
        assert body["entries"][0]["sample_size"] == 1

    @pytest.mark.asyncio
    async def test_knowledge_shortcuts(self, client, override_get_db) -> None:
        await _seed_evaluated(override_get_db, name="cmp-1")

        for path in (
            "/api/v1/learning/sector",
            "/api/v1/learning/technology",
            "/api/v1/learning/country",
            "/api/v1/learning/stage",
            "/api/v1/learning/founders",
            "/api/v1/learning/business-model",
        ):
            resp = await client.get(path)
            assert resp.status_code == 200, path
            assert body_is_knowledge(resp.json())

    @pytest.mark.asyncio
    async def test_knowledge_invalid_dimension(self, client, override_get_db) -> None:
        resp = await client.get("/api/v1/learning/knowledge/not-a-dim")
        assert resp.status_code == 422
        assert "dimension" in resp.json()["detail"]


class TestLearningReportsApi:
    @pytest.mark.asyncio
    async def test_reports_empty_and_404(self, client, override_async_session) -> None:
        resp = await client.get("/api/v1/learning/reports")
        assert resp.status_code == 200
        assert resp.json()["reports"] == []

        missing = await client.get("/api/v1/learning/reports/gone")
        assert missing.status_code == 404
        assert "no learning report recorded" in missing.json()["detail"]

    @pytest.mark.asyncio
    async def test_reports_after_snapshot_cli(
        self, client, override_async_session, override_get_db, sqlite_session
    ) -> None:
        from app.services.learning import LearningService
        from predictron_engine.learning.models import LearningPeriodKind

        await _seed_evaluated(override_get_db, name="cmp-1")
        service = LearningService()
        snapshot = await service.snapshot(
            sqlite_session,
            as_of=datetime(2026, 9, 21, 12, 0, tzinfo=UTC),
            period_kind=LearningPeriodKind.DAILY,
        )
        # report endpoints read through AsyncSessionLocal (not get_db), so the
        # recorded rows must be committed before they become visible there.
        await sqlite_session.commit()

        resp = await client.get("/api/v1/learning/reports")
        assert resp.status_code == 200
        reports = resp.json()["reports"]
        assert len(reports) == 1
        assert reports[0]["snapshot_id"] == snapshot.snapshot_id

        detail = await client.get(
            f"/api/v1/learning/reports/{snapshot.snapshot_id}"
        )
        assert detail.status_code == 200
        body = detail.json()
        assert body["content_hash"] == snapshot.content_hash
        assert body["payload"]["counts"]["evaluations"] == 1

    @pytest.mark.asyncio
    async def test_reports_invalid_period(self, client, override_async_session) -> None:
        resp = await client.get("/api/v1/learning/reports", params={"period": "hourly"})
        assert resp.status_code == 422
        assert "period must be one of" in resp.json()["detail"]


def body_is_knowledge(body: dict) -> bool:
    return body.get("dimension") in {
        "sector",
        "stage",
        "country",
        "technology",
        "business_model",
        "founder",
    } and isinstance(body.get("entries"), list)
