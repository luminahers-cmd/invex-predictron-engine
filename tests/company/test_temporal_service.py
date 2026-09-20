"""Tests for the Phase 3 temporal services (history + delta services).

Covers chronological ordering, pagination, report linkage and enrichment,
benchmark derivation against a real offline dataset, delta sequencing, and
ownership gating. Store behavior is exercised through the in-memory protocol
double and the SQLite-backed real store.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.services.company_deltas import CompanyDeltaService
from app.services.company_history import CompanyHistoryService
from app.services.company_postgres import PostgresCompanyStore
from app.services.company_protocols import (
    CompanyRecord,
    CompanySnapshotPayload,
    CompanySnapshotRecord,
)
from predictron_engine.dataset.models import (
    CompanyProfile,
    DatasetRecord,
    DecisionLabel,
    PredictionSummary,
)
from predictron_engine.dataset.store import DatasetStore

pytestmark = pytest.mark.asyncio

REF = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)


def _seed_company(store, company_id: str, *, user_id: str | None = None):
    from app.services.companies import CompanyIdentityResolver

    resolved = CompanyIdentityResolver().resolve(company_id, None)
    return store._companies.setdefault(
        company_id,
        CompanyRecord(
            company_id=company_id,
            canonical_name=resolved.canonical_name,
            canonical_domain=resolved.canonical_domain,
            primary_name=resolved.primary_name,
            website=resolved.website,
            latest_decision=None,
            latest_confidence=None,
            latest_composite_score=None,
            snapshot_count=0,
            first_seen=REF,
            last_seen=REF,
            user_id=user_id,
            canonical_name_key=resolved.canonical_name_key,
            fallback_slug=resolved.fallback_slug,
        ),
    )


def _snapshot(
    *,
    snap_id: str = "s1",
    company_id: str = "comp-1",
    created_at: datetime = REF,
    decision: str | None = "invest",
    confidence: float | None = 0.8,
    composite: float | None = 70.0,
    readiness: float | None = 60.0,
    dimensions: dict[str, float] | None = None,
) -> CompanySnapshotRecord:
    return CompanySnapshotRecord(
        id=snap_id,
        company_id=company_id,
        analysis_id=f"analysis-{snap_id}",
        report_id=f"report-{snap_id}",
        decision=decision,
        confidence=confidence,
        composite_score=composite,
        readiness_score=readiness,
        dimension_scores=dimensions or {},
        created_at=created_at,
    )


def _push_snapshot(store, snapshot: CompanySnapshotRecord) -> None:
    store._snapshots.append(snapshot)
    company = store._companies.get(snapshot.company_id)
    if company is not None:
        company.last_seen = snapshot.created_at


def _seed_series(store, company_id: str = "comp-1", count: int = 3) -> list:
    """Seed ``count`` chronological snapshots ending at REF."""
    snapshots = [
        _snapshot(
            snap_id=f"s{i}",
            company_id=company_id,
            created_at=REF.replace(month=month),
            confidence=0.7 + (i * 0.05),
        )
        for i, month in enumerate(range(max(1, 12 - count + 1), 13))
    ]
    for snapshot in snapshots:
        _push_snapshot(store, snapshot)
    store._companies[company_id].snapshot_count = len(snapshots)
    return snapshots


async def _fake_resolver(reports: dict[str, dict]):
    async def _resolve(session, report_id: str):
        del session
        return reports.get(report_id)

    return _resolve


# ── history service: ordering, pagination, ownership ──────────────────


class TestHistoryServiceOrdering:
    async def test_timeline_ordering_newest_first(self, memory_store, sqlite_session):
        _seed_company(memory_store, "comp-1")
        snapshots = _seed_series(memory_store, count=3)
        service = CompanyHistoryService(store=memory_store)

        result = await service.timeline(sqlite_session, "comp-1")

        assert result is not None
        assert result.total == 3
        assert [view.snapshot.id for view in result.entries] == [
            snapshots[2].id,
            snapshots[1].id,
            snapshots[0].id,
        ]
        assert result.trend.snapshot_count == 3

    async def test_timeline_pagination(self, memory_store, sqlite_session):
        _seed_company(memory_store, "comp-1")
        _seed_series(memory_store, count=3)
        service = CompanyHistoryService(store=memory_store)

        result = await service.timeline(sqlite_session, "comp-1", offset=1, limit=1)

        assert result is not None
        assert result.total == 3
        assert len(result.entries) == 1
        assert result.entries[0].snapshot.id == "s1"

    async def test_timeline_missing_company_returns_none(self, memory_store, sqlite_session):
        service = CompanyHistoryService(store=memory_store)

        assert await service.timeline(sqlite_session, "nope") is None

    async def test_latest_returns_newest_with_trend(self, memory_store, sqlite_session):
        _seed_company(memory_store, "comp-1")
        _seed_series(memory_store, count=3)
        service = CompanyHistoryService(store=memory_store)

        result = await service.latest(sqlite_session, "comp-1")

        assert result is not None
        assert result.total == 3
        assert len(result.entries) == 1
        assert result.entries[0].snapshot.id == "s2"
        assert result.trend.snapshot_count == 3

    async def test_latest_empty_company(self, memory_store, sqlite_session):
        _seed_company(memory_store, "comp-1")
        service = CompanyHistoryService(store=memory_store)

        result = await service.latest(sqlite_session, "comp-1")

        assert result is not None
        assert result.total == 0
        assert result.entries == []
        assert result.trend.snapshot_count == 0

    async def test_ownership_gating(self, memory_store, sqlite_session):
        _seed_company(memory_store, "comp-1", user_id="u1")
        _seed_series(memory_store, count=2)
        service = CompanyHistoryService(store=memory_store)

        assert await service.timeline(sqlite_session, "comp-1", user_id="u2") is None
        assert await service.timeline(sqlite_session, "comp-1", user_id=None) is None

        result = await service.timeline(sqlite_session, "comp-1", user_id="u1")
        assert result is not None
        assert result.total == 2


# ── history service: report enrichment ────────────────────────────────


class TestHistoryServiceEnrichment:
    def _report_payload(self):
        return {
            "evidence": [
                {"statement": "e1", "source": "s1"},
                {"statement": "e2", "source": "s2"},
            ],
            "recommendations": [{"action": "Hire CTO"}, "Raise runway"],
            "investment_decision": {
                "category": "invest",
                "composite_score": 70.0,
                "rationale": {
                    "key_evidence_summary": "Two evidence items",
                    "confidence_explanation": "High confidence",
                    "primary_reasons_for": ["Traction is strong"],
                },
            },
            "investment_readiness": {
                "readiness_score": 60.0,
                "readiness_level": "moderate",
                "key_strengths": ["Strong market"],
                "key_concerns": ["Thin team"],
            },
        }

    async def test_enrichment_extracts_report_fields(
        self, memory_store, sqlite_session
    ):
        _seed_company(memory_store, "comp-1")
        snap = _snapshot(snap_id="s1")
        _push_snapshot(memory_store, snap)
        reports = {"report-s1": self._report_payload()}
        resolver = await _fake_resolver(reports)
        service = CompanyHistoryService(store=memory_store, report_resolver=resolver)

        result = await service.latest(sqlite_session, "comp-1")

        assert result is not None
        view = result.entries[0]
        assert view.evidence_count == 2
        assert "Two evidence items" in view.decision_explanation
        assert "Traction is strong" in view.decision_explanation
        assert {f.split("=", 1)[0] for f in view.key_facts} >= {
            "decision",
            "confidence",
            "readiness_level",
            "recommendation",
            "key_strength",
            "key_concern",
        }

    async def test_enrichment_missing_report_is_empty(
        self, memory_store, sqlite_session
    ):
        _seed_company(memory_store, "comp-1")
        _push_snapshot(memory_store, _snapshot(snap_id="s1"))
        service = CompanyHistoryService(store=memory_store)

        result = await service.latest(sqlite_session, "comp-1")

        assert result is not None
        view = result.entries[0]
        assert view.evidence_count is None
        assert view.decision_explanation == ()
        assert view.key_facts == (
            "composite_score=70",
            "confidence=0.8",
            "decision=invest",
            "readiness_score=60",
        )

    async def test_enrichment_missing_report_uses_snapshot_facts(
        self, memory_store, sqlite_session
    ):
        _seed_company(memory_store, "comp-1")
        _push_snapshot(
            memory_store,
            _snapshot(
                snap_id="s1",
                dimensions={"market_opportunity": 82.0, "founder_quality": 64.0},
            ),
        )
        service = CompanyHistoryService(store=memory_store)

        result = await service.latest(sqlite_session, "comp-1")

        assert result is not None
        facts = result.entries[0].key_facts
        assert "dimension.market_opportunity=82" in facts
        assert "dimension.founder_quality=64" in facts

    async def test_benchmark_enrichment_against_dataset(
        self, tmp_path, memory_store, sqlite_session
    ):
        records = [
            _grounded_record(record_id=f"r{i}", composite_score=score)
            for i, score in enumerate([10.0, 20.0, 30.0, 40.0, 50.0])
        ]
        dataset = _dataset_adapter(tmp_path, records)
        _seed_company(memory_store, "comp-1")
        _push_snapshot(memory_store, _snapshot(snap_id="s1", composite=30.0))
        service = CompanyHistoryService(store=memory_store, dataset=dataset)

        result = await service.latest(sqlite_session, "comp-1")

        assert result is not None
        benchmark = result.entries[0].benchmark
        assert benchmark is not None
        assert benchmark.percentile_rank == 50.0
        assert benchmark.sample_size == 5

    async def test_benchmark_missing_without_dataset(
        self, memory_store, sqlite_session
    ):
        _seed_company(memory_store, "comp-1")
        _push_snapshot(memory_store, _snapshot(snap_id="s1", composite=30.0))
        service = CompanyHistoryService(store=memory_store)

        result = await service.latest(sqlite_session, "comp-1")

        assert result is not None
        assert result.entries[0].benchmark is None

    async def test_benchmark_missing_for_none_composite(
        self, tmp_path, memory_store, sqlite_session
    ):
        dataset = _dataset_adapter(tmp_path, [])
        _seed_company(memory_store, "comp-1")
        _push_snapshot(memory_store, _snapshot(snap_id="s1", composite=None))
        service = CompanyHistoryService(store=memory_store, dataset=dataset)

        result = await service.latest(sqlite_session, "comp-1")

        assert result is not None
        assert result.entries[0].benchmark is None


# ── history service: real SQL store ───────────────────────────────────


class TestHistoryServiceSqlStore:
    async def test_sql_store_timeline_and_deltas(self, sqlite_session):
        store = PostgresCompanyStore()
        from app.services.companies import CompanyIdentityResolver

        identity = CompanyIdentityResolver().resolve(
            "Acme Inc", "https://acme.com"
        )
        company = await store.upsert_company(
            sqlite_session,
            identity,
            user_id="u1",
            latest_decision="invest",
            latest_confidence=0.7,
            latest_composite_score=60.0,
        )
        for i, confidence in enumerate([0.7, 0.8, 0.85]):
            await store.append_snapshot(
                sqlite_session,
                CompanySnapshotPayload(
                    company_id=company.company_id,
                    analysis_id=f"analysis-{i}",
                    report_id=f"report-{i}",
                    decision="invest",
                    confidence=confidence,
                    composite_score=60.0 + i * 5.0,
                    readiness_score=55.0 + i,
                    dimension_scores={},
                    created_at=REF.replace(month=1 + i),
                ),
            )
        async def _noop_resolver(session, report_id):
            del session, report_id
            return None

        service = CompanyHistoryService(store=store, report_resolver=_noop_resolver)

        result = await service.timeline(sqlite_session, company.company_id, user_id="u1")

        assert result is not None
        assert result.total == 3
        assert [view.snapshot.analysis_id for view in result.entries] == [
            "analysis-2",
            "analysis-1",
            "analysis-0",
        ]
        snapshot_ids = [view.snapshot.id for view in result.entries]

        delta_service = CompanyDeltaService(history=service)
        deltas = await delta_service.deltas(
            sqlite_session, company.company_id, user_id="u1"
        )

        assert deltas is not None
        assert deltas.total == 2
        assert deltas.deltas[0].current_snapshot_id == snapshot_ids[0]
        assert deltas.deltas[0].previous_snapshot_id == snapshot_ids[1]
        assert deltas.deltas[1].current_snapshot_id == snapshot_ids[1]
        assert deltas.deltas[1].previous_snapshot_id == snapshot_ids[2]


# ── delta service ─────────────────────────────────────────────────────


class TestDeltaService:
    async def test_deltas_newest_first_with_chained_ids(
        self, memory_store, sqlite_session
    ):
        _seed_company(memory_store, "comp-1")
        snapshots = _seed_series(memory_store, count=4)
        service = CompanyDeltaService(
            history=CompanyHistoryService(store=memory_store)
        )

        result = await service.deltas(sqlite_session, "comp-1")

        assert result is not None
        assert result.total == 3
        newest_first = list(reversed(snapshots))
        for position, delta in enumerate(result.deltas):
            assert delta.previous_snapshot_id == newest_first[position + 1].id
            assert delta.current_snapshot_id == newest_first[position].id

    async def test_deltas_confidence_increasing(self, memory_store, sqlite_session):
        _seed_company(memory_store, "comp-1")
        _seed_series(memory_store, count=3)
        service = CompanyDeltaService(
            history=CompanyHistoryService(store=memory_store)
        )

        result = await service.deltas(sqlite_session, "comp-1")

        assert result is not None
        field = next(
            f
            for delta in result.deltas
            for f in delta.fields
            if f.field == "confidence"
        )
        assert field.change == pytest.approx(0.05)
        assert result.trend.confidence.value == "increasing"

    async def test_deltas_pagination(self, memory_store, sqlite_session):
        _seed_company(memory_store, "comp-1")
        _seed_series(memory_store, count=5)
        service = CompanyDeltaService(
            history=CompanyHistoryService(store=memory_store)
        )

        result = await service.deltas(sqlite_session, "comp-1", offset=1, limit=2)

        assert result is not None
        assert result.total == 4
        assert len(result.deltas) == 2

    async def test_deltas_missing_company_none(self, memory_store, sqlite_session):
        service = CompanyDeltaService(
            history=CompanyHistoryService(store=memory_store)
        )

        assert await service.deltas(sqlite_session, "nope") is None

    async def test_deltas_ownership_gating(self, memory_store, sqlite_session):
        _seed_company(memory_store, "comp-1", user_id="u1")
        _seed_series(memory_store, count=3)
        service = CompanyDeltaService(
            history=CompanyHistoryService(store=memory_store)
        )

        assert await service.deltas(sqlite_session, "comp-1", user_id="u2") is None


# ── shared helpers ────────────────────────────────────────────────────


def _grounded_record(
    *,
    record_id: str,
    composite_score: float,
) -> DatasetRecord:
    """A dataset record passing every CIH placeholder quality gate."""
    return DatasetRecord(
        startup_name="Acme Inc",
        website="https://acme.com",
        analysis_date=REF,
        engine_version="0.12.1",
        evidence_bundle_reference="bundle://acme",
        prediction=PredictionSummary(
            decision=DecisionLabel.INVEST,
            confidence=0.8,
            composite_score=composite_score,
            dimension_scores={"market": 70.0},
            recommendation_count=2,
        ),
        profile=CompanyProfile(domain="acme.com", country_code="US"),
        record_id=record_id,
    )


def _dataset_adapter(tmp_path, records):
    from app.services.company_dataset import DatasetCompanyStore

    store = DatasetStore(tmp_path / "dataset")
    store.initialize()
    for record in records:
        store.save_record(record)
    return DatasetCompanyStore(store, as_of=REF)
