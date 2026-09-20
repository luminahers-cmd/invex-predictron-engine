"""Tests for the Phase 2 profile read layer (DatasetCompanyStore + read service).

Exercises the read-only joins introduced for profile linkage: benchmark
percentiles and the decision-trace summary, plus their deterministic wiring
into :meth:`CompanyReadService._assemble`.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

from app.schemas.company_profile import (
    CompanyBenchmarkSummary,
    CompanyDecisionSummary,
)
from app.services.companies import CompanyIdentityResolver
from app.services.company_dataset import DatasetCompanyStore
from app.services.company_protocols import CompanyRecord, CompanySnapshotRecord
from app.services.company_read import CompanyReadService, classify_coverage
from predictron_engine.dataset.models import (
    CompanyProfile,
    DatasetRecord,
    DecisionLabel,
    PredictionSummary,
)
from predictron_engine.dataset.store import DatasetStore
from tests.company.conftest import MemoryCompanyStore

CompanyService = CompanyReadService

REF = datetime(2025, 5, 1, 12, 0, tzinfo=UTC)

_SCORES = [10.0, 20.0, 30.0, 40.0, 50.0]


def _grounded_record(
    startup_name: str = "Acme Corp",
    website: str = "https://acme.example.com",
    composite_score: float = 68.5,
    record_id: str | None = None,
    analysis_date: datetime = REF,
) -> DatasetRecord:
    """A dataset record that passes every CIH placeholder quality gate."""
    return DatasetRecord(
        startup_name=startup_name,
        website=website,
        analysis_date=analysis_date,
        engine_version="0.12.1",
        evidence_bundle_reference="bundle://acme",
        prediction=PredictionSummary(
            decision=DecisionLabel.INVEST,
            confidence=0.75,
            composite_score=composite_score,
            dimension_scores={"market": 70.0, "team": 75.0},
            recommendation_count=2,
        ),
        profile=CompanyProfile(
            domain="acme.example.com",
            country_code="US",
            industries=["saas"],
        ),
        record_id=record_id,
    )


def _dataset_with(tmp_path, records: list[DatasetRecord]) -> DatasetStore:
    store = DatasetStore(tmp_path / "dataset")
    store.initialize()
    for record in records:
        store.save_record(record)
    return store


def _company_record() -> CompanyRecord:
    return CompanyRecord(
        company_id="comp-001",
        canonical_name="acme",
        canonical_domain="acme.example.com",
        primary_name="Acme Corp",
        website="https://acme.example.com",
        latest_decision=None,
        latest_confidence=None,
        latest_composite_score=None,
        snapshot_count=0,
        first_seen=REF,
        last_seen=REF,
        user_id=None,
        canonical_name_key="acme",
        fallback_slug="acme",
    )


def _snapshot(composite_score: float) -> CompanySnapshotRecord:
    return CompanySnapshotRecord(
        id="snap-1",
        company_id="comp-001",
        analysis_id="aid-1",
        report_id="rid-1",
        decision="invest",
        confidence=0.8,
        composite_score=composite_score,
        readiness_score=62.0,
        dimension_scores={"market": 70.0},
        created_at=REF,
    )


def _service(
    tmp_path, records, *, store: MemoryCompanyStore | None = None
) -> tuple[CompanyService, DatasetCompanyStore]:
    dataset = DatasetCompanyStore(
        _dataset_with(tmp_path, records),
        as_of=REF,
    )
    live_store = store or MemoryCompanyStore()
    service = CompanyReadService(store=live_store, dataset=dataset, as_of=REF)
    return service, dataset


def test_benchmark_summary_population_and_percentile(tmp_path):
    records = [
        _grounded_record(record_id=f"r{i}", composite_score=score)
        for i, score in enumerate(_SCORES)
    ]
    _, dataset = _service(tmp_path, records)
    summary = CompanyBenchmarkSummary(**dataset.benchmark_summary(30.0))
    assert summary.sample_size == 5
    assert summary.composite_score == 30.0
    assert summary.benchmark_mean == 30.0
    assert summary.percentile_rank == 50.0
    assert summary.z_score == 0.0

    low = CompanyBenchmarkSummary(**dataset.benchmark_summary(10.0))
    assert low.percentile_rank == 10.0
    high = CompanyBenchmarkSummary(**dataset.benchmark_summary(55.0))
    assert high.percentile_rank == 100.0


def test_benchmark_summary_empty_store(tmp_path):
    _, dataset = _service(tmp_path, [])
    summary = CompanyBenchmarkSummary(**dataset.benchmark_summary(30.0))
    assert summary.sample_size == 0
    assert summary.benchmark_mean == 0.0
    assert summary.percentile_rank == 0.0


def test_decision_summary_deterministic_and_references_record(tmp_path):
    records = [_grounded_record(record_id="acme-1")]
    _, dataset = _service(tmp_path, records)
    first = CompanyDecisionSummary(**dataset.decision_summary(records[0]))
    second = CompanyDecisionSummary(**dataset.decision_summary(records[0]))

    assert first.verdict
    assert 0.0 <= first.confidence <= 1.0
    assert first.feature_count > 0
    assert first.node_count > 0
    assert first.source_record_id == "acme-1"

    for field in ("verdict", "confidence", "overall_score", "feature_count",
                  "positive_factor_count", "negative_factor_count",
                  "node_count", "edge_count", "top_strengths",
                  "top_weaknesses", "source_record_id"):
        assert getattr(first, field) == getattr(second, field)


def test_assemble_offline_links_benchmark_and_decision(tmp_path):
    records = [
        _grounded_record(composite_score=55.0, record_id="acme-old", analysis_date=REF),
        _grounded_record(
            composite_score=80.0, record_id="acme-new", analysis_date=REF.replace(day=2)
        ),
    ]
    service, _ = _service(tmp_path, records)

    profile = service._assemble(_company_record(), [], 0)

    assert profile.coverage.value == "offline"
    assert profile.benchmark_summary is not None
    assert profile.benchmark_summary.composite_score == 80.0
    assert profile.benchmark_summary.sample_size == 2
    assert profile.decision_summary is not None
    assert profile.decision_summary.source_record_id == "acme-new"
    assert profile.decision_summary.verdict
    if profile.feature_summary is not None:
        assert profile.feature_summary.feature_count == profile.decision_summary.feature_count


def test_assemble_live_benchmark_uses_latest_snapshot(tmp_path):
    records = [_grounded_record(composite_score=60.0, record_id="acme-1")]
    service, _ = _service(tmp_path, records)

    profile = service._assemble(_company_record(), [_snapshot(99.0)], 1)

    assert profile.coverage.value == "live"
    assert profile.latest_snapshot is not None
    assert profile.latest_snapshot.composite_score == 99.0
    assert profile.benchmark_summary is not None
    assert profile.benchmark_summary.composite_score == 99.0
    assert profile.decision_summary is not None
    assert profile.decision_summary.source_record_id == "acme-1"


def test_assemble_no_data_yields_no_linkage(tmp_path):
    service, _ = _service(tmp_path, [])

    profile = service._assemble(_company_record(), [], 0)

    assert profile.coverage.value == "none"
    assert profile.benchmark_summary is None
    assert profile.decision_summary is None


def test_completeness_and_coverage_helpers() -> None:
    coverage, reasons = classify_coverage(
        live_has_data=False,
        has_offline=True,
        offline_placeholder=False,
    )
    assert coverage.value == "offline"
    assert reasons


def _identity():
    return CompanyIdentityResolver().resolve("Acme Corp", "https://acme.example.com")


def _placeholder_record(record_id: str = "acme-ph") -> DatasetRecord:
    return DatasetRecord(
        startup_name="Acme Corp",
        website="https://acme.example.com",
        engine_version="0.12.1",
        prediction=PredictionSummary(
            decision=DecisionLabel.INVEST,
            confidence=0.75,
            composite_score=68.5,
            dimension_scores={"market": 70.0},
        ),
        record_id=record_id,
    )


def test_record_for_gates_placeholders(tmp_path):
    _, dataset = _service(tmp_path, [])

    assert dataset.record_for([]) is None
    assert dataset.record_for([_placeholder_record()]) is None

    grounded = _grounded_record(record_id="acme-1")
    record = dataset.record_for([grounded])
    assert record is not None
    assert record.company_id == _identity().company_id
    assert record.snapshot_count == 1
    assert record.latest_composite_score == 68.5


async def test_resolve_offline_only_fallback(tmp_path):
    records = [_grounded_record(record_id="acme-1", composite_score=70.0)]
    service, _ = _service(tmp_path, records)

    profile = await service.resolve(
        None, name="Acme Corp", website="https://acme.example.com"
    )

    assert profile is not None
    assert profile.coverage.value == "offline"
    assert profile.company_id == _identity().company_id
    assert profile.benchmark_summary is not None
    assert profile.decision_summary is not None
    assert profile.history.snapshot_count == 0


async def test_resolve_prefers_live_registry(tmp_path):
    identity = _identity()
    live = dataclasses.replace(
        _company_record(),
        company_id=identity.company_id,
        canonical_name=identity.canonical_name,
        canonical_domain=identity.canonical_domain,
        website=identity.website,
    )
    store = MemoryCompanyStore()
    store._companies[identity.company_id] = live
    service, _ = _service(tmp_path, [], store=store)

    profile = await service.resolve(
        None, name="Acme Corp", website="https://acme.example.com"
    )

    assert profile is not None
    assert profile.coverage.value == "none"


async def test_resolve_unknown_returns_none(tmp_path):
    service, _ = _service(tmp_path, [])

    assert await service.resolve(None, name="Nobody Inc", website=None) is None
