"""Tests for DatasetCompanyStore's conformance to the CompanyStore protocol.

The offline adapter must be structurally interchangeable with the SQL store
for *reads* (tick the runtime protocol check) while remaining strictly
read-only: the two write methods raise ``NotImplementedError``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.services.companies import CompanyIdentityResolver
from app.services.company_dataset import DatasetCompanyStore
from app.services.company_protocols import (
    CompanyPage,
    CompanyRecord,
    CompanySnapshotPayload,
    CompanySnapshotRecord,
    CompanyStore,
    SnapshotPage,
)
from predictron_engine.dataset.models import (
    CompanyProfile,
    DatasetRecord,
    DecisionLabel,
    PredictionSummary,
)
from predictron_engine.dataset.store import DatasetStore

REF = datetime(2025, 5, 1, 12, 0, tzinfo=UTC)

_ACME_ID = CompanyIdentityResolver().resolve(
    "Acme Corp", "https://acme.example.com"
).company_id
_BETA_ID = CompanyIdentityResolver().resolve(
    "Beta Inc", "https://beta.example.com"
).company_id


def _grounded_record(
    startup_name: str,
    website: str,
    *,
    record_id: str,
    composite_score: float,
    analysis_date: datetime = REF,
) -> DatasetRecord:
    return DatasetRecord(
        startup_name=startup_name,
        website=website,
        analysis_date=analysis_date,
        engine_version="0.12.1",
        evidence_bundle_reference=f"bundle://{record_id}",
        prediction=PredictionSummary(
            decision=DecisionLabel.INVEST,
            confidence=0.75,
            composite_score=composite_score,
            dimension_scores={"market": 70.0, "team": 75.0},
            recommendation_count=2,
        ),
        profile=CompanyProfile(
            domain=website,
            country_code="US",
            industries=["saas"],
        ),
        record_id=record_id,
    )


def _dataset(tmp_path, records) -> DatasetStore:
    store = DatasetStore(tmp_path / "dataset")
    store.initialize()
    for record in records:
        store.save_record(record)
    return store


def _adapter(tmp_path, records) -> DatasetCompanyStore:
    return DatasetCompanyStore(_dataset(tmp_path, records), as_of=REF)


async def test_dataset_store_is_company_store_protocol(tmp_path):
    adapter = _adapter(tmp_path, [])
    assert isinstance(adapter, CompanyStore)


async def test_protocol_get_company_maps_grounded_records(tmp_path):
    adapter = _adapter(
        tmp_path,
        [
            _grounded_record(
                "Acme Corp",
                "https://acme.example.com",
                record_id="acme-1",
                composite_score=60.0,
                analysis_date=REF,
            ),
            _grounded_record(
                "Acme Corp",
                "https://acme.example.com",
                record_id="acme-2",
                composite_score=80.0,
                analysis_date=REF.replace(day=2),
            ),
        ],
    )

    company = await adapter.get_company(None, _ACME_ID)

    assert isinstance(company, CompanyRecord)
    assert company.company_id == _ACME_ID
    assert company.snapshot_count == 2
    assert company.latest_composite_score == 80.0
    assert company.latest_decision == "invest"
    assert company.latest_confidence == 0.75
    assert company.user_id is None


async def test_protocol_get_company_unknown_id_is_none(tmp_path):
    adapter = _adapter(tmp_path, [])
    assert await adapter.get_company(None, "does-not-exist") is None


async def test_protocol_list_companies_groups_sorts_and_paginates(tmp_path):
    adapter = _adapter(
        tmp_path,
        [
            _grounded_record(
                "Acme Corp",
                "https://acme.example.com",
                record_id="acme-1",
                composite_score=60.0,
            ),
            _grounded_record(
                "Beta Inc",
                "https://beta.example.com",
                record_id="beta-1",
                composite_score=70.0,
                analysis_date=REF.replace(day=3),
            ),
        ],
    )

    page = await adapter.list_companies(None, limit=1)

    assert isinstance(page, CompanyPage)
    assert page.total == 2
    assert [c.company_id for c in page.companies] == [_BETA_ID]

    second = await adapter.list_companies(None, offset=1, limit=1)
    assert [c.company_id for c in second.companies] == [_ACME_ID]
    assert second.companies[0].snapshot_count == 1


async def test_protocol_list_snapshots_newest_first(tmp_path):
    adapter = _adapter(
        tmp_path,
        [
            _grounded_record(
                "Acme Corp",
                "https://acme.example.com",
                record_id="acme-old",
                composite_score=60.0,
                analysis_date=REF,
            ),
            _grounded_record(
                "Acme Corp",
                "https://acme.example.com",
                record_id="acme-new",
                composite_score=80.0,
                analysis_date=REF.replace(day=2),
            ),
        ],
    )

    page = await adapter.list_snapshots(None, _ACME_ID)

    assert isinstance(page, SnapshotPage)
    assert page.total == 2
    assert [s.analysis_id for s in page.snapshots] == ["acme-new", "acme-old"]
    snap = page.snapshots[0]
    assert isinstance(snap, CompanySnapshotRecord)
    assert snap.company_id == _ACME_ID
    assert snap.composite_score == 80.0
    assert snap.created_at == REF.replace(day=2)


async def test_protocol_reads_ignore_placeholder_records(tmp_path):
    store = _dataset(
        tmp_path,
        [
            DatasetRecord(
                startup_name="Acme Corp",
                website="https://acme.example.com",
                engine_version="0.12.1",
                prediction=PredictionSummary(
                    decision=DecisionLabel.INVEST,
                    confidence=0.75,
                    composite_score=68.5,
                    dimension_scores={"market": 70.0},
                ),
                record_id="acme-ph",
            )
        ],
    )
    adapter = DatasetCompanyStore(store, as_of=REF)

    assert await adapter.get_company(None, _ACME_ID) is None
    page = await adapter.list_companies(None)
    assert page.total == 0
    snap_page = await adapter.list_snapshots(None, _ACME_ID)
    assert snap_page.total == 0


async def test_protocol_writes_are_unsupported(tmp_path):
    import pytest

    adapter = _adapter(tmp_path, [])
    identity = CompanyIdentityResolver().resolve("Acme Corp", None)

    with pytest.raises(NotImplementedError):
        await adapter.upsert_company(None, identity)
    with pytest.raises(NotImplementedError):
        await adapter.append_snapshot(
            None,
            CompanySnapshotPayload(
                company_id=identity.company_id, analysis_id="a", report_id="a"
            ),
        )
