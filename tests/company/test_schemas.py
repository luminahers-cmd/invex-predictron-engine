"""Tests for the read-side Pydantic response schemas of the CIH."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.schemas.company import (
    CompanyDetailResponse,
    CompanyHistoryResponse,
    CompanyListResponse,
    CompanySnapshotResponse,
    CompanySummaryResponse,
)

NOW = datetime(2025, 5, 1, 12, 0, tzinfo=UTC)


def _summary(**overrides):
    base = {
        "company_id": "abc123",
        "canonical_name": "acme",
        "canonical_domain": "acme.com",
        "primary_name": "Acme Inc",
        "website": "https://acme.com",
        "latest_decision": "invest",
        "latest_confidence": 0.85,
        "latest_composite_score": 75.0,
        "snapshot_count": 3,
        "first_seen": NOW,
        "last_seen": NOW,
    }
    base.update(overrides)
    return CompanySummaryResponse(**base)


def test_summary_roundtrip():
    s = _summary()
    assert s.company_id == "abc123"
    assert s.canonical_domain == "acme.com"
    assert s.snapshot_count == 3


def test_summary_nullable_fields_default():
    s = _summary(
        canonical_domain=None,
        website=None,
        latest_decision=None,
        latest_confidence=None,
        latest_composite_score=None,
    )
    assert s.canonical_domain is None
    assert s.website is None
    assert s.latest_decision is None


def test_summary_requires_company_id():
    with pytest.raises(ValidationError):
        _summary(company_id=None)


def test_summary_requires_canonical_name():
    with pytest.raises(ValidationError):
        _summary(canonical_name=None)


def test_summary_requires_primary_name():
    with pytest.raises(ValidationError):
        _summary(primary_name=None)


def test_summary_rejects_negative_snapshot_count():
    with pytest.raises(ValidationError):
        _summary(snapshot_count=-1)


def test_summary_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        _summary(latest_confidence=1.5)


def test_summary_rejects_confidence_below_zero():
    with pytest.raises(ValidationError):
        _summary(latest_confidence=-0.1)


def test_summary_rejects_composite_above_100():
    with pytest.raises(ValidationError):
        _summary(latest_composite_score=101)


def test_summary_composite_zero_ok():
    assert _summary(latest_composite_score=0).latest_composite_score == 0


def test_snapshot_roundtrip():
    snap = CompanySnapshotResponse(
        id="snap1",
        company_id="abc123",
        analysis_id="aid1",
        report_id="rid1",
        decision="invest",
        confidence=0.8,
        composite_score=70.0,
        readiness_score=65.0,
        dimension_scores={"market": 80.0},
        created_at=NOW,
    )
    assert snap.decision == "invest"
    assert snap.dimension_scores == {"market": 80.0}


def test_snapshot_nullable_fields():
    snap = CompanySnapshotResponse(
        id="snap2",
        company_id="abc123",
        analysis_id="aid2",
        report_id="rid2",
        created_at=NOW,
    )
    assert snap.decision is None
    assert snap.dimension_scores == {}


def test_snapshot_requires_created_at():
    with pytest.raises(ValidationError):
        CompanySnapshotResponse(
            id="s",
            company_id="c",
            analysis_id="a",
            report_id="r",
        )


def test_snapshot_requires_ids():
    with pytest.raises(ValidationError):
        CompanySnapshotResponse(
            id="s",
            company_id="c",
            report_id="r",
            created_at=NOW,
        )


def test_snapshot_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        CompanySnapshotResponse(
            id="s",
            company_id="c",
            analysis_id="a",
            report_id="r",
            confidence=2.0,
            created_at=NOW,
        )


def test_snapshot_rejects_out_of_range_readiness():
    with pytest.raises(ValidationError):
        CompanySnapshotResponse(
            id="s",
            company_id="c",
            analysis_id="a",
            report_id="r",
            readiness_score=200.0,
            created_at=NOW,
        )


def test_detail_inherits_summary_fields():
    d = CompanyDetailResponse(
        **_summary().model_dump(),
        latest_snapshot=CompanySnapshotResponse(
            id="s1",
            company_id="abc123",
            analysis_id="aid9",
            report_id="rid9",
            created_at=NOW,
        ),
    )
    assert d.snapshot_count == 3
    assert d.latest_snapshot is not None
    assert d.latest_snapshot.id == "s1"


def test_detail_latest_snapshot_optional():
    d = CompanyDetailResponse(**_summary().model_dump(), latest_snapshot=None)
    assert d.latest_snapshot is None


def test_detail_without_snapshot_serializes():
    d = CompanyDetailResponse(**_summary().model_dump())
    assert d.latest_snapshot is None


def test_list_roundtrip():
    items = [_summary(company_id="a"), _summary(company_id="b")]
    lst = CompanyListResponse(companies=items, total=2)
    assert lst.total == 2
    assert [c.company_id for c in lst.companies] == ["a", "b"]


def test_list_total_must_be_non_negative():
    with pytest.raises(ValidationError):
        CompanyListResponse(companies=[], total=-1)


def test_list_defaults():
    lst = CompanyListResponse()
    assert lst.companies == []
    assert lst.total == 0


def test_history_roundtrip():
    snaps = [
        CompanySnapshotResponse(
            id=f"s{i}",
            company_id="c",
            analysis_id=f"a{i}",
            report_id=f"r{i}",
            created_at=NOW,
        )
        for i in range(2)
    ]
    h = CompanyHistoryResponse(company_id="c", snapshots=snaps, total=2)
    assert h.company_id == "c"
    assert h.total == 2


def test_history_requires_company_id():
    with pytest.raises(ValidationError):
        CompanyHistoryResponse(snapshots=[], total=0)


def test_history_total_non_negative():
    with pytest.raises(ValidationError):
        CompanyHistoryResponse(company_id="c", snapshots=[], total=-1)


def test_summary_serializes_to_expected_keys():
    data = _summary().model_dump()
    expected = {
        "company_id",
        "canonical_name",
        "canonical_domain",
        "primary_name",
        "website",
        "latest_decision",
        "latest_confidence",
        "latest_composite_score",
        "snapshot_count",
        "first_seen",
        "last_seen",
    }
    assert set(data.keys()) == expected


def test_snapshot_serializes_to_expected_keys():
    snap = CompanySnapshotResponse(
        id="s1",
        company_id="c",
        analysis_id="a",
        report_id="r",
        decision="invest",
        confidence=0.8,
        composite_score=70.0,
        readiness_score=65.0,
        dimension_scores={},
        created_at=NOW,
    )
    expected = {
        "id",
        "company_id",
        "analysis_id",
        "report_id",
        "decision",
        "confidence",
        "composite_score",
        "readiness_score",
        "dimension_scores",
        "created_at",
    }
    assert set(snap.model_dump().keys()) == expected


def test_summary_dump_json_compatible():
    import json

    json.dumps(_summary().model_dump(mode="json"))
