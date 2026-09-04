"""Tests for provenance tracking (Part D)."""

from __future__ import annotations

from datetime import UTC, datetime

from predictron_engine.dataset.imports import RawImportRecord
from predictron_engine.dataset.provenance import FieldProvenance, ProvenanceTracker
from tests.dataset.conftest import make_record

FIXED_DATE = datetime(2024, 1, 1, tzinfo=UTC)


def test_build_provenance_tracks_present_fields() -> None:
    raw = RawImportRecord(
        startup_name="Acme",
        website="https://acme.example.com",
        engine_version="0.12.1",
        prediction_data={"decision": "invest", "confidence": 0.8},
        metadata={"sector": "SaaS", "country": "US"},
        source="yc_oss",
    )
    tracker = ProvenanceTracker(FIXED_DATE)
    provenance = tracker.build_provenance(
        raw, ["startup_name", "website", "analysis_date", "engine_version"]
    )
    assert provenance["startup_name"]["source"] == "yc_oss"
    assert provenance["startup_name"]["retrieval_date"] == FIXED_DATE
    assert provenance["website"]["source"] == "yc_oss"
    assert "analysis_date" not in provenance  # not set -> not tracked
    assert provenance["prediction.decision"]["source"] == "yc_oss"
    assert provenance["metadata.sector"]["source"] == "yc_oss"


def test_attach_and_extract() -> None:
    raw = RawImportRecord(
        startup_name="Acme",
        website="https://acme.example.com",
        source="json_file",
    )
    tracker = ProvenanceTracker(FIXED_DATE)
    provenance = tracker.build_provenance(raw, ["startup_name", "website"])
    record = make_record()
    updated = tracker.attach_to_record(record, provenance)

    # Original record unmodified
    assert "provenance" not in record.analysis_metadata
    assert updated.analysis_metadata["provenance"] == provenance
    assert ProvenanceTracker.extract_from_record(updated) == provenance


def test_field_provenance_dict() -> None:
    fp = FieldProvenance("website", "sec_edgar", FIXED_DATE)
    d = fp.to_dict()
    assert d == {
        "field": "website",
        "source": "sec_edgar",
        "retrieval_date": FIXED_DATE,
    }
