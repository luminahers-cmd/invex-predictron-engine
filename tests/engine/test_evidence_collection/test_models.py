"""Tests for the Evidence Collection Layer Pydantic models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import HttpUrl

from predictron_engine.evidence.models import (
    DocumentStatus,
    EvidenceBundle,
    EvidenceDocument,
    EvidenceSource,
    PageType,
    RetrievalMethod,
    make_document_id,
)


def _document(url: str = "https://example.com") -> EvidenceDocument:
    return EvidenceDocument(
        id="doc-1",
        original_url=HttpUrl(url),
        url=HttpUrl(url),
        page_type=PageType.ABOUT,
        status=DocumentStatus.SUCCESS,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        response_time_ms=100,
        http_status=200,
    )


def _source(success: bool, url: str = "https://example.com") -> EvidenceSource:
    return EvidenceSource(
        original_url=HttpUrl(url),
        page_type=PageType.ABOUT,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        success=success,
        error=None if success else "HTTP 404",
    )


class TestMakeDocumentId:
    def test_deterministic_for_same_url(self) -> None:
        assert make_document_id("https://example.com/a") == make_document_id("https://example.com/a")

    def test_differs_across_urls(self) -> None:
        assert make_document_id("https://example.com/a") != make_document_id("https://example.com/b")

    def test_produces_valid_uuid(self) -> None:
        assert isinstance(uuid.UUID(make_document_id("https://example.com/a")), uuid.UUID)


class TestEvidenceDocument:
    def test_defaults(self) -> None:
        doc = _document()
        assert doc.retrieval_method == RetrievalMethod.HTTP_GET
        assert doc.text == ""
        assert doc.truncated is False

    def test_serializes_to_json(self) -> None:
        data = _document().model_dump_json()
        assert '"id":"doc-1"' in data
        assert '"page_type":"about"' in data


class TestEvidenceSource:
    def test_success_source(self) -> None:
        source = _source(success=True)
        assert source.success is True
        assert source.error is None

    def test_failure_source(self) -> None:
        source = _source(success=False)
        assert source.success is False
        assert source.error == "HTTP 404"


class TestEvidenceBundle:
    def test_total_pages_matches_documents(self) -> None:
        bundle = EvidenceBundle(
            startup_name="ExampleCo",
            website=HttpUrl("https://example.com"),
            documents=[_document()],
        )
        assert bundle.total_pages == 1

    def test_total_pages_serialized(self) -> None:
        bundle = EvidenceBundle(
            startup_name="ExampleCo",
            website=HttpUrl("https://example.com"),
            documents=[_document()],
        )
        assert bundle.model_dump()["total_pages"] == 1

    def test_failures_filters_failed_sources(self) -> None:
        bundle = EvidenceBundle(
            startup_name="ExampleCo",
            website=HttpUrl("https://example.com"),
            sources=[_source(success=True), _source(success=False)],
        )
        assert len(bundle.failures) == 1
        assert bundle.failures[0].error == "HTTP 404"

    def test_defaults(self) -> None:
        bundle = EvidenceBundle(startup_name="ExampleCo", website=HttpUrl("https://example.com"))
        assert bundle.documents == []
        assert bundle.sources == []
        assert bundle.duration_ms == 0
        assert bundle.attempted_pages == 0
