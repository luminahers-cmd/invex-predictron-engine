"""Tests for the Evidence Retrieval module (Sprint 5B).

Tests cover:
- retrieve_documents_by_type() filtering
- retrieve_documents_by_provider() filtering
- retrieve_trusted_documents() filtering and ranking
- retrieve_best_source() selection
- retrieve_evidence_for_domain() filtering
- retrieve_citations_for_domain() integration
- retrieve_citations_for_observation() reference matching
- _match_evidence_by_refs() reference parsing
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from predictron_engine.evidence.models import (
    DocumentMetadata,
    DocumentStatus,
    DocumentType,
    EvidenceBundle,
    EvidenceDocument,
    EvidenceSource,
    PageType,
    make_document_id,
)
from predictron_engine.evidence.provenance import TrustFactor, TrustScore
from predictron_engine.evidence.retrieval import (
    retrieve_best_source,
    retrieve_citations_for_domain,
    retrieve_citations_for_observation,
    retrieve_documents_by_provider,
    retrieve_documents_by_type,
    retrieve_evidence_for_domain,
    retrieve_trusted_documents,
)
from predictron_engine.models.report import EvidenceItem

# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────


@pytest.fixture
def about_doc() -> EvidenceDocument:
    return EvidenceDocument(
        id=make_document_id("https://example.com/about"),
        original_url="https://example.com/about",
        url="https://example.com/about",
        page_type=PageType.ABOUT,
        status=DocumentStatus.SUCCESS,
        fetched_at=datetime(2025, 6, 1, tzinfo=UTC),
        response_time_ms=200,
        http_status=200,
        title="About ExampleCorp",
        text="ExampleCorp is a leader...",
        metadata=DocumentMetadata(
            document_type=DocumentType.ABOUT,
            authority_score=0.9,
            quality_score=0.85,
            trust_score=TrustScore(
                overall=0.85,
                factors=[TrustFactor(name="authority", weight=0.3, value=0.9)],
                freshness_days=30,
            ),
            source_provider="website_provider",
        ),
    )


@pytest.fixture
def blog_doc() -> EvidenceDocument:
    return EvidenceDocument(
        id=make_document_id("https://example.com/blog/post1"),
        original_url="https://example.com/blog/post1",
        url="https://example.com/blog/post1",
        page_type=PageType.UNKNOWN,
        status=DocumentStatus.SUCCESS,
        fetched_at=datetime(2025, 6, 1, tzinfo=UTC),
        response_time_ms=300,
        http_status=200,
        title="Blog Post",
        text="Blog content...",
        metadata=DocumentMetadata(
            document_type=DocumentType.BLOG,
            authority_score=0.5,
            quality_score=0.6,
            trust_score=TrustScore(
                overall=0.55,
                factors=[],
                freshness_days=60,
            ),
            source_provider="search_provider",
        ),
    )


@pytest.fixture
def failed_doc() -> EvidenceDocument:
    return EvidenceDocument(
        id=make_document_id("https://failed.com/page"),
        original_url="https://failed.com/page",
        url="https://failed.com/page",
        page_type=PageType.UNKNOWN,
        status=DocumentStatus.FAILED,
        fetched_at=datetime(2025, 6, 1, tzinfo=UTC),
        response_time_ms=5000,
        http_status=500,
        error="Server error",
    )


@pytest.fixture
def sample_bundle(
    about_doc: EvidenceDocument,
    blog_doc: EvidenceDocument,
    failed_doc: EvidenceDocument,
) -> EvidenceBundle:
    return EvidenceBundle(
        startup_name="ExampleCorp",
        website="https://example.com",
        documents=[about_doc, blog_doc, failed_doc],
        sources=[
            EvidenceSource(
                original_url="https://example.com/about",
                page_type=PageType.ABOUT,
                fetched_at=datetime(2025, 6, 1, tzinfo=UTC),
                success=True,
                url="https://example.com/about",
                http_status=200,
            ),
            EvidenceSource(
                original_url="https://failed.com/page",
                page_type=PageType.UNKNOWN,
                fetched_at=datetime(2025, 6, 1, tzinfo=UTC),
                success=False,
                error="Server error",
            ),
        ],
    )


@pytest.fixture
def evidence_items() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            domain="industry",
            category="sales_cycle",
            statement="Enterprise sales cycles are 12-18 months",
            source="industry_provider",
            relevance_score=0.9,
        ),
        EvidenceItem(
            domain="industry",
            category="regulatory",
            statement="Healthcare requires FDA approval",
            source="industry_provider",
            relevance_score=0.8,
        ),
        EvidenceItem(
            domain="team",
            category="founder_experience",
            statement="Serial founders have higher success rates",
            source="team_provider",
            relevance_score=0.7,
        ),
    ]


# ────────────────────────────────────────────────────────────────────
# retrieve_documents_by_type Tests
# ────────────────────────────────────────────────────────────────────


class TestRetrieveDocumentsByType:
    def test_filters_by_type(self, sample_bundle: EvidenceBundle) -> None:
        result = retrieve_documents_by_type(sample_bundle, DocumentType.ABOUT)
        assert len(result) == 1
        assert result[0].metadata.document_type == DocumentType.ABOUT

    def test_excludes_failed_documents(self, sample_bundle: EvidenceBundle) -> None:
        result = retrieve_documents_by_type(sample_bundle, DocumentType.UNKNOWN)
        assert len(result) == 0

    def test_empty_bundle(self) -> None:
        bundle = EvidenceBundle(startup_name="empty")
        result = retrieve_documents_by_type(bundle, DocumentType.ABOUT)
        assert result == []

    def test_no_matching_type(self, sample_bundle: EvidenceBundle) -> None:
        result = retrieve_documents_by_type(sample_bundle, DocumentType.PRICING)
        assert result == []


# ────────────────────────────────────────────────────────────────────
# retrieve_documents_by_provider Tests
# ────────────────────────────────────────────────────────────────────


class TestRetrieveDocumentsByProvider:
    def test_filters_by_provider(self, sample_bundle: EvidenceBundle) -> None:
        result = retrieve_documents_by_provider(sample_bundle, "website_provider")
        assert len(result) == 1
        assert result[0].metadata.source_provider == "website_provider"

    def test_search_provider(self, sample_bundle: EvidenceBundle) -> None:
        result = retrieve_documents_by_provider(sample_bundle, "search_provider")
        assert len(result) == 1
        assert result[0].metadata.source_provider == "search_provider"

    def test_no_match(self, sample_bundle: EvidenceBundle) -> None:
        result = retrieve_documents_by_provider(sample_bundle, "nonexistent")
        assert result == []


# ────────────────────────────────────────────────────────────────────
# retrieve_trusted_documents Tests
# ────────────────────────────────────────────────────────────────────


class TestRetrieveTrustedDocuments:
    def test_filters_by_min_trust(self, sample_bundle: EvidenceBundle) -> None:
        result = retrieve_trusted_documents(sample_bundle, min_trust=0.7)
        assert len(result) == 1
        assert result[0].metadata.trust_score.overall >= 0.7

    def test_default_min_trust(self, sample_bundle: EvidenceBundle) -> None:
        result = retrieve_trusted_documents(sample_bundle)
        assert len(result) >= 1

    def test_sorted_by_trust_descending(self, sample_bundle: EvidenceBundle) -> None:
        result = retrieve_trusted_documents(sample_bundle, min_trust=0.0)
        if len(result) >= 2:
            for i in range(len(result) - 1):
                left = result[i].metadata.trust_score.overall
                right = result[i + 1].metadata.trust_score.overall
                assert left >= right

    def test_empty_bundle(self) -> None:
        bundle = EvidenceBundle(startup_name="empty")
        result = retrieve_trusted_documents(bundle)
        assert result == []


# ────────────────────────────────────────────────────────────────────
# retrieve_best_source Tests
# ────────────────────────────────────────────────────────────────────


class TestRetrieveBestSource:
    def test_returns_highest_trust(self, sample_bundle: EvidenceBundle) -> None:
        result = retrieve_best_source(sample_bundle)
        assert result is not None
        assert result.metadata.trust_score.overall == 0.85

    def test_empty_bundle(self) -> None:
        bundle = EvidenceBundle(startup_name="empty")
        assert retrieve_best_source(bundle) is None

    def test_no_trusted_documents(self) -> None:
        bundle = EvidenceBundle(
            startup_name="test",
            documents=[
                EvidenceDocument(
                    id=make_document_id("https://x.com"),
                    original_url="https://x.com",
                    url="https://x.com",
                    page_type=PageType.UNKNOWN,
                    status=DocumentStatus.SUCCESS,
                    fetched_at=datetime(2025, 6, 1, tzinfo=UTC),
                    response_time_ms=100,
                    http_status=200,
                    metadata=DocumentMetadata(trust_score=None),
                ),
            ],
        )
        assert retrieve_best_source(bundle) is None


# ────────────────────────────────────────────────────────────────────
# retrieve_evidence_for_domain Tests
# ────────────────────────────────────────────────────────────────────


class TestRetrieveEvidenceForDomain:
    def test_filters_by_domain(self, evidence_items: list[EvidenceItem]) -> None:
        result = retrieve_evidence_for_domain(evidence_items, "industry")
        assert len(result) == 2
        assert all(e.domain == "industry" for e in result)

    def test_single_match(self, evidence_items: list[EvidenceItem]) -> None:
        result = retrieve_evidence_for_domain(evidence_items, "team")
        assert len(result) == 1

    def test_no_match(self, evidence_items: list[EvidenceItem]) -> None:
        result = retrieve_evidence_for_domain(evidence_items, "nonexistent")
        assert result == []


# ────────────────────────────────────────────────────────────────────
# retrieve_citations_for_domain Tests
# ────────────────────────────────────────────────────────────────────


class TestRetrieveCitationsForDomain:
    def test_builds_citations_for_domain(
        self, evidence_items: list[EvidenceItem]
    ) -> None:
        result = retrieve_citations_for_domain(evidence_items, "industry")
        assert len(result) == 2
        assert all(c.domain == "industry" for c in result)

    def test_empty_items(self) -> None:
        result = retrieve_citations_for_domain([], "industry")
        assert result == []


# ────────────────────────────────────────────────────────────────────
# retrieve_citations_for_observation Tests
# ────────────────────────────────────────────────────────────────────


class TestRetrieveCitationsForObservation:
    def test_matches_refs_to_items(
        self, evidence_items: list[EvidenceItem]
    ) -> None:
        refs = [
            "evidence:industry/sales_cycle: Enterprise sales cycles are 12-18 months",
        ]
        result = retrieve_citations_for_observation(refs, evidence_items)
        assert len(result) == 1
        assert result[0].claim == "Enterprise sales cycles are 12-18 months"

    def test_multiple_refs(self, evidence_items: list[EvidenceItem]) -> None:
        refs = [
            "evidence:industry/sales_cycle: Enterprise sales cycles are 12-18 months",
            "evidence:industry/regulatory: Healthcare requires FDA approval",
        ]
        result = retrieve_citations_for_observation(refs, evidence_items)
        assert len(result) == 2

    def test_non_evidence_refs_ignored(self, evidence_items: list[EvidenceItem]) -> None:
        refs = [
            "feature:industry=healthtech",
            "evidence:industry/sales_cycle: Enterprise sales cycles are 12-18 months",
        ]
        result = retrieve_citations_for_observation(refs, evidence_items)
        assert len(result) == 1

    def test_no_matching_refs(self, evidence_items: list[EvidenceItem]) -> None:
        refs = [
            "evidence:nonexistent/category: This does not exist",
        ]
        result = retrieve_citations_for_observation(refs, evidence_items)
        assert len(result) == 0
