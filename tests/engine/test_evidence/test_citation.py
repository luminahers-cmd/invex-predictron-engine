"""Tests for the Evidence Citation Builder (Sprint 5B).

Tests cover:
- EvidenceCitation model construction and serialization
- build_citation() pure function with various document scenarios
- build_citations() bulk citation building
- format_citation() and format_citation_list() rendering
- rank_citations_by_trust() ordering
- deduplicate_citations() dedup logic
- Document matching heuristics (_match_documents)
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from predictron_engine.evidence.citation import (
    build_citation,
    build_citations,
    deduplicate_citations,
    format_citation,
    format_citation_list,
    rank_citations_by_trust,
)
from predictron_engine.evidence.models import (
    DocumentMetadata,
    DocumentStatus,
    DocumentType,
    EvidenceDocument,
    PageType,
    make_document_id,
)
from predictron_engine.evidence.provenance import TrustFactor, TrustScore
from predictron_engine.models.report import EvidenceCitation, EvidenceItem

# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_evidence_item() -> EvidenceItem:
    return EvidenceItem(
        domain="industry",
        category="sales_cycle",
        statement="Enterprise sales cycles are 12-18 months",
        source="industry_provider",
        relevance_score=0.9,
    )


@pytest.fixture
def high_trust_doc() -> EvidenceDocument:
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
        text="ExampleCorp is a leader in enterprise solutions...",
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
def low_trust_doc() -> EvidenceDocument:
    return EvidenceDocument(
        id=make_document_id("https://blog.example.com/post1"),
        original_url="https://blog.example.com/post1",
        url="https://blog.example.com/post1",
        page_type=PageType.UNKNOWN,
        status=DocumentStatus.SUCCESS,
        fetched_at=datetime(2025, 1, 1, tzinfo=UTC),
        response_time_ms=500,
        http_status=200,
        title="Blog Post",
        text="Some blog content...",
        metadata=DocumentMetadata(
            document_type=DocumentType.BLOG,
            authority_score=0.3,
            quality_score=0.4,
            trust_score=TrustScore(
                overall=0.35,
                factors=[TrustFactor(name="authority", weight=0.3, value=0.3)],
                freshness_days=180,
            ),
            source_provider="search_provider",
        ),
    )


@pytest.fixture
def no_trust_doc() -> EvidenceDocument:
    return EvidenceDocument(
        id=make_document_id("https://unknown.com/page"),
        original_url="https://unknown.com/page",
        url="https://unknown.com/page",
        page_type=PageType.UNKNOWN,
        status=DocumentStatus.SUCCESS,
        fetched_at=datetime(2025, 6, 1, tzinfo=UTC),
        response_time_ms=100,
        http_status=200,
        title="Unknown Page",
        text="Some content...",
        metadata=DocumentMetadata(
            document_type=DocumentType.UNKNOWN,
            authority_score=0.1,
            quality_score=0.2,
            trust_score=None,
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


# ────────────────────────────────────────────────────────────────────
# EvidenceCitation Model Tests
# ────────────────────────────────────────────────────────────────────


class TestEvidenceCitationModel:
    """Tests for the EvidenceCitation Pydantic model."""

    def test_minimal_construction(self) -> None:
        citation = EvidenceCitation(
            claim="Test claim",
            domain="industry",
            category="test",
        )
        assert citation.claim == "Test claim"
        assert citation.domain == "industry"
        assert citation.category == "test"
        assert citation.source_document_ids == []
        assert citation.source_urls == []
        assert citation.best_trust_score == 0.0
        assert citation.citation_text == ""
        assert citation.provider == ""

    def test_full_construction(self) -> None:
        citation = EvidenceCitation(
            claim="Enterprise sales cycles are long",
            domain="industry",
            category="sales_cycle",
            source_document_ids=["doc-1", "doc-2"],
            source_urls=["https://example.com/about"],
            best_trust_score=0.85,
            citation_text=(
                "[industry/sales_cycle] Enterprise sales cycles are long"
                " (source: example.com)"
            ),
            provider="industry_provider",
        )
        assert len(citation.source_document_ids) == 2
        assert citation.best_trust_score == 0.85
        assert citation.provider == "industry_provider"

    def test_serialization_roundtrip(self) -> None:
        citation = EvidenceCitation(
            claim="Test",
            domain="industry",
            category="test",
            best_trust_score=0.75,
        )
        data = citation.model_dump()
        restored = EvidenceCitation.model_validate(data)
        assert restored == citation


# ────────────────────────────────────────────────────────────────────
# build_citation Tests
# ────────────────────────────────────────────────────────────────────


class TestBuildCitation:
    """Tests for the build_citation pure function."""

    def test_no_documents(self, sample_evidence_item: EvidenceItem) -> None:
        citation = build_citation(sample_evidence_item)
        assert citation.claim == "Enterprise sales cycles are 12-18 months"
        assert citation.domain == "industry"
        assert citation.category == "sales_cycle"
        assert citation.source_document_ids == []
        assert citation.source_urls == []
        assert citation.best_trust_score == 0.0

    def test_with_high_trust_doc(
        self, sample_evidence_item: EvidenceItem, high_trust_doc: EvidenceDocument
    ) -> None:
        citation = build_citation(sample_evidence_item, [high_trust_doc])
        assert citation.best_trust_score == 0.85
        assert high_trust_doc.id in citation.source_document_ids
        assert "example.com" in citation.citation_text

    def test_picks_highest_trust(
        self,
        sample_evidence_item: EvidenceItem,
        high_trust_doc: EvidenceDocument,
        low_trust_doc: EvidenceDocument,
    ) -> None:
        citation = build_citation(sample_evidence_item, [high_trust_doc, low_trust_doc])
        assert citation.best_trust_score == 0.85

    def test_empty_documents_list(self, sample_evidence_item: EvidenceItem) -> None:
        citation = build_citation(sample_evidence_item, [])
        assert citation.best_trust_score == 0.0
        assert citation.source_document_ids == []

    def test_deduplicates_document_ids(
        self, sample_evidence_item: EvidenceItem, high_trust_doc: EvidenceDocument
    ) -> None:
        citation = build_citation(sample_evidence_item, [high_trust_doc, high_trust_doc])
        assert citation.source_document_ids.count(high_trust_doc.id) == 1

    def test_provider_from_evidence_item(
        self, sample_evidence_item: EvidenceItem
    ) -> None:
        citation = build_citation(sample_evidence_item)
        assert citation.provider == "industry_provider"


# ────────────────────────────────────────────────────────────────────
# build_citations Tests
# ────────────────────────────────────────────────────────────────────


class TestBuildCitations:
    """Tests for the build_citations bulk function."""

    def test_empty_items(self) -> None:
        citations = build_citations([])
        assert citations == []

    def test_multiple_items(self) -> None:
        items = [
            EvidenceItem(
                domain="industry", category="a", statement="A", source="prov1",
            ),
            EvidenceItem(
                domain="industry", category="b", statement="B", source="prov2",
            ),
        ]
        citations = build_citations(items)
        assert len(citations) == 2
        assert citations[0].claim == "A"
        assert citations[1].claim == "B"

    def test_items_with_documents(self) -> None:
        items = [
            EvidenceItem(
                domain="industry",
                category="test",
                statement="Test statement about enterprise sales",
                source="https://example.com/about",
            ),
        ]
        doc = EvidenceDocument(
            id=make_document_id("https://example.com/about"),
            original_url="https://example.com/about",
            url="https://example.com/about",
            page_type=PageType.ABOUT,
            status=DocumentStatus.SUCCESS,
            fetched_at=datetime(2025, 6, 1, tzinfo=UTC),
            response_time_ms=200,
            http_status=200,
            title="About ExampleCorp",
            metadata=DocumentMetadata(
                trust_score=TrustScore(
                    overall=0.9,
                    factors=[],
                ),
            ),
        )
        citations = build_citations(items, [doc])
        assert citations[0].best_trust_score == 0.9


# ────────────────────────────────────────────────────────────────────
# format_citation Tests
# ────────────────────────────────────────────────────────────────────


class TestFormatCitation:
    """Tests for citation formatting."""

    def test_format_with_precomputed_text(self) -> None:
        citation = EvidenceCitation(
            claim="Test",
            domain="industry",
            category="test",
            citation_text="[industry/test] Custom text (source: example.com)",
        )
        assert format_citation(citation) == "[industry/test] Custom text (source: example.com)"

    def test_format_with_urls(self) -> None:
        citation = EvidenceCitation(
            claim="Enterprise sales cycles are long",
            domain="industry",
            category="sales_cycle",
            source_urls=["https://example.com/about"],
            provider="industry_provider",
        )
        result = format_citation(citation)
        assert "[industry/sales_cycle]" in result
        assert "Enterprise sales cycles are long" in result
        assert "example.com" in result

    def test_format_with_provider_only(self) -> None:
        citation = EvidenceCitation(
            claim="Test claim",
            domain="industry",
            category="test",
            provider="industry_provider",
        )
        result = format_citation(citation)
        assert "(source: industry_provider)" in result

    def test_format_list(self) -> None:
        citations = [
            EvidenceCitation(
                claim="A", domain="d", category="c", citation_text="Citation A",
            ),
            EvidenceCitation(
                claim="B", domain="d", category="c", citation_text="Citation B",
            ),
        ]
        results = format_citation_list(citations)
        assert results == ["Citation A", "Citation B"]


# ────────────────────────────────────────────────────────────────────
# rank_citations_by_trust Tests
# ────────────────────────────────────────────────────────────────────


class TestRankCitationsByTrust:
    """Tests for trust-based citation ranking."""

    def test_empty_list(self) -> None:
        assert rank_citations_by_trust([]) == []

    def test_single_citation(self) -> None:
        c = EvidenceCitation(claim="A", domain="d", category="c", best_trust_score=0.5)
        assert rank_citations_by_trust([c]) == [c]

    def test_descending_order(self) -> None:
        low = EvidenceCitation(claim="L", domain="d", category="c", best_trust_score=0.2)
        mid = EvidenceCitation(claim="M", domain="d", category="c", best_trust_score=0.5)
        high = EvidenceCitation(claim="H", domain="d", category="c", best_trust_score=0.9)
        result = rank_citations_by_trust([low, high, mid])
        assert [c.claim for c in result] == ["H", "M", "L"]

    def test_stable_sort_equal_scores(self) -> None:
        a = EvidenceCitation(claim="A", domain="d", category="c", best_trust_score=0.5)
        b = EvidenceCitation(claim="B", domain="d", category="c", best_trust_score=0.5)
        result = rank_citations_by_trust([a, b])
        assert result[0].claim == "A"
        assert result[1].claim == "B"


# ────────────────────────────────────────────────────────────────────
# deduplicate_citations Tests
# ────────────────────────────────────────────────────────────────────


class TestDeduplicateCitations:
    """Tests for citation deduplication."""

    def test_empty_list(self) -> None:
        assert deduplicate_citations([]) == []

    def test_no_duplicates(self) -> None:
        a = EvidenceCitation(claim="A", domain="d1", category="c", best_trust_score=0.5)
        b = EvidenceCitation(claim="B", domain="d2", category="c", best_trust_score=0.5)
        result = deduplicate_citations([a, b])
        assert len(result) == 2

    def test_duplicates_keeps_highest_trust(self) -> None:
        low = EvidenceCitation(claim="A", domain="d", category="c", best_trust_score=0.3)
        high = EvidenceCitation(claim="A", domain="d", category="c", best_trust_score=0.8)
        result = deduplicate_citations([low, high])
        assert len(result) == 1
        assert result[0].best_trust_score == 0.8

    def test_same_claim_different_domain_not_deduped(self) -> None:
        a = EvidenceCitation(claim="A", domain="d1", category="c", best_trust_score=0.5)
        b = EvidenceCitation(claim="A", domain="d2", category="c", best_trust_score=0.5)
        result = deduplicate_citations([a, b])
        assert len(result) == 2
