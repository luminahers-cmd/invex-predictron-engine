"""Comprehensive tests for Sprint 5C extraction modules.

Covers:
  - evidence_retrieval.py — domain-specific retrieval strategies
  - evidence_confidence.py — deterministic confidence scoring
  - evidence_agreement.py — corroboration & conflict detection
  - extraction_diagnostics.py — structured diagnostic records
  - BaseExtractor evidence-aware helpers
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
    PageType,
)
from predictron_engine.evidence.provenance import TrustScore
from predictron_engine.extraction.evidence_agreement import (
    ConflictingGroup,
    CorroboratedGroup,
    compute_agreement_ratio,
    detect_conflicts,
    detect_corroboration,
    detect_single_source_claims,
)
from predictron_engine.extraction.evidence_confidence import (
    _compute_evidence_agreement_score,
    _compute_evidence_quality_score,
    _compute_evidence_trust_score,
    _compute_signal_density_score,
    _compute_source_diversity_score,
    compute_evidence_confidence,
)
from predictron_engine.extraction.evidence_retrieval import (
    BusinessModelRetrievalStrategy,
    FounderRetrievalStrategy,
    MarketRetrievalStrategy,
    ProductRetrievalStrategy,
    TechnologyRetrievalStrategy,
    _filter_by_keywords,
    _filter_evidence_by_keywords,
    _sort_by_trust,
    get_all_strategies,
    get_strategy_for_domain,
)
from predictron_engine.extraction.extraction_diagnostics import (
    ExtractionDiagnostic,
    build_diagnostic,
)
from predictron_engine.models.report import EvidenceCitation, EvidenceItem

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_doc(
    *,
    doc_id: str = "doc-1",
    text: str = "sample text",
    status: DocumentStatus = DocumentStatus.SUCCESS,
    trust_overall: float | None = 0.8,
    quality: float = 0.7,
    source_provider: str = "web_crawl",
    page_type: PageType = PageType.HOMEPAGE,
) -> EvidenceDocument:
    metadata = None
    if trust_overall is not None or quality:
        metadata = DocumentMetadata(
            document_type=DocumentType.HOMEPAGE,
            quality_score=quality,
            source_provider=source_provider,
        )
        if trust_overall is not None:
            metadata.trust_score = TrustScore(overall=trust_overall)
    return EvidenceDocument(
        id=doc_id,
        original_url="https://example.com",
        url="https://example.com",
        page_type=page_type,
        status=status,
        fetched_at=datetime.now(UTC),
        response_time_ms=100,
        http_status=200,
        title="Test Document",
        text=text,
        metadata=metadata,
    )


def _make_bundle(*docs: EvidenceDocument) -> EvidenceBundle:
    return EvidenceBundle(
        startup_name="TestCo",
        website="https://example.com",
        documents=list(docs),
    )


def _make_evidence_item(
    *,
    domain: str = "market",
    category: str = "tam",
    statement: str = "Market is large",
    source: str = "provider_a",
    relevance: float = 0.8,
    citations: list[EvidenceCitation] | None = None,
    provenance_record: str | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        domain=domain,
        category=category,
        statement=statement,
        source=source,
        relevance_score=relevance,
        citations=citations or [],
        provenance_record=provenance_record,
    )


def _make_evidence_bundle(*docs: EvidenceDocument) -> EvidenceBundle:
    return EvidenceBundle(
        startup_name="TestCo",
        website="https://example.com",
        documents=list(docs),
    )


# ============================================================================
# evidence_retrieval.py
# ============================================================================


class TestFilterByKeywords:
    def test_empty_keywords_returns_all(self):
        docs = [_make_doc(text="anything")]
        result = _filter_by_keywords(docs, [])
        assert result == docs

    def test_filters_by_matching_keyword(self):
        doc_match = _make_doc(doc_id="d1", text="Our AI platform uses machine learning")
        doc_no_match = _make_doc(doc_id="d2", text="Contact us at info@example.com")
        result = _filter_by_keywords([doc_match, doc_no_match], ["machine learning"])
        assert len(result) == 1
        assert result[0].id == "d1"

    def test_case_insensitive(self):
        doc = _make_doc(text="WE USE PYTHON AND DJANGO")
        result = _filter_by_keywords([doc], ["python"])
        assert len(result) == 1


class TestFilterEvidenceByKeywords:
    def test_empty_keywords_returns_all(self):
        items = [_make_evidence_item()]
        result = _filter_evidence_by_keywords(items, [])
        assert result == items

    def test_filters_by_statement(self):
        item_match = _make_evidence_item(statement="Market is growing rapidly")
        item_no_match = _make_evidence_item(statement="Team has strong experience")
        result = _filter_evidence_by_keywords([item_match, item_no_match], ["growing"])
        assert len(result) == 1
        assert result[0].statement == "Market is growing rapidly"


class TestSortByTrust:
    def test_sorts_highest_trust_first(self):
        low_trust = _make_doc(doc_id="low", trust_overall=0.3)
        high_trust = _make_doc(doc_id="high", trust_overall=0.9)
        result = _sort_by_trust([low_trust, high_trust])
        assert result[0].id == "high"
        assert result[1].id == "low"

    def test_handles_no_metadata(self):
        no_meta = _make_doc(doc_id="no_meta", trust_overall=None)
        has_trust = _make_doc(doc_id="has_trust", trust_overall=0.7)
        result = _sort_by_trust([no_meta, has_trust])
        assert result[0].id == "has_trust"


class TestMarketRetrievalStrategy:
    def test_get_keywords_returns_list(self):
        strategy = MarketRetrievalStrategy()
        keywords = strategy.get_keywords()
        assert isinstance(keywords, list)
        assert "market" in keywords
        assert "competitor" in keywords

    def test_get_document_types(self):
        strategy = MarketRetrievalStrategy()
        types = strategy.get_document_types()
        assert "homepage" in types
        assert "blog" in types

    def test_retrieve_documents_filters_and_sorts(self):
        doc1 = _make_doc(doc_id="d1", text="This market analysis shows growth", trust_overall=0.9)
        doc2 = _make_doc(doc_id="d2", text="Contact page info", trust_overall=0.3)
        failed = _make_doc(
            doc_id="d3", text="This failed",
            status=DocumentStatus.FAILED, trust_overall=0.5,
        )
        bundle = _make_evidence_bundle(doc1, doc2, failed)
        strategy = MarketRetrievalStrategy()
        result = strategy.retrieve_documents(bundle)
        # Failed docs should be excluded
        assert all(d.status == DocumentStatus.SUCCESS for d in result)
        # Market-related doc should be prioritized (it has keyword "market")
        assert result[0].id == "d1"

    def test_retrieve_evidence_items_prioritizes_market(self):
        item_market = _make_evidence_item(domain="market", relevance=0.9)
        item_other = _make_evidence_item(domain="founder", relevance=0.6)
        strategy = MarketRetrievalStrategy()
        result = strategy.retrieve_evidence_items([item_market, item_other])
        # Market items come first in the combined list, sorted by relevance
        assert result[0].domain == "market"


class TestFounderRetrievalStrategy:
    def test_keywords_include_founder_terms(self):
        strategy = FounderRetrievalStrategy()
        keywords = strategy.get_keywords()
        assert "founder" in keywords
        assert "team" in keywords
        assert "ceo" in keywords

    def test_retrieve_documents_filters(self):
        doc = _make_doc(doc_id="d1", text="The founder has 15 years experience", trust_overall=0.8)
        bundle = _make_evidence_bundle(doc)
        strategy = FounderRetrievalStrategy()
        result = strategy.retrieve_documents(bundle)
        assert len(result) == 1


class TestTechnologyRetrievalStrategy:
    def test_keywords_include_tech_terms(self):
        strategy = TechnologyRetrievalStrategy()
        keywords = strategy.get_keywords()
        assert "python" in keywords
        assert "aws" in keywords
        assert "api" in keywords


class TestBusinessModelRetrievalStrategy:
    def test_keywords_include_bm_terms(self):
        strategy = BusinessModelRetrievalStrategy()
        keywords = strategy.get_keywords()
        assert "saas" in keywords
        assert "subscription" in keywords
        assert "pricing" in keywords


class TestProductRetrievalStrategy:
    def test_keywords_include_product_terms(self):
        strategy = ProductRetrievalStrategy()
        keywords = strategy.get_keywords()
        assert "product" in keywords
        assert "platform" in keywords
        assert "api" in keywords


class TestGetStrategyForDomain:
    def test_known_domains_return_strategy(self):
        assert isinstance(get_strategy_for_domain("market"), MarketRetrievalStrategy)
        assert isinstance(get_strategy_for_domain("founder"), FounderRetrievalStrategy)
        assert isinstance(get_strategy_for_domain("technology"), TechnologyRetrievalStrategy)
        assert isinstance(get_strategy_for_domain("business_model"), BusinessModelRetrievalStrategy)
        assert isinstance(get_strategy_for_domain("product"), ProductRetrievalStrategy)

    def test_unknown_domain_returns_none(self):
        assert get_strategy_for_domain("unknown_domain") is None

    def test_get_all_strategies_returns_five(self):
        all_strats = get_all_strategies()
        assert len(all_strats) == 5
        assert set(all_strats.keys()) == {
            "market", "founder", "technology", "business_model", "product"
        }


# ============================================================================
# evidence_confidence.py
# ============================================================================


class TestEvidenceQualityScore:
    def test_empty_docs(self):
        assert _compute_evidence_quality_score([]) == 0.0

    def test_all_with_trust_score(self):
        docs = [_make_doc(trust_overall=0.8), _make_doc(trust_overall=0.6)]
        assert _compute_evidence_quality_score(docs) == 1.0

    def test_none_without_trust_score(self):
        docs = [_make_doc(trust_overall=None), _make_doc(trust_overall=None)]
        assert _compute_evidence_quality_score(docs) == 0.0

    def test_mixed(self):
        docs = [_make_doc(trust_overall=0.8), _make_doc(trust_overall=None)]
        assert _compute_evidence_quality_score(docs) == 0.5


class TestEvidenceTrustScore:
    def test_empty_docs(self):
        assert _compute_evidence_trust_score([]) == 0.0

    def test_average_trust(self):
        docs = [_make_doc(trust_overall=0.8), _make_doc(trust_overall=0.6)]
        result = _compute_evidence_trust_score(docs)
        assert abs(result - 0.7) < 1e-6

    def test_no_trust_scores(self):
        docs = [_make_doc(trust_overall=None)]
        assert _compute_evidence_trust_score(docs) == 0.0


class TestEvidenceAgreementScore:
    def test_empty_items(self):
        assert _compute_evidence_agreement_score([], []) == 0.0

    def test_item_with_provenance(self):
        item = _make_evidence_item(provenance_record="some_record")
        assert _compute_evidence_agreement_score([item], []) == 1.0

    def test_item_with_citations(self):
        citation = EvidenceCitation(
            claim="test", domain="market", category="tam",
            source_document_ids=["doc-1"],
        )
        item = _make_evidence_item(citations=[citation])
        assert _compute_evidence_agreement_score([item], []) == 1.0

    def test_item_without_corroboration(self):
        item = _make_evidence_item()
        assert _compute_evidence_agreement_score([item], []) == 0.0

    def test_mixed_corroboration(self):
        item_corroborated = _make_evidence_item(provenance_record="record")
        item_bare = _make_evidence_item(statement="bare claim")
        result = _compute_evidence_agreement_score([item_corroborated, item_bare], [])
        assert abs(result - 0.5) < 1e-6


class TestSourceDiversityScore:
    def test_empty_docs(self):
        assert _compute_source_diversity_score([]) == 0.0

    def test_single_provider(self):
        docs = [_make_doc(source_provider="web_crawl")]
        assert _compute_source_diversity_score(docs) == pytest.approx(1.0 / 3.0)

    def test_three_providers_saturates(self):
        docs = [
            _make_doc(source_provider="provider_a"),
            _make_doc(source_provider="provider_b"),
            _make_doc(source_provider="provider_c"),
        ]
        assert _compute_source_diversity_score(docs) == 1.0

    def test_no_provider_metadata(self):
        doc = _make_doc()
        doc.metadata.source_provider = ""
        assert _compute_source_diversity_score([doc]) == 0.0


class TestSignalDensityScore:
    def test_zero_keywords(self):
        assert _compute_signal_density_score(0, 0, 100) == 0.0

    def test_full_coverage(self):
        result = _compute_signal_density_score(10, 10, 1000)
        # 10/10 = 1.0 * 0.7 + min(1000/500, 0.5)*0.3 = 0.7 + 0.15 = 0.85
        assert abs(result - 0.85) < 1e-4

    def test_no_description(self):
        result = _compute_signal_density_score(5, 10, 0)
        # 0.5 * 0.7 + 0 * 0.3 = 0.35
        assert abs(result - 0.35) < 1e-4


class TestComputeEvidenceConfidence:
    def test_returns_float_in_range(self):
        result = compute_evidence_confidence(
            evidence_items=[],
            documents=[_make_doc()],
            keywords_matched=5,
            total_keywords=10,
            description_length=500,
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_high_quality_inputs_gives_higher_score(self):
        docs = [
            _make_doc(trust_overall=0.9, source_provider="provider_a"),
            _make_doc(trust_overall=0.85, source_provider="provider_b"),
        ]
        item = _make_evidence_item(
            provenance_record="record",
            domain="market",
            category="tam",
        )
        high = compute_evidence_confidence(
            evidence_items=[item],
            documents=docs,
            keywords_matched=8,
            total_keywords=10,
            description_length=1000,
        )
        low = compute_evidence_confidence(
            evidence_items=[],
            documents=[_make_doc(trust_overall=None, source_provider="")],
            keywords_matched=1,
            total_keywords=10,
            description_length=100,
        )
        assert high > low

    def test_empty_everything_returns_zero(self):
        result = compute_evidence_confidence(
            evidence_items=[],
            documents=[],
            keywords_matched=0,
            total_keywords=0,
            description_length=0,
        )
        assert result == 0.0


# ============================================================================
# evidence_agreement.py
# ============================================================================


class TestDetectCorroboration:
    def test_no_groups_when_single_items(self):
        items = [_make_evidence_item(domain="market", category="tam")]
        result = detect_corroboration(items)
        assert result == []

    def test_corroboration_with_multiple_sources(self):
        item_a = _make_evidence_item(
            domain="market", category="tam", source="provider_a",
            statement="TAM is $10B",
        )
        item_b = _make_evidence_item(
            domain="market", category="tam", source="provider_b",
            statement="TAM estimated at $12B",
        )
        result = detect_corroboration([item_a, item_b])
        assert len(result) == 1
        group = result[0]
        assert isinstance(group, CorroboratedGroup)
        assert group.domain == "market"
        assert group.category == "tam"
        assert group.source_count == 2

    def test_sorted_by_source_count_desc(self):
        items_a = [
            _make_evidence_item(domain="market", category="tam", source=f"p{i}")
            for i in range(3)
        ]
        items_b = [
            _make_evidence_item(domain="founder", category="team", source=f"p{i}")
            for i in range(2)
        ]
        result = detect_corroboration(items_a + items_b)
        assert result[0].source_count >= result[-1].source_count


class TestDetectConflicts:
    def test_no_conflict_when_single_item(self):
        items = [_make_evidence_item(domain="market", category="tam")]
        result = detect_conflicts(items)
        assert result == []

    def test_no_conflict_when_statements_agree(self):
        item_a = _make_evidence_item(
            domain="market", category="tam", statement="Market is growing"
        )
        item_b = _make_evidence_item(
            domain="market", category="tam", statement="Market shows strong growth"
        )
        result = detect_conflicts([item_a, item_b])
        assert result == []

    def test_conflict_detected(self):
        item_a = _make_evidence_item(
            domain="business_model", category="revenue",
            statement="Recurring revenue model with high growth",
        )
        item_b = _make_evidence_item(
            domain="business_model", category="revenue",
            statement="Transaction-based model with low margins",
        )
        result = detect_conflicts([item_a, item_b])
        assert len(result) >= 1
        assert isinstance(result[0], ConflictingGroup)

    def test_deduplicates_conflicts(self):
        item_a = _make_evidence_item(
            domain="market", category="tam",
            statement="The enterprise segment is high growth",
        )
        item_b = _make_evidence_item(
            domain="market", category="tam",
            statement="Consumer market is low risk",
        )
        item_c = _make_evidence_item(
            domain="market", category="tam",
            statement="Consumer segment is low cost",
        )
        result = detect_conflicts([item_a, item_b, item_c])
        # Should be deduplicated
        seen = set()
        for c in result:
            key = (c.domain, c.item_a.statement[:50], c.item_b.statement[:50])
            assert key not in seen
            seen.add(key)


class TestDetectSingleSourceClaims:
    def test_no_citations_no_provenance(self):
        item = _make_evidence_item()
        result = detect_single_source_claims([item])
        assert len(result) == 1

    def test_multiple_citations_not_single_source(self):
        citations = [
            EvidenceCitation(
                claim="test", domain="m", category="c",
                source_document_ids=["d1"],
            ),
            EvidenceCitation(
                claim="test", domain="m", category="c",
                source_document_ids=["d2"],
            ),
        ]
        item = _make_evidence_item(citations=citations)
        result = detect_single_source_claims([item])
        assert len(result) == 0

    def test_single_citation_is_single_source(self):
        citation = EvidenceCitation(
            claim="test", domain="m", category="c",
            source_document_ids=["d1"],
        )
        item = _make_evidence_item(citations=[citation])
        result = detect_single_source_claims([item])
        assert len(result) == 1


class TestComputeAgreementRatio:
    def test_empty_returns_one(self):
        assert compute_agreement_ratio([]) == 1.0

    def test_all_corroborated(self):
        items = [
            _make_evidence_item(domain="m", category="c", source="p1", statement="a"),
            _make_evidence_item(domain="m", category="c", source="p2", statement="b"),
        ]
        result = compute_agreement_ratio(items)
        assert result == 1.0

    def test_no_corroboration(self):
        items = [
            _make_evidence_item(domain="m", category="c1", source="p1", statement="a"),
            _make_evidence_item(domain="m", category="c2", source="p1", statement="b"),
        ]
        result = compute_agreement_ratio(items)
        assert result == 0.0

    def test_partial_corroboration(self):
        items = [
            _make_evidence_item(domain="m", category="c", source="p1", statement="a"),
            _make_evidence_item(domain="m", category="c", source="p2", statement="b"),
            _make_evidence_item(domain="f", category="c", source="p3", statement="c"),
        ]
        result = compute_agreement_ratio(items)
        # 2 out of 3 items corroborated (market category has 2 sources)
        assert abs(result - 2.0 / 3.0) < 1e-4


# ============================================================================
# extraction_diagnostics.py
# ============================================================================


class TestBuildDiagnostic:
    def test_basic_construction(self):
        diag = build_diagnostic(
            extractor_name="MarketExtractor",
            documents_examined=[_make_doc(doc_id="d1"), _make_doc(doc_id="d2")],
            documents_selected=[_make_doc(doc_id="d1", trust_overall=0.8)],
            citation_count=3,
            conflict_count=1,
            corroborated_count=2,
            single_source_count=1,
            evidence_confidence=0.75,
        )
        assert isinstance(diag, ExtractionDiagnostic)
        assert diag.extractor_name == "MarketExtractor"
        assert diag.documents_examined == 2
        assert diag.documents_selected == 1
        assert diag.citation_count == 3
        assert diag.conflict_count == 1
        assert diag.corroborated_count == 2
        assert diag.single_source_count == 1
        assert diag.evidence_confidence == 0.75

    def test_average_trust_computed(self):
        doc1 = _make_doc(trust_overall=0.6)
        doc2 = _make_doc(trust_overall=0.8)
        diag = build_diagnostic(
            extractor_name="TestExtractor",
            documents_examined=[doc1, doc2],
            documents_selected=[doc1, doc2],
            citation_count=0,
            conflict_count=0,
            corroborated_count=0,
            single_source_count=0,
            evidence_confidence=0.5,
        )
        assert abs(diag.average_trust_score - 0.7) < 1e-4

    def test_selected_document_ids(self):
        doc = _make_doc(doc_id="unique-id-123")
        diag = build_diagnostic(
            extractor_name="TestExtractor",
            documents_examined=[doc],
            documents_selected=[doc],
            citation_count=0,
            conflict_count=0,
            corroborated_count=0,
            single_source_count=0,
            evidence_confidence=0.0,
        )
        assert "unique-id-123" in diag.selected_document_ids

    def test_keywords_fields(self):
        diag = build_diagnostic(
            extractor_name="TestExtractor",
            documents_examined=[],
            documents_selected=[],
            citation_count=0,
            conflict_count=0,
            corroborated_count=0,
            single_source_count=0,
            evidence_confidence=0.0,
            keywords_matched=5,
            total_keywords=10,
        )
        assert diag.keywords_matched == 5
        assert diag.total_keywords == 10

    def test_empty_selected_documents(self):
        diag = build_diagnostic(
            extractor_name="TestExtractor",
            documents_examined=[_make_doc()],
            documents_selected=[],
            citation_count=0,
            conflict_count=0,
            corroborated_count=0,
            single_source_count=0,
            evidence_confidence=0.0,
        )
        assert diag.average_trust_score == 0.0
        assert diag.selected_document_ids == []


# ============================================================================
# BaseExtractor evidence helpers
# ============================================================================


class TestBaseExtractorHelpers:
    def test_count_keyword_matches(self):
        from predictron_engine.extraction.extractors.base import BaseExtractor
        ext = BaseExtractor()
        # keyword matching is case-sensitive substring check
        assert ext._count_keyword_matches("the ai market is growing", ["ai", "blockchain"]) == 1
        assert ext._count_keyword_matches("ai and ml and cloud", ["ai", "ml", "cloud"]) == 3

    def test_combined_text_no_evidence(self):
        from predictron_engine.extraction.extractors.base import BaseExtractor
        ext = BaseExtractor()
        assert ext._combined_text("hello", None) == "hello"

    def test_combined_text_with_evidence(self):
        from predictron_engine.extraction.extractors.base import BaseExtractor
        ext = BaseExtractor()
        doc = _make_doc(text="evidence text here")
        bundle = _make_evidence_bundle(doc)
        result = ext._combined_text("description", bundle)
        assert "description" in result
        assert "evidence text here" in result

    def test_combined_text_from_docs(self):
        from predictron_engine.extraction.extractors.base import BaseExtractor
        ext = BaseExtractor()
        doc = _make_doc(text="filtered doc text")
        result = ext._combined_text_from_docs("my startup", [doc])
        assert "my startup" in result
        assert "filtered doc text" in result

    def test_combined_text_from_docs_empty(self):
        from predictron_engine.extraction.extractors.base import BaseExtractor
        ext = BaseExtractor()
        assert ext._combined_text_from_docs("description", []) == "description"

    def test_build_trust_summary(self):
        from predictron_engine.extraction.extractors.base import BaseExtractor
        ext = BaseExtractor()
        high = _make_doc(trust_overall=0.8)
        low = _make_doc(trust_overall=0.2)
        result = ext._build_trust_summary([high, low])
        assert "high-trust" in result
        assert "low-trust" in result

    def test_build_trust_summary_empty(self):
        from predictron_engine.extraction.extractors.base import BaseExtractor
        ext = BaseExtractor()
        assert ext._build_trust_summary([]) == "no evidence documents"

    def test_populate_evidence_provenance(self):
        from predictron_engine.extraction.extractors.base import BaseExtractor
        from predictron_engine.models.extracted_features import ExtractedFeatures
        ext = BaseExtractor()
        features = ExtractedFeatures()
        doc = _make_doc(doc_id="prov-doc-1")
        item = _make_evidence_item(
            source="test_provider",
            domain="market",
            category="tam",
            statement="Test statement",
        )
        ext._populate_evidence_provenance(
            features,
            documents=[doc],
            evidence_items=[item],
            citations=[],
            evidence_confidence=0.75,
            agreement_ratio=0.5,
            conflict_count=1,
        )
        assert features.evidence_confidence == 0.75
        assert features.evidence_agreement_ratio == 0.5
        assert features.evidence_conflict_count == 1
        assert "test_provider" in features.evidence_sources_used
        assert features.evidence_document_count == 1
        assert "prov-doc-1" in features.provenance_document_ids
