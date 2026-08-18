"""Tests for the Document Intelligence layer (Sprint 4C).

Covers document classification, quality analysis, duplicate detection,
authority estimation, metadata enrichment, and diagnostics.  Every test
is pure-deterministic — identical inputs always produce identical outputs.
No network, no LLMs, all inputs are mocked.
"""

from __future__ import annotations

from datetime import UTC, datetime

from predictron_engine.evidence.document_intelligence import (
    DocumentIntelligence,
    QualityMetrics,
    classify_document,
    compute_content_hash,
    compute_quality_metrics,
    detect_duplicates,
    enrich_documents,
    estimate_authority,
)
from predictron_engine.evidence.models import (
    DocumentMetadata,
    DocumentStatus,
    DocumentType,
    EvidenceDocument,
    make_document_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TIME = datetime(2025, 1, 1, tzinfo=UTC)


def _doc(
    url: str = "https://example.com",
    *,
    title: str = "",
    text: str = "Hello world content.",
    status: DocumentStatus = DocumentStatus.SUCCESS,
    page_type: str = "unknown",
) -> EvidenceDocument:
    """Build an EvidenceDocument from minimal inputs."""
    return EvidenceDocument(
        id=make_document_id(url),
        original_url=url,
        url=url,
        page_type=page_type,
        status=status,
        fetched_at=_TIME,
        response_time_ms=100,
        http_status=200,
        title=title,
        text=text,
    )


def _success_doc(url: str = "https://example.com", **kw: object) -> EvidenceDocument:
    return _doc(url, status=DocumentStatus.SUCCESS, **kw)


def _empty_doc(url: str = "https://example.com") -> EvidenceDocument:
    return _doc(url, status=DocumentStatus.EMPTY, text="")


def _failed_doc(url: str = "https://example.com") -> EvidenceDocument:
    return _doc(url, status=DocumentStatus.FAILED, text="Error occurred")


# ===================================================================
# Part 1 — Document Classification
# ===================================================================


class TestDocumentType:
    def test_all_types_are_strings(self) -> None:
        for dtype in DocumentType:
            assert isinstance(dtype, str)

    def test_unique_values(self) -> None:
        values = [dt.value for dt in DocumentType]
        assert len(values) == len(set(values))

    def test_has_expected_types(self) -> None:
        expected = {
            "homepage", "about", "product", "pricing", "documentation",
            "api_docs", "blog", "careers", "security", "privacy", "terms",
            "faq", "contact", "news", "press_release", "investor",
            "repository", "unknown",
        }
        assert {dt.value for dt in DocumentType} == expected


class TestClassifyDocument:
    def test_homepage_root(self) -> None:
        assert classify_document("https://example.com") == DocumentType.HOMEPAGE

    def test_homepage_root_slash(self) -> None:
        assert classify_document("https://example.com/") == DocumentType.HOMEPAGE

    def test_about_page(self) -> None:
        assert classify_document("https://example.com/about") == DocumentType.ABOUT

    def test_about_company(self) -> None:
        assert classify_document("https://example.com/company") == DocumentType.ABOUT

    def test_product_page(self) -> None:
        assert classify_document("https://example.com/product") == DocumentType.PRODUCT

    def test_products_page(self) -> None:
        assert classify_document("https://example.com/products") == DocumentType.PRODUCT

    def test_features_page(self) -> None:
        assert classify_document("https://example.com/features") == DocumentType.PRODUCT

    def test_pricing_page(self) -> None:
        assert classify_document("https://example.com/pricing") == DocumentType.PRICING

    def test_plans_page(self) -> None:
        assert classify_document("https://example.com/plans") == DocumentType.PRICING

    def test_docs_page(self) -> None:
        assert classify_document("https://example.com/docs") == DocumentType.DOCUMENTATION

    def test_documentation_page(self) -> None:
        assert classify_document("https://example.com/documentation") == DocumentType.DOCUMENTATION

    def test_guide_page(self) -> None:
        assert classify_document("https://example.com/guide/intro") == DocumentType.DOCUMENTATION

    def test_api_docs(self) -> None:
        assert classify_document("https://example.com/api/docs") == DocumentType.API_DOCS

    def test_api_reference(self) -> None:
        assert classify_document("https://example.com/api-reference") == DocumentType.API_DOCS

    def test_developer_page(self) -> None:
        assert classify_document("https://example.com/developer") == DocumentType.API_DOCS

    def test_blog_page(self) -> None:
        assert classify_document("https://example.com/blog/post-1") == DocumentType.BLOG

    def test_careers_page(self) -> None:
        assert classify_document("https://example.com/careers") == DocumentType.CAREERS

    def test_jobs_page(self) -> None:
        assert classify_document("https://example.com/jobs") == DocumentType.CAREERS

    def test_security_page(self) -> None:
        assert classify_document("https://example.com/security") == DocumentType.SECURITY

    def test_privacy_page(self) -> None:
        assert classify_document("https://example.com/privacy") == DocumentType.PRIVACY

    def test_privacy_policy(self) -> None:
        assert classify_document("https://example.com/privacy-policy") == DocumentType.PRIVACY

    def test_terms_page(self) -> None:
        assert classify_document("https://example.com/terms") == DocumentType.TERMS

    def test_terms_of_service(self) -> None:
        assert classify_document("https://example.com/terms-of-service") == DocumentType.TERMS

    def test_faq_page(self) -> None:
        assert classify_document("https://example.com/faq") == DocumentType.FAQ

    def test_help_page(self) -> None:
        assert classify_document("https://example.com/help") == DocumentType.FAQ

    def test_contact_page(self) -> None:
        assert classify_document("https://example.com/contact") == DocumentType.CONTACT

    def test_news_page(self) -> None:
        assert classify_document("https://example.com/news/latest") == DocumentType.NEWS

    def test_press_page(self) -> None:
        assert classify_document("https://example.com/press") == DocumentType.PRESS_RELEASE

    def test_investor_page(self) -> None:
        assert classify_document("https://example.com/investor") == DocumentType.INVESTOR

    def test_investors_page(self) -> None:
        assert classify_document("https://example.com/investors") == DocumentType.INVESTOR

    def test_github_repository(self) -> None:
        assert classify_document("https://github.com/org/repo") == DocumentType.REPOSITORY

    def test_gitlab_repository(self) -> None:
        assert classify_document("https://gitlab.com/org/repo") == DocumentType.REPOSITORY

    def test_deep_unknown_path(self) -> None:
        assert classify_document("https://example.com/some/random/path") == DocumentType.UNKNOWN

    def test_title_signals_pricing(self) -> None:
        result = classify_document(
            "https://example.com/page",
            title="Our Pricing Plans",
        )
        assert result == DocumentType.PRICING

    def test_title_signals_about(self) -> None:
        result = classify_document(
            "https://example.com/page",
            title="About Us - Company Info",
        )
        assert result == DocumentType.ABOUT

    def test_title_signals_blog(self) -> None:
        result = classify_document(
            "https://example.com/page",
            title="Latest Blog Post",
        )
        assert result == DocumentType.BLOG

    def test_heading_signals(self) -> None:
        result = classify_document(
            "https://example.com/unclear",
            headings=["Pricing", "Features", "Compare Plans"],
        )
        assert result == DocumentType.PRICING

    def test_path_beats_title(self) -> None:
        """Path signals should take priority over title signals."""
        result = classify_document(
            "https://example.com/about",
            title="Our Pricing Guide",
        )
        assert result == DocumentType.ABOUT

    def test_deterministic(self) -> None:
        r1 = classify_document("https://example.com/about")
        r2 = classify_document("https://example.com/about")
        assert r1 == r2

    def test_empty_url(self) -> None:
        assert classify_document("") == DocumentType.UNKNOWN


# ===================================================================
# Part 2 — DocumentMetadata
# ===================================================================


class TestDocumentMetadata:
    def test_defaults(self) -> None:
        m = DocumentMetadata()
        assert m.document_type == DocumentType.UNKNOWN
        assert m.authority_score == 0.0
        assert m.quality_score == 0.0
        assert m.priority == 6
        assert m.language == "en"
        assert m.is_duplicate is False
        assert m.trust_level == "unknown"

    def test_all_fields_populated(self) -> None:
        m = DocumentMetadata(
            document_type=DocumentType.HOMEPAGE,
            authority_score=0.9,
            quality_score=0.85,
            priority=1,
            canonical_url="https://example.com",
            language="en",
            word_count=500,
            heading_count=5,
            table_count=2,
            list_count=3,
            content_hash="abc123",
            duplicate_of=None,
            is_duplicate=False,
            trust_level="official",
            source_provider="website",
        )
        assert m.document_type == DocumentType.HOMEPAGE
        assert m.authority_score == 0.9

    def test_backward_compatible(self) -> None:
        """Adding metadata to EvidenceDocument should be backward-compatible."""
        doc = _doc()
        assert doc.metadata is None

    def test_with_metadata(self) -> None:
        doc = _doc()
        meta = DocumentMetadata(document_type=DocumentType.ABOUT)
        enriched = doc.model_copy(update={"metadata": meta})
        assert enriched.metadata is not None
        assert enriched.metadata.document_type == DocumentType.ABOUT
        assert enriched.title == doc.title


# ===================================================================
# Part 3 — Quality Analysis
# ===================================================================


class TestQualityMetrics:
    def test_defaults(self) -> None:
        m = QualityMetrics()
        assert m.text_density == 0.0
        assert m.navigation_ratio == 0.0
        assert m.boilerplate_ratio == 0.0
        assert m.heading_quality == 0.0
        assert m.duplicate_heading_ratio == 0.0
        assert m.content_completeness == 0.0


class TestComputeQualityMetrics:
    def test_empty_document(self) -> None:
        doc = _doc(text="")
        q = compute_quality_metrics(doc)
        assert q.text_density == 0.0
        assert q.content_completeness == 0.0

    def test_content_document(self) -> None:
        doc = _doc(
            text="This is a detailed product description with multiple "
            "paragraphs explaining the features and benefits of our "
            "enterprise software platform for business automation.",
        )
        q = compute_quality_metrics(
            doc,
            headings=["Introduction", "Features", "Benefits"],
            paragraphs=["Para 1", "Para 2", "Para 3"],
        )
        assert q.text_density > 0.0
        assert q.heading_quality > 0.0
        assert q.content_completeness > 0.0

    def test_navigation_heavy(self) -> None:
        doc = _doc(
            text="Home menu skip to content breadcrumb login sign up "
            "register subscribe newsletter back to top",
        )
        q = compute_quality_metrics(doc)
        assert q.navigation_ratio > 0.0

    def test_boilerplate_heavy(self) -> None:
        doc = _doc(
            text="Copyright 2024 all rights reserved cookie privacy "
            "terms of service powered by built with subscribe newsletter",
        )
        q = compute_quality_metrics(doc)
        assert q.boilerplate_ratio > 0.0

    def test_heading_quality(self) -> None:
        doc = _doc(text="Some content text here.")
        q_good = compute_quality_metrics(
            doc, headings=["Introduction", "Features", "Pricing", "Contact"],
        )
        q_bad = compute_quality_metrics(doc, headings=[])
        assert q_good.heading_quality > q_bad.heading_quality

    def test_duplicate_headings(self) -> None:
        doc = _doc(text="Some content text here.")
        q = compute_quality_metrics(
            doc, headings=["Features", "Features", "About"],
        )
        assert q.duplicate_heading_ratio > 0.0

    def test_content_completeness(self) -> None:
        doc = _doc(text=" ".join(["word"] * 500))
        q_full = compute_quality_metrics(
            doc,
            headings=["H1", "H2"],
            paragraphs=["p1", "p2", "p3"],
        )
        q_empty = compute_quality_metrics(doc)
        assert q_full.content_completeness > q_empty.content_completeness

    def test_deterministic(self) -> None:
        doc = _doc(text="Test content for quality analysis check.")
        q1 = compute_quality_metrics(doc, headings=["Test"])
        q2 = compute_quality_metrics(doc, headings=["Test"])
        assert q1 == q2


class TestComputeContentHash:
    def test_identical_text(self) -> None:
        h1 = compute_content_hash("Hello world.")
        h2 = compute_content_hash("Hello world.")
        assert h1 == h2

    def test_different_text(self) -> None:
        h1 = compute_content_hash("Hello world.")
        h2 = compute_content_hash("Goodbye world.")
        assert h1 != h2

    def test_case_insensitive(self) -> None:
        h1 = compute_content_hash("Hello World")
        h2 = compute_content_hash("hello world")
        assert h1 == h2

    def test_whitespace_insensitive(self) -> None:
        h1 = compute_content_hash("Hello  world")
        h2 = compute_content_hash("Hello world")
        assert h1 == h2

    def test_empty_string(self) -> None:
        h = compute_content_hash("")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest

    def test_deterministic(self) -> None:
        h1 = compute_content_hash("test content")
        h2 = compute_content_hash("test content")
        assert h1 == h2


# ===================================================================
# Part 4 — Duplicate Detection
# ===================================================================


class TestDetectDuplicates:
    def test_no_duplicates(self) -> None:
        docs = [
            _success_doc("https://a.com/page1", text="Unique content for page 1."),
            _success_doc("https://a.com/page2", text="Different content for page 2."),
            _success_doc("https://a.com/page3", text="Another unique page 3."),
        ]
        result, groups = detect_duplicates(docs)
        assert len(result) == 3
        assert len(groups) == 0

    def test_identical_content(self) -> None:
        docs = [
            _success_doc("https://a.com/page1", text="Identical content here."),
            _success_doc("https://a.com/page2", text="Identical content here."),
        ]
        result, groups = detect_duplicates(docs)
        assert len(result) == 1
        assert len(groups) == 1
        assert groups[0].reason == "identical content hash"

    def test_canonical_is_shortest_url(self) -> None:
        docs = [
            _success_doc("https://a.com/about/team", text="About us."),
            _success_doc("https://a.com/about", text="About us."),
        ]
        result, groups = detect_duplicates(docs)
        assert len(result) == 1
        assert str(result[0].url) == "https://a.com/about"

    def test_failed_docs_preserved(self) -> None:
        docs = [
            _success_doc("https://a.com/page1", text="Content."),
            _failed_doc("https://a.com/page2"),
        ]
        result, groups = detect_duplicates(docs)
        assert len(result) == 2  # both kept

    def test_empty_input(self) -> None:
        result, groups = detect_duplicates([])
        assert result == []
        assert groups == []

    def test_single_document(self) -> None:
        docs = [_success_doc("https://a.com/page")]
        result, groups = detect_duplicates(docs)
        assert len(result) == 1
        assert len(groups) == 0

    def test_deterministic(self) -> None:
        docs = [
            _success_doc("https://a.com/p1", text="Same content."),
            _success_doc("https://a.com/p2", text="Same content."),
        ]
        r1, g1 = detect_duplicates(docs)
        r2, g2 = detect_duplicates(docs)
        assert len(r1) == len(r2)
        assert len(g1) == len(g2)

    def test_multiple_groups(self) -> None:
        docs = [
            _success_doc("https://a.com/page1", text="Group A content."),
            _success_doc("https://a.com/page2", text="Group A content."),
            _success_doc("https://a.com/page3", text="Group B content."),
            _success_doc("https://a.com/page4", text="Group B content."),
        ]
        result, groups = detect_duplicates(docs)
        assert len(result) == 2
        assert len(groups) == 2


# ===================================================================
# Part 5 — Authority Estimation
# ===================================================================


class TestEstimateAuthority:
    def test_homepage_high_authority(self) -> None:
        doc = _success_doc("https://example.com/")
        auth = estimate_authority(
            doc,
            document_type=DocumentType.HOMEPAGE,
        )
        assert auth > 0.7

    def test_documentation_high_authority(self) -> None:
        doc = _success_doc("https://example.com/docs")
        auth = estimate_authority(
            doc,
            document_type=DocumentType.DOCUMENTATION,
        )
        assert auth > 0.7

    def test_unknown_low_authority(self) -> None:
        doc = _success_doc("https://random-site.com/page")
        auth = estimate_authority(
            doc,
            document_type=DocumentType.UNKNOWN,
        )
        assert auth < 0.6

    def test_official_domain_bonus(self) -> None:
        doc = _success_doc("https://example.com/about")
        auth_with = estimate_authority(
            doc,
            document_type=DocumentType.ABOUT,
            official_host="example.com",
        )
        auth_without = estimate_authority(
            doc,
            document_type=DocumentType.ABOUT,
            official_host=None,
        )
        assert auth_with > auth_without

    def test_subdomain_bonus(self) -> None:
        doc = _success_doc("https://docs.example.com/api")
        auth = estimate_authority(
            doc,
            document_type=DocumentType.API_DOCS,
            official_host="example.com",
        )
        assert auth > 0.7

    def test_third_party_bonus(self) -> None:
        doc = _success_doc("https://github.com/org/repo")
        auth = estimate_authority(
            doc,
            document_type=DocumentType.REPOSITORY,
        )
        assert auth > 0.6

    def test_quality_bonus(self) -> None:
        doc = _success_doc("https://example.com/page")
        auth_high = estimate_authority(
            doc, document_type=DocumentType.UNKNOWN, quality_score=0.9,
        )
        auth_low = estimate_authority(
            doc, document_type=DocumentType.UNKNOWN, quality_score=0.1,
        )
        assert auth_high > auth_low

    def test_success_bonus(self) -> None:
        doc_ok = _success_doc("https://example.com/page")
        doc_fail = _doc("https://example.com/page", status=DocumentStatus.FAILED)
        auth_ok = estimate_authority(doc_ok, document_type=DocumentType.UNKNOWN)
        auth_fail = estimate_authority(doc_fail, document_type=DocumentType.UNKNOWN)
        assert auth_ok > auth_fail

    def test_score_capped_at_1(self) -> None:
        doc = _success_doc("https://github.com/org/repo")
        auth = estimate_authority(
            doc,
            document_type=DocumentType.HOMEPAGE,
            official_host="github.com",
            quality_score=1.0,
        )
        assert auth <= 1.0

    def test_score_floored_at_0(self) -> None:
        doc = _success_doc("https://example.com")
        auth = estimate_authority(doc, document_type=DocumentType.UNKNOWN)
        assert auth >= 0.0

    def test_deterministic(self) -> None:
        doc = _success_doc("https://example.com/")
        a1 = estimate_authority(doc, document_type=DocumentType.HOMEPAGE)
        a2 = estimate_authority(doc, document_type=DocumentType.HOMEPAGE)
        assert a1 == a2


# ===================================================================
# Part 6 — Integration (enrich_documents)
# ===================================================================


class TestEnrichDocuments:
    def test_empty_input(self) -> None:
        result, summary = enrich_documents([])
        assert result == []
        assert summary.documents_input == 0

    def test_single_document(self) -> None:
        docs = [_success_doc("https://example.com/", text="Welcome to our site.")]
        result, summary = enrich_documents(docs)
        assert len(result) == 1
        assert result[0].metadata is not None
        assert result[0].metadata.document_type == DocumentType.HOMEPAGE
        assert summary.documents_input == 1

    def test_metadata_populated(self) -> None:
        docs = [_success_doc("https://example.com/about", text="About us page.")]
        result, _ = enrich_documents(docs, official_host="example.com")
        meta = result[0].metadata
        assert meta is not None
        assert meta.document_type == DocumentType.ABOUT
        assert meta.authority_score > 0.0
        assert meta.quality_score > 0.0
        assert meta.word_count > 0
        assert meta.content_hash != ""
        assert meta.trust_level == "official"
        assert meta.source_provider == ""

    def test_custom_source_provider(self) -> None:
        docs = [_success_doc("https://example.com/")]
        result, _ = enrich_documents(docs, source_provider="website")
        assert result[0].metadata.source_provider == "website"

    def test_duplicates_removed(self) -> None:
        docs = [
            _success_doc("https://a.com/p1", text="Same content."),
            _success_doc("https://a.com/p2", text="Same content."),
        ]
        result, summary = enrich_documents(docs)
        assert len(result) == 1
        assert summary.duplicates_removed == 1

    def test_diagnostic_summary(self) -> None:
        docs = [
            _success_doc("https://a.com/", text="Home content."),
            _success_doc("https://a.com/about", text="About content."),
        ]
        _, summary = enrich_documents(docs)
        assert summary.documents_input == 2
        assert summary.documents_classified >= 1
        assert summary.average_authority > 0.0
        assert summary.average_quality > 0.0
        assert summary.classification_confidence > 0.0
        assert summary.processing_duration_ms >= 0
        assert "homepage" in summary.document_type_distribution

    def test_structured_content(self) -> None:
        doc = _success_doc("https://a.com/page", text="Page content.")
        sc = {doc.id: {"headings": ["Features", "Pricing"], "paragraphs": ["P1", "P2"]}}
        result, _ = enrich_documents(
            [doc], structured_content=sc,
        )
        meta = result[0].metadata
        assert meta.heading_count == 2
        assert meta.heading_count > 0

    def test_deterministic(self) -> None:
        docs = [_success_doc("https://a.com/", text="Content.")]
        r1, s1 = enrich_documents(docs)
        r2, s2 = enrich_documents(docs)
        assert r1[0].metadata.document_type == r2[0].metadata.document_type
        assert s1.average_quality == s2.average_quality

    def test_failed_docs_get_no_metadata(self) -> None:
        docs = [_failed_doc("https://a.com/page")]
        result, _ = enrich_documents(docs)
        assert result[0].metadata is None

    def test_results_sorted_by_url(self) -> None:
        docs = [
            _success_doc("https://b.com/page", text="B content."),
            _success_doc("https://a.com/page", text="A content."),
        ]
        result, _ = enrich_documents(docs)
        assert str(result[0].url) < str(result[1].url)


# ===================================================================
# Part 7 — DocumentIntelligence class
# ===================================================================


class TestDocumentIntelligence:
    def test_basic_enrich(self) -> None:
        di = DocumentIntelligence(official_host="example.com", source_provider="test")
        docs = [_success_doc("https://example.com/", text="Welcome.")]
        result, summary = di.enrich(docs)
        assert len(result) == 1
        assert result[0].metadata is not None
        assert result[0].metadata.source_provider == "test"

    def test_default_config(self) -> None:
        di = DocumentIntelligence()
        docs = [_success_doc("https://example.com/")]
        result, _ = di.enrich(docs)
        assert result[0].metadata is not None

    def test_name(self) -> None:
        di = DocumentIntelligence()
        assert di.name == "document_intelligence"


# ===================================================================
# Backward Compatibility
# ===================================================================


class TestBackwardCompatibility:
    def test_evidence_document_without_metadata(self) -> None:
        doc = _doc()
        assert doc.metadata is None
        assert doc.title == ""
        assert doc.text == "Hello world content."

    def test_evidence_bundle_without_intelligence(self) -> None:
        from predictron_engine.evidence.models import EvidenceBundle
        bundle = EvidenceBundle(startup_name="TestCo")
        assert bundle.intelligence is None
        assert bundle.documents == []

    def test_existing_providers_unchanged(self) -> None:
        """EvidenceProvider protocol and ProviderResult should be unchanged."""
        from predictron_engine.evidence.provider_contracts import ProviderResult
        r = ProviderResult(provider="test")
        assert r.provider == "test"
        assert r.documents == []

    def test_existing_extractors_unaffected(self) -> None:
        """Extractors access doc.text and doc.title — these are unchanged."""
        doc = _success_doc(
            "https://example.com/about",
            title="About Us",
            text="We are a company that does things.",
        )
        assert doc.title == "About Us"
        assert "company" in doc.text
