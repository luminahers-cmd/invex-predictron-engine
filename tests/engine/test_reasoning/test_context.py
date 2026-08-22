"""Tests for the Sprint 6A ReasoningContext."""

import json

from predictron_engine.evidence.evidence_models import EvidenceItem
from predictron_engine.evidence.models import EvidenceBundle
from predictron_engine.reasoning.context import ReasoningContext
from tests.engine.test_reasoning.conftest import make_document


class TestReasoningContextEvidenceLookups:
    """Domain-level evidence lookups are pre-indexed and deterministic."""

    def test_evidence_for_domain_filters_correctly(self, rich_features, rich_evidence):
        ctx = ReasoningContext(rich_features, rich_evidence)
        industry = ctx.evidence_for_domain("industry")
        assert len(industry) == 2
        assert all(i.domain == "industry" for i in industry)

    def test_evidence_for_unknown_domain_returns_empty(self, rich_features, rich_evidence):
        ctx = ReasoningContext(rich_features, rich_evidence)
        assert ctx.evidence_for_domain("nonexistent") == []

    def test_evidence_for_domains_combines_multiple(self, rich_features, rich_evidence):
        ctx = ReasoningContext(rich_features, rich_evidence)
        combined = ctx.evidence_for_domains(["industry", "geography"])
        assert {i.domain for i in combined} == {"industry", "geography"}

    def test_lookups_are_deterministic(self, rich_features, rich_evidence):
        a = ReasoningContext(rich_features, rich_evidence).evidence_for_domain("industry")
        b = ReasoningContext(rich_features, rich_evidence).evidence_for_domain("industry")
        assert [i.statement for i in a] == [i.statement for i in b]


class TestReasoningContextDocuments:
    """Document accessors only expose successful documents."""

    def test_documents_exclude_failed(self, rich_features):
        from predictron_engine.evidence.models import DocumentStatus

        good = make_document("doc-1")
        bad = make_document("doc-2")
        bad.status = DocumentStatus.FAILED
        bundle = EvidenceBundle(
            startup_name="x", documents=[good, bad]
        )
        ctx = ReasoningContext(rich_features, [], bundle)
        assert [d.id for d in ctx.documents] == ["doc-1"]

    def test_documents_by_provider(self, rich_features):
        doc_a = make_document("doc-1", provider="website")
        doc_b = make_document("doc-2", url="https://other.com", provider="search")
        bundle = EvidenceBundle(startup_name="x", documents=[doc_a, doc_b])
        ctx = ReasoningContext(rich_features, [], bundle)
        assert [d.id for d in ctx.documents_by_provider("search")] == ["doc-2"]

    def test_document_by_id_resolves(self, rich_features):
        doc = make_document("doc-1")
        bundle = EvidenceBundle(startup_name="x", documents=[doc])
        ctx = ReasoningContext(rich_features, [], bundle)
        assert ctx.document_by_id("doc-1") is doc
        assert ctx.document_by_id("missing") is None

    def test_trusted_documents_sorted_desc(self, rich_features):
        low = make_document("low", trust=0.3)
        high = make_document("high", url="https://example.com/h", trust=0.9)
        mid = make_document("mid", url="https://example.com/m", trust=0.6)
        bundle = EvidenceBundle(startup_name="x", documents=[low, high, mid])
        ctx = ReasoningContext(rich_features, [], bundle)
        trusted = ctx.trusted_documents(min_trust=0.5)
        assert [d.id for d in trusted] == ["high", "mid"]

    def test_best_source_returns_highest_trust(self, rich_features):
        low = make_document("low", trust=0.3)
        high = make_document("high", url="https://example.com/h", trust=0.95)
        bundle = EvidenceBundle(startup_name="x", documents=[low, high])
        ctx = ReasoningContext(rich_features, [], bundle)
        assert ctx.best_source().id == "high"

    def test_best_source_none_without_scores(self, rich_features):
        doc = make_document("doc-1")  # no trust metadata
        bundle = EvidenceBundle(startup_name="x", documents=[doc])
        ctx = ReasoningContext(rich_features, [], bundle)
        assert ctx.best_source() is None


class TestReasoningContextTrustAndDiagnostics:
    """Trust summary, retrieval and extraction diagnostics exposure."""

    def test_average_trust_computed_from_documents(self, rich_features):
        docs = [
            make_document("a", trust=0.8),
            make_document("b", url="https://example.com/b", trust=0.4),
        ]
        bundle = EvidenceBundle(startup_name="x", documents=docs)
        ctx = ReasoningContext(rich_features, [], bundle)
        assert ctx.average_trust == round((0.8 + 0.4) / 2, 4)

    def test_average_trust_zero_without_scores(self, rich_features):
        bundle = EvidenceBundle(startup_name="x")
        ctx = ReasoningContext(rich_features, [], bundle)
        assert ctx.average_trust == 0.0

    def test_trust_summary_exposed_from_bundle(self, rich_features):
        from predictron_engine.evidence.provenance import TrustSummary

        bundle = EvidenceBundle(
            startup_name="x",
            trust_summary=TrustSummary(average_trust_score=0.7, total_documents=2),
        )
        ctx = ReasoningContext(rich_features, [], bundle)
        assert ctx.trust_summary.average_trust_score == 0.7

    def test_retrieval_diagnostics_from_bundle_providers(self, rich_features):
        from predictron_engine.evidence.models import ProviderRun

        run = ProviderRun(provider="website", documents=3, success=True)
        bundle = EvidenceBundle(startup_name="x", providers=[run])
        ctx = ReasoningContext(rich_features, [], bundle)
        assert ctx.retrieval_diagnostics == [run]

    def test_extraction_diagnostics_from_features(self, rich_features):
        features = rich_features.model_copy(
            update={
                "provider_run_metadata": {
                    "industry": {"documents": 4, "average_trust": 0.8}
                }
            }
        )
        ctx = ReasoningContext(features, [])
        assert ctx.extraction_diagnostics["industry"]["documents"] == 4


class TestReasoningContextProvenance:
    """Provenance reference accessors."""

    def test_provenance_document_ids_deduplicated(self, rich_features):
        features = rich_features.model_copy(
            update={"provenance_document_ids": ["d1", "d2", "d1"]}
        )
        ctx = ReasoningContext(features, [])
        assert ctx.provenance_document_ids == ["d1", "d2"]

    def test_provenance_records_parsed_from_items(self, rich_features):
        record = {"document_id": "d1", "trust_score": 0.65}
        item = EvidenceItem(
            domain="industry",
            category="market_size",
            statement="Market is large.",
            source="src",
            provenance_record=json.dumps(record),
        )
        ctx = ReasoningContext(rich_features, [item])
        assert ctx.provenance_records() == [record]

    def test_unparsable_provenance_records_skipped(self, rich_features):
        item = EvidenceItem(
            domain="industry",
            category="market_size",
            statement="Market is large.",
            source="src",
            provenance_record="{not json",
        )
        ctx = ReasoningContext(rich_features, [item])
        assert ctx.provenance_records() == []


class TestReasoningContextCitations:
    """Citation aggregation per domain."""

    def test_citations_for_domain_collects_item_citations(
        self, rich_features
    ):
        from predictron_engine.models.report import EvidenceCitation

        citation = EvidenceCitation(
            claim="Big market",
            domain="industry",
            category="market_size",
            best_trust_score=0.9,
        )
        item = EvidenceItem(
            domain="industry",
            category="market_size",
            statement="Market is large.",
            source="src",
            citations=[citation],
        )
        ctx = ReasoningContext(rich_features, [item])
        assert ctx.citations_for_domain("industry") == [citation]
        assert ctx.citations_for_domain("geography") == []
