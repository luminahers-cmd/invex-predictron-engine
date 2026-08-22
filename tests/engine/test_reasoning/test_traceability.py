"""Tests for Sprint 6A provenance traceability.

Chain under test:
    Observation -> EvidenceItem -> EvidenceCitation ->
    EvidenceDocument -> Provider -> Original URL
"""

import json

from predictron_engine.evidence.evidence_models import EvidenceItem
from predictron_engine.models.report import EvidenceCitation, Observation
from predictron_engine.reasoning.traceability import (
    resolve_provenance_chain,
    unresolved_document_ids,
    verify_traceability,
)
from tests.engine.test_reasoning.conftest import make_document


def _observation_with_citation(doc_ids, urls, provider="knowledge/src"):
    citation = EvidenceCitation(
        claim="Big market.",
        domain="industry",
        category="market_size",
        source_document_ids=doc_ids,
        source_urls=urls,
        best_trust_score=0.9,
        provider=provider,
    )
    return Observation(
        dimension="market_opportunity",
        category="market_context",
        statement="Observation.",
        citations=[citation],
        source_rule="R",
    )


class TestProvenanceChainResolution:
    def test_full_chain_resolves_to_url_and_provider(self):
        doc = make_document(
            "doc-1",
            url="https://example.com/about",
            trust=0.9,
            provider="website",
        )
        obs = _observation_with_citation(["doc-1"], ["https://example.com/about"])
        entries = resolve_provenance_chain(obs, [doc])

        assert len(entries) == 1
        entry = entries[0]
        assert entry.resolved is True
        assert entry.document_id == "doc-1"
        assert entry.original_url == "https://example.com/about"
        assert entry.provider == "website"
        assert entry.knowledge_provider == "knowledge/src"
        assert entry.trust_score == 0.9

    def test_chain_preserves_claim_and_domain(self):
        doc = make_document("doc-1")
        obs = _observation_with_citation(["doc-1"], [])
        entry = resolve_provenance_chain(obs, [doc])[0]
        assert entry.claim == "Big market."
        assert entry.domain == "industry"
        assert entry.category == "market_size"

    def test_unresolved_id_falls_back_to_citation_url(self):
        obs = _observation_with_citation(["ghost"], ["https://example.com/x"])
        entries = resolve_provenance_chain(obs, [])
        assert entries[0].resolved is False
        assert entries[0].original_url == "https://example.com/x"

    def test_multiple_document_ids_resolve_in_order(self):
        docs = [
            make_document("doc-1", url="https://a.com"),
            make_document("doc-2", url="https://b.com"),
        ]
        obs = _observation_with_citation(
            ["doc-1", "doc-2"], ["https://a.com", "https://b.com"]
        )
        entries = resolve_provenance_chain(obs, docs)
        assert [e.document_id for e in entries] == ["doc-1", "doc-2"]
        assert all(e.resolved for e in entries)


class TestTraceabilityVerification:
    def test_fully_cited_observation_verifies(self):
        doc = make_document("doc-1")
        obs = _observation_with_citation(["doc-1"], [])
        assert verify_traceability(obs, [doc]) is True
        assert unresolved_document_ids(obs, [doc]) == []

    def test_missing_document_detected(self):
        obs = _observation_with_citation(["ghost"], [])
        assert verify_traceability(obs, []) is False
        assert unresolved_document_ids(obs, []) == ["ghost"]

    def test_observation_without_citations_is_trivially_traceable(self):
        obs = Observation(
            dimension="d",
            category="c",
            statement="No citations.",
            source_rule="R",
        )
        assert verify_traceability(obs, []) is True


class TestEndToEndProvenancePreservation:
    """Enrichment must carry document ids through to the observation."""

    def test_enriched_observation_traces_to_original_url(self):
        from predictron_engine.reasoning.evidence_backed import (
            enrich_observation,
        )

        doc = make_document(
            "doc-1",
            url="https://example.com/about",
            provider="website",
        )
        citation = EvidenceCitation(
            claim="Big market.",
            domain="industry",
            category="market_size",
            source_document_ids=["doc-1"],
            best_trust_score=0.9,
        )
        item = EvidenceItem(
            domain="industry",
            category="market_size",
            statement="Big market.",
            source="src",
            provenance_record=json.dumps({"document_id": "doc-1"}),
            citations=[citation],
        )
        raw_obs = Observation(
            dimension="market_opportunity",
            category="market_context",
            statement="Obs.",
            evidence=[
                "evidence:industry/market_size: Big market."
            ],
            source_rule="R",
        )

        enriched = enrich_observation(raw_obs, [item], [doc])
        assert enriched.provenance_document_ids == ["doc-1"]
        assert verify_traceability(enriched, [doc]) is True

        chain = resolve_provenance_chain(enriched, [doc])
        assert chain[0].original_url == "https://example.com/about"
        assert chain[0].provider == "website"

    def test_deterministic_chain_resolution(self):
        doc = make_document("doc-1")
        obs = _observation_with_citation(["doc-1"], [])
        a = resolve_provenance_chain(obs, [doc])
        b = resolve_provenance_chain(obs, [doc])
        assert a == b
