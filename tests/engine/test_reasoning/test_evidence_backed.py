"""Tests for Sprint 6A evidence-backed observation enrichment."""

import json

from predictron_engine.evidence.evidence_models import EvidenceItem
from predictron_engine.models.report import EvidenceCitation, Observation
from predictron_engine.reasoning.evidence_backed import (
    enrich_observation,
    match_evidence_for_observation,
)
from tests.engine.test_reasoning.conftest import make_document


def _item(domain, category, statement, source="src-a", **extra):
    return EvidenceItem(
        domain=domain,
        category=category,
        statement=statement,
        source=source,
        **extra,
    )


def _ref(item):
    return f"evidence:{item.domain}/{item.category}: {item.statement}"


def _obs(*refs, **extra):
    return Observation(
        dimension="market_opportunity",
        category="market_context",
        statement="Observation statement.",
        evidence=list(refs),
        confidence=0.7,
        source_rule="TestRule",
        **extra,
    )


class TestMatchEvidence:
    def test_refs_resolve_to_items(self):
        item = _item("industry", "market_size", "Big market.")
        obs = _obs(_ref(item))
        assert match_evidence_for_observation(obs, [item]) == [item]

    def test_feature_refs_never_match(self):
        item = _item("industry", "market_size", "Big market.")
        obs = _obs("feature:industry=fintech")
        assert match_evidence_for_observation(obs, [item]) == []

    def test_duplicates_removed_preserving_order(self):
        item = _item("industry", "market_size", "Big market.")
        obs = _obs(_ref(item), _ref(item))
        assert match_evidence_for_observation(obs, [item]) == [item]


class TestEnrichTrustScore:
    def test_trust_from_citations(self):
        doc = make_document("doc-1", trust=0.85)
        citation = EvidenceCitation(
            claim="Big market.",
            domain="industry",
            category="market_size",
            source_document_ids=["doc-1"],
            best_trust_score=0.85,
        )
        item = _item(
            "industry",
            "market_size",
            "Big market.",
            citations=[citation],
        )
        enriched = enrich_observation(_obs(_ref(item)), [item], [doc])
        assert enriched.trust_score == 0.85

    def test_trust_falls_back_to_provenance_record(self):
        record = {"document_id": "doc-9", "trust_score": 0.65}
        item = _item(
            "industry", "market_size", "Big market.",
            provenance_record=json.dumps(record),
        )
        enriched = enrich_observation(_obs(_ref(item)), [item])
        assert enriched.trust_score == 0.65

    def test_trust_zero_without_evidence_metadata(self):
        item = _item("industry", "market_size", "Big market.")
        enriched = enrich_observation(_obs(_ref(item)), [item])
        assert enriched.trust_score == 0.0


class TestEnrichProvenance:
    def test_provenance_ids_from_citations(self):
        citation = EvidenceCitation(
            claim="Big market.",
            domain="industry",
            category="market_size",
            source_document_ids=["doc-1", "doc-2"],
        )
        item = _item(
            "industry", "market_size", "Big market.",
            citations=[citation],
        )
        enriched = enrich_observation(_obs(_ref(item)), [item])
        assert enriched.provenance_document_ids == ["doc-1", "doc-2"]

    def test_provenance_ids_merge_and_deduplicate(self):
        citation = EvidenceCitation(
            claim="Big market.",
            domain="industry",
            category="market_size",
            source_document_ids=["doc-1"],
        )
        record = {"document_id": "doc-2"}
        cited_item = _item(
            "industry", "market_size", "Big market.",
            source="src-a",
            citations=[citation],
        )
        provenance_item = _item(
            "industry", "market_size", "Big market too.",
            source="src-b",
            provenance_record=json.dumps(record),
        )
        obs = _obs(_ref(cited_item), _ref(provenance_item))
        enriched = enrich_observation(obs, [cited_item, provenance_item])
        assert enriched.provenance_document_ids == ["doc-1", "doc-2"]


class TestEnrichAgreementAndConflicts:
    def test_corroborated_items_yield_full_agreement(self):
        a = _item(
            "industry", "market_size", "Big market.", source="src-a",
            citations=[EvidenceCitation(
                claim="Big market.", domain="industry", category="market_size",
                source_document_ids=["doc-1"],
            )],
        )
        b = _item(
            "industry", "market_size", "Bigger market.", source="src-b",
            citations=[EvidenceCitation(
                claim="Bigger market.", domain="industry", category="market_size",
                source_document_ids=["doc-2"],
            )],
        )
        enriched = enrich_observation(_obs(_ref(a), _ref(b)), [a, b])
        assert enriched.evidence_agreement_ratio == 1.0

    def test_single_source_items_yield_zero_agreement(self):
        a = _item("industry", "market_size", "Big market.", source="src-a")
        b = _item("geography", "market_size", "Large region.", source="src-a")
        enriched = enrich_observation(_obs(_ref(a), _ref(b)), [a, b])
        assert enriched.evidence_agreement_ratio == 0.0

    def test_one_source_document_is_not_corroboration(self):
        a = _item("industry", "market_size", "Big market.", source="src-a")
        b = _item("industry", "market_size", "Bigger market.", source="src-b")
        enriched = enrich_observation(_obs(_ref(a), _ref(b)), [a, b])
        assert enriched.evidence_agreement_ratio == 0.0

    def test_conflicting_signals_counted(self):
        a = _item(
            "business_model",
            "revenue",
            "The company has recurring revenue streams.",
            source="src-a",
        )
        b = _item(
            "business_model",
            "revenue",
            "The company has transactional revenue streams.",
            source="src-b",
        )
        enriched = enrich_observation(_obs(_ref(a), _ref(b)), [a, b])
        assert enriched.evidence_conflict_count >= 1

    def test_no_conflicts_for_agreeing_items(self):
        a = _item("industry", "market_size", "Big market.", source="src-a")
        b = _item("industry", "regulatory", "Regulated sector.", source="src-b")
        enriched = enrich_observation(_obs(_ref(a), _ref(b)), [a, b])
        assert enriched.evidence_conflict_count == 0


class TestEnrichCitations:
    def test_existing_citations_preserved(self):
        own = EvidenceCitation(
            claim="Own claim", domain="d", category="c"
        )
        obs = _obs("feature:x=1", citations=[own])
        enriched = enrich_observation(obs, [])
        assert enriched.citations == [own]

    def test_citations_adopted_from_matched_items(self):
        adopted = EvidenceCitation(
            claim="Big market.",
            domain="industry",
            category="market_size",
        )
        item = _item(
            "industry", "market_size", "Big market.",
            citations=[adopted],
        )
        enriched = enrich_observation(_obs(_ref(item)), [item])
        assert enriched.citations == [adopted]


class TestEnrichmentProperties:
    def test_enrichment_does_not_mutate_input(self):
        item = _item("industry", "market_size", "Big market.")
        obs = _obs(_ref(item))
        enrich_observation(obs, [item])
        assert obs.trust_score == 0.0
        assert obs.provenance_document_ids == []
        assert obs.evidence_agreement_ratio == 0.0

    def test_core_fields_unchanged(self):
        item = _item("industry", "market_size", "Big market.")
        obs = _obs(_ref(item))
        enriched = enrich_observation(obs, [item])
        assert enriched.statement == obs.statement
        assert enriched.confidence == obs.confidence
        assert enriched.source_rule == obs.source_rule
        assert enriched.dimension == obs.dimension
        assert enriched.importance == obs.importance

    def test_deterministic_outputs(self):
        a = _item("industry", "market_size", "Big market.", source="s1")
        b = _item("industry", "market_size", "Bigger market.", source="s2")
        obs = _obs(_ref(a), _ref(b))
        first = enrich_observation(obs, [a, b])
        second = enrich_observation(obs, [a, b])
        assert first.model_dump() == second.model_dump()
