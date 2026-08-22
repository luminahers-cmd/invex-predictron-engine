"""Base extractor — shared utilities for all domain-specific extractors.

Provides common text processing, keyword matching, NLP delegation,
and evidence-aware extraction utilities that individual extractors
inherit.  Extractors with no need for shared logic can bypass this
base class entirely.
"""

from __future__ import annotations

import re

from predictron_engine.evidence.models import EvidenceBundle
from predictron_engine.extraction.feature_models import NlpService

STOPWORDS: frozenset[str] = frozenset({
    "the", "that", "this", "with", "from", "have", "been",
    "were", "they", "their", "which", "about", "would",
    "could", "should", "there", "than", "into", "also",
    "more", "most", "some", "only", "very", "just",
})


class BaseExtractor:
    """Shared utilities for keyword-based feature extraction.

    Provides:
      - Keyword counting against text
      - Stopword-filtered word extraction
      - NLP delegation when an NlpService is injected
      - Evidence-aware extraction utilities (Sprint 5C)

    Subclasses are free to use or ignore these utilities.
    """

    def __init__(self, nlp_service: NlpService | None = None) -> None:
        self._nlp = nlp_service

    def _count_keyword_matches(self, text: str, keywords: list[str]) -> int:
        """Return the number of keywords found in text."""
        return sum(1 for kw in keywords if kw in text)

    def _evidence_text(self, evidence: EvidenceBundle | None) -> str:
        """Return the concatenated text of all successfully cleaned evidence pages.

        Returns an empty string when no evidence was collected. Used by
        extractors to enrich keyword-based classification with website
        content without changing behavior when evidence is absent.
        """
        if evidence is None:
            return ""
        return "\n\n".join(
            doc.text for doc in evidence.documents if doc.status.value == "success"
        )

    def _evidence_text_from_docs(self, docs: list) -> str:
        """Return concatenated text from a filtered list of evidence documents.

        Used by extractors after domain-specific retrieval to get text
        only from the most relevant documents.
        """
        return "\n\n".join(doc.text for doc in docs if doc.text)

    def _combined_text(self, description: str, evidence: EvidenceBundle | None) -> str:
        """Return the description unchanged when no evidence is available.

        When evidence exists, returns the description combined with the
        cleaned website text so keyword rules can observe both inputs.
        """
        if evidence is None:
            return description
        evidence_text = self._evidence_text(evidence)
        if not evidence_text:
            return description
        return f"{description}\n{evidence_text}"

    def _combined_text_from_docs(self, description: str, docs: list) -> str:
        """Combine description with text from filtered evidence documents.

        Used by extractors after domain-specific retrieval.
        """
        docs_text = self._evidence_text_from_docs(docs)
        if not docs_text:
            return description
        return f"{description}\n{docs_text}"

    def _build_citations_for_items(
        self,
        evidence_items: list,
        documents: list,
    ) -> list:
        """Build citations for a list of evidence items against source documents.

        Delegates to the citation builder module. Returns a list of
        EvidenceCitation objects.
        """
        from predictron_engine.evidence.citation import build_citations
        return build_citations(evidence_items, documents)

    def _build_trust_summary(self, documents: list) -> str:
        """Build a human-readable trust summary string from documents."""
        if not documents:
            return "no evidence documents"

        high = 0
        medium = 0
        low = 0
        no_trust = 0

        for doc in documents:
            if doc.metadata is None or doc.metadata.trust_score is None:
                no_trust += 1
                continue
            score = doc.metadata.trust_score.overall
            if score >= 0.7:
                high += 1
            elif score >= 0.4:
                medium += 1
            else:
                low += 1

        parts: list[str] = []
        if high:
            parts.append(f"{high} high-trust")
        if medium:
            parts.append(f"{medium} medium-trust")
        if low:
            parts.append(f"{low} low-trust")
        if no_trust:
            parts.append(f"{no_trust} untrusted")

        return ", ".join(parts) + " sources" if parts else "no trust-scored sources"

    def _populate_evidence_provenance(
        self,
        features,
        *,
        documents: list,
        evidence_items: list,
        citations: list,
        evidence_confidence: float,
        agreement_ratio: float,
        conflict_count: int,
    ) -> None:
        """Populate evidence-aware provenance fields on ExtractedFeatures.

        Mutates the features object in place to add Sprint 5C fields.
        """
        doc_ids = list({doc.id for doc in documents})
        sources = list({item.source for item in evidence_items})

        features.provenance_document_ids = list(
            dict.fromkeys(features.provenance_document_ids + doc_ids)
        )
        features.evidence_confidence = evidence_confidence
        features.evidence_agreement_ratio = agreement_ratio
        features.evidence_conflict_count = conflict_count
        features.evidence_sources_used = sources
        features.evidence_document_count = len(documents)
        features.provenance_trust_summary = self._build_trust_summary(documents)

        provider_items: dict[str, list[dict]] = {}
        provider_meta: dict[str, dict] = {}
        for item in evidence_items:
            provider_items.setdefault(item.source, []).append({
                "domain": item.domain,
                "category": item.category,
                "statement": item.statement,
                "source": item.source,
                "relevance_score": item.relevance_score,
            })

        for source, items_list in provider_items.items():
            provider_meta[source] = {
                "item_count": len(items_list),
                "domains": list({i["domain"] for i in items_list}),
            }

        features.provider_evidence = provider_items
        features.provider_run_metadata = provider_meta

    def _best_keyword_match(
        self, text: str, keyword_map: dict[str, list[str]]
    ) -> str | None:
        """Return the key with the most keyword matches, or None."""
        best_key: str | None = None
        best_score = 0
        for key, keywords in keyword_map.items():
            score = self._count_keyword_matches(text, keywords)
            if score > best_score:
                best_score = score
                best_key = key
        return best_key if best_score > 0 else None

    def _extract_words(self, text: str, max_words: int = 20) -> list[str]:
        """Extract significant words from text, filtering stopwords."""
        if self._nlp is not None:
            return self._nlp.extract_keywords(text, top_n=max_words)

        words = re.findall(r"[a-z]{4,}", text.lower())
        seen: set[str] = set()
        result: list[str] = []
        for word in words:
            if word not in STOPWORDS and word not in seen:
                seen.add(word)
                result.append(word)
        return result[:max_words]

    def _extract_tech_terms(self, text: str) -> list[str]:
        """Extract technology-related terms from text."""
        tech_patterns = [
            "python", "javascript", "typescript", "java", "golang", "rust",
            "react", "angular", "vue", "node", "django", "flask", "fastapi",
            "aws", "gcp", "azure", "docker", "kubernetes", "terraform",
            "postgresql", "mysql", "mongodb", "redis", "graphql", "rest",
            "tensorflow", "pytorch", "kafka", "spark", "airflow",
            "blockchain", "smart contract", "solidity",
        ]
        text_lower = text.lower()
        return [term for term in tech_patterns if term in text_lower]
