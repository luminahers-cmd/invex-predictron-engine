"""Base extractor — shared utilities for all domain-specific extractors.

Provides common text processing, keyword matching, and NLP delegation
that individual extractors inherit. Extractors with no need for shared
logic can bypass this base class entirely.
"""

from __future__ import annotations

import re

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

    Subclasses are free to use or ignore these utilities.
    """

    def __init__(self, nlp_service: NlpService | None = None) -> None:
        self._nlp = nlp_service

    def _count_keyword_matches(self, text: str, keywords: list[str]) -> int:
        """Return the number of keywords found in text."""
        return sum(1 for kw in keywords if kw in text)

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
