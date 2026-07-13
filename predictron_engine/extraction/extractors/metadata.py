"""Metadata extractor — extracts pipeline metadata attributes.

Responsibilities:
  - Data completeness calculation
  - Pitch deck presence propagation
  - Key keyword extraction from description
"""

from __future__ import annotations

from predictron_engine.extraction.extractors.base import BaseExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup


class MetadataExtractor(BaseExtractor):
    """Extracts pipeline metadata attributes from startup data.

    Populates: data_completeness, has_pitch_deck, key_keywords.
    """

    def extract(self, startup: Startup, data: CollectedData) -> ExtractedFeatures:
        keywords = self._extract_words(startup.description)
        completeness = self._compute_completeness(data)

        return ExtractedFeatures(
            key_keywords=keywords,
            has_pitch_deck=data.has_pitch_deck,
            data_completeness=completeness,
        )

    def _compute_completeness(self, data: CollectedData) -> float:
        """Fraction of available data signals that are non-empty."""
        checks = [
            bool(data.website_domain),
            bool(data.description_tokens),
            data.description_word_count > 0,
            data.has_website,
            data.has_pitch_deck,
            data.founder_count > 0,
            bool(data.url_metadata),
            bool(data.enrichment_signals),
        ]
        if not checks:
            return 0.0
        return round(sum(checks) / len(checks), 2)
