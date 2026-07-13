"""Competition extractor — extracts competitive landscape attributes.

Responsibilities:
  - Competitor identification from description mentions
  - Competitive positioning signals (placeholder)

This extractor currently returns default features. It is framework-ready
for future NLP/LLM-based competitive analysis that will populate
competition-specific fields once added to ExtractedFeatures.
"""

from __future__ import annotations

from predictron_engine.extraction.extractors.base import BaseExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup


class CompetitionExtractor(BaseExtractor):
    """Extracts competitive landscape attributes from startup data.

    Currently a placeholder. Future implementations will populate
    competition-specific fields (competitor names, positioning, etc.)
    once those fields are added to ExtractedFeatures.
    """

    def extract(self, startup: Startup, data: CollectedData) -> ExtractedFeatures:
        # Placeholder: real implementation will use NLP to identify
        # competitor mentions, positioning claims, and market overlap.
        return ExtractedFeatures()
