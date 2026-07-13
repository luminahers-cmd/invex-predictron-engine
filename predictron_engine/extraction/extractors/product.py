"""Product extractor — extracts product-related factual attributes.

Responsibilities:
  - Technology stack identification from description text
  - Product type inference from description keywords
"""

from __future__ import annotations

from predictron_engine.extraction.extractors.base import BaseExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup


class ProductExtractor(BaseExtractor):
    """Extracts product attributes from startup data.

    Populates: technology_stack (from description text signals).
    """

    def extract(self, startup: Startup, data: CollectedData) -> ExtractedFeatures:
        tech_stack = self._extract_tech_terms(startup.description)

        return ExtractedFeatures(
            technology_stack=tech_stack,
        )
