"""Business model extractor — extracts business model attributes.

Responsibilities:
  - Business model classification via keyword taxonomy matching
  - Customer type detection (B2B, B2C, B2B2C keywords)
"""

from __future__ import annotations

from predictron_engine.extraction.extractors.base import BaseExtractor
from predictron_engine.knowledge.taxonomies import MODEL_KEYWORDS
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

CUSTOMER_TYPE_KEYWORDS: dict[str, list[str]] = {
    "b2b": ["b2b", "enterprise", "business", "companies", "organizations"],
    "b2c": ["b2c", "consumer", "end user", "individual", "personal"],
    "b2b2c": ["b2b2c", "two-sided", "platform"],
}


class BusinessModelExtractor(BaseExtractor):
    """Extracts business model attributes from startup data.

    Populates: business_model, customer_type.
    """

    def extract(self, startup: Startup, data: CollectedData) -> ExtractedFeatures:
        description_lower = startup.description.lower()
        business_model = self._classify_business_model(description_lower)
        customer_type = self._classify_customer_type(description_lower)

        return ExtractedFeatures(
            business_model=business_model,
            customer_type=customer_type,
        )

    def _classify_business_model(self, text: str) -> str | None:
        keyword_map: dict[str, list[str]] = {
            model.value: keywords
            for model, keywords in MODEL_KEYWORDS.items()
        }
        return self._best_keyword_match(text, keyword_map)

    def _classify_customer_type(self, text: str) -> str | None:
        return self._best_keyword_match(text, CUSTOMER_TYPE_KEYWORDS)
