"""Traction extractor — extracts traction-related factual attributes.

Responsibilities:
  - Funding stage classification via keyword taxonomy matching
  - Revenue signal detection (from description keywords)
"""

from __future__ import annotations

from predictron_engine.extraction.extractors.base import BaseExtractor
from predictron_engine.knowledge.stages import STAGE_KEYWORDS
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

REVENUE_KEYWORDS: list[str] = [
    "revenue", "mrr", "arr", "recurring", "paying customers",
    "monetized", "profitable", "cash flow", "sales",
]


class TractionExtractor(BaseExtractor):
    """Extracts traction attributes from startup data.

    Populates: funding_stage, has_revenue.
    """

    def extract(self, startup: Startup, data: CollectedData) -> ExtractedFeatures:
        description_lower = startup.description.lower()
        funding_stage = self._classify_funding_stage(description_lower)
        has_revenue = self._detect_revenue_signal(description_lower)

        return ExtractedFeatures(
            funding_stage=funding_stage,
            has_revenue=has_revenue,
        )

    def _classify_funding_stage(self, text: str) -> str | None:
        for stage, keywords in STAGE_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    return stage.name.lower()
        return None

    def _detect_revenue_signal(self, text: str) -> bool | None:
        for kw in REVENUE_KEYWORDS:
            if kw in text:
                return True
        return None
