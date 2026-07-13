"""Technology extractor — extracts technology stack attributes.

Responsibilities:
  - Technology stack detection from website domain and URL metadata
  - Supplementary tech signal aggregation from enrichment data
"""

from __future__ import annotations

from predictron_engine.extraction.extractors.base import BaseExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

DOMAIN_TECH_HINTS: dict[str, str] = {
    "vercel": "vercel",
    "netlify": "netlify",
    "heroku": "heroku",
    "shopify": "shopify",
    "wordpress": "wordpress",
    "webflow": "webflow",
    "squarespace": "squarespace",
    "github": "github",
}


class TechnologyExtractor(BaseExtractor):
    """Extracts technology stack attributes from startup data.

    Populates: technology_stack (from domain/URL signals).
    """

    def extract(self, startup: Startup, data: CollectedData) -> ExtractedFeatures:
        tech_from_domain = self._detect_tech_from_domain(data.website_domain or "")
        tech_from_enrichment = self._detect_tech_from_enrichment(data.enrichment_signals)

        combined = list(dict.fromkeys(tech_from_domain + tech_from_enrichment))

        return ExtractedFeatures(
            technology_stack=combined,
        )

    def _detect_tech_from_domain(self, domain: str) -> list[str]:
        results: list[str] = []
        for hint, tech in DOMAIN_TECH_HINTS.items():
            if hint in domain:
                results.append(tech)
        return results

    def _detect_tech_from_enrichment(
        self, signals: dict[str, str | int | float | bool]
    ) -> list[str]:
        results: list[str] = []
        for key, value in signals.items():
            if "tech" in key.lower() and isinstance(value, str):
                results.append(value)
        return results
