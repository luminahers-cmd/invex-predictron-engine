"""Company extractor — extracts company-level factual attributes.

Responsibilities:
  - Company name propagation
  - Description length computation
  - Founded year detection (from description text patterns)
  - Headquarters region detection (from description text patterns)
"""

from __future__ import annotations

import re

from predictron_engine.extraction.extractors.base import BaseExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

FOUNDED_YEAR_PATTERN = re.compile(
    r"(?:founded|established|started|incorporated)\s+(?:in\s+)?(\d{4})",
    re.IGNORECASE,
)

REGION_KEYWORDS: dict[str, list[str]] = {
    "north_america": [
        "us", "usa", "united states", "north america",
        "silicon valley", "new york", "san francisco",
    ],
    "europe": [
        "europe", "uk", "united kingdom", "london",
        "berlin", "paris", "amsterdam",
    ],
    "asia_pacific": [
        "asia", "india", "china", "japan",
        "singapore", "asia pacific", "apac",
    ],
    "latin_america": ["latin america", "brazil", "mexico", "latam"],
    "middle_east_africa": ["middle east", "africa", "uae", "dubai"],
    "global": ["global", "worldwide", "international"],
}


class CompanyExtractor(BaseExtractor):
    """Extracts company-level attributes from startup data.

    Populates: description_length, founded_year, headquarters_region.
    """

    def extract(self, startup: Startup, data: CollectedData) -> ExtractedFeatures:
        description_lower = startup.description.lower()

        founded_year = self._detect_founded_year(startup.description)
        headquarters_region = self._detect_region(description_lower)

        return ExtractedFeatures(
            description_length=len(startup.description),
            founded_year=founded_year,
            headquarters_region=headquarters_region,
        )

    def _detect_founded_year(self, text: str) -> int | None:
        match = FOUNDED_YEAR_PATTERN.search(text)
        if match:
            year = int(match.group(1))
            if 1900 <= year <= 2100:
                return year
        return None

    def _detect_region(self, text: str) -> str | None:
        for region, keywords in REGION_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    return region
        return None
