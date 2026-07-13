"""Founder extractor — extracts founder-related factual attributes.

Responsibilities:
  - Founder profile count propagation from collected data
  - Team size indicator detection (from description keywords)
"""

from __future__ import annotations

import re

from predictron_engine.extraction.extractors.base import BaseExtractor
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

TEAM_SIZE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("1-10", re.compile(r"\b(?:team of|employees?|people)\s+(?:about\s+)?(\d{1,2})\b", re.I)),
    ("11-50", re.compile(r"\b(?:team of|employees?|people)\s+(?:about\s+)?(\d{2,3})\b", re.I)),
    ("51-200", re.compile(r"\b(?:team of|employees?|people)\s+(?:about\s+)?(\d{3})\b", re.I)),
]


class FounderExtractor(BaseExtractor):
    """Extracts founder attributes from startup data.

    Populates: founder_profile_count, team_size_indicator.
    """

    def extract(self, startup: Startup, data: CollectedData) -> ExtractedFeatures:
        team_size = self._detect_team_size(startup.description)

        return ExtractedFeatures(
            founder_profile_count=data.founder_count,
            team_size_indicator=team_size,
        )

    def _detect_team_size(self, text: str) -> str | None:
        for label, pattern in TEAM_SIZE_PATTERNS:
            match = pattern.search(text)
            if match:
                count = int(match.group(1))
                if count <= 10:
                    return "1-10"
                if count <= 50:
                    return "11-50"
                if count <= 200:
                    return "51-200"
                return "200+"
        return None
