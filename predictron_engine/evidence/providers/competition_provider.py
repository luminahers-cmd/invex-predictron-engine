"""Competition evidence provider — retrieves domain knowledge for competitive landscape.

Gathers contextual facts about the startup's competitive environment from
the knowledge base. Each fact is an objective observation about market
competition, not a conclusion about the startup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from predictron_engine.evidence.evidence_models import EvidenceItem

if TYPE_CHECKING:
    from predictron_engine.models.extracted_features import ExtractedFeatures

MARKET_CONCENTRATION_EVIDENCE: dict[str, list[dict[str, str]]] = {
    "fragmented": [
        {
            "category": "concentration",
            "statement": (
                "The market is fragmented with many participants. "
                "Fragmentation can indicate low barriers to entry but "
                "may limit winner-take-most dynamics."
            ),
        },
        {
            "category": "strategy",
            "statement": (
                "In fragmented markets, differentiation and consolidation "
                "are key competitive strategies."
            ),
        },
    ],
    "moderately_concentrated": [
        {
            "category": "concentration",
            "statement": (
                "The market has moderate concentration with several "
                "established players. This suggests viable competitive "
                "positioning opportunities."
            ),
        },
    ],
    "concentrated": [
        {
            "category": "concentration",
            "statement": (
                "The market is concentrated among a few dominant players. "
                "New entrants face significant competitive headwinds."
            ),
        },
        {
            "category": "strategy",
            "statement": (
                "Concentrated markets require clear differentiation "
                "or niche positioning to gain traction."
            ),
        },
    ],
    "dominated": [
        {
            "category": "concentration",
            "statement": (
                "The market is dominated by one or two established players. "
                "Disruption requires significant innovation or a differentiated "
                "approach to defensible market segments."
            ),
        },
        {
            "category": "strategy",
            "statement": (
                "In dominated markets, alternative distribution channels "
                "or underserved niches may provide entry points."
            ),
        },
    ],
}

COMPETITIVE_DENSITY_EVIDENCE: dict[str, list[dict[str, str]]] = {
    "sparse": [
        {
            "category": "density",
            "statement": (
                "Low competitive density suggests limited direct competition. "
                "This may indicate either an untapped market or a challenging "
                "business model."
            ),
        },
    ],
    "moderate": [
        {
            "category": "density",
            "statement": (
                "Moderate competitive density indicates a balanced market "
                "with room for differentiated entrants."
            ),
        },
    ],
    "dense": [
        {
            "category": "density",
            "statement": (
                "High competitive density indicates intense competition. "
                "Startups must demonstrate clear differentiation to survive."
            ),
        },
    ],
    "hyper_competitive": [
        {
            "category": "density",
            "statement": (
                "Hyper-competitive environments have rapid innovation cycles "
                "and aggressive pricing. Sustained advantage requires strong "
                "network effects or proprietary technology."
            ),
        },
    ],
}

NETWORK_EFFECT_EVIDENCE: dict[str, list[dict[str, str]]] = {
    "strong_network_effects": [
        {
            "category": "network_effects",
            "statement": (
                "Strong network effects create winner-take-most dynamics. "
                "Once established, these markets are extremely difficult "
                "for new entrants to crack."
            ),
        },
    ],
    "moderate_network_effects": [
        {
            "category": "network_effects",
            "statement": (
                "Moderate network effects provide some competitive advantage "
                "to established players but do not create insurmountable barriers."
            ),
        },
    ],
    "no_network_effects": [
        {
            "category": "network_effects",
            "statement": (
                "Absence of network effects means competition is based "
                "primarily on product features, pricing, and distribution."
            ),
        },
    ],
}


class CompetitionEvidenceProvider:
    """Gathers domain evidence based on the startup's competitive landscape features."""

    def gather(self, features: object) -> list[EvidenceItem]:
        from predictron_engine.models.extracted_features import (
            ExtractedFeatures,
        )

        if not isinstance(features, ExtractedFeatures):
            return []

        items: list[EvidenceItem] = []
        items.extend(self._gather_concentration(features))
        items.extend(self._gather_density(features))
        items.extend(self._gather_network_effects(features))
        items.extend(self._gather_barriers(features))
        items.extend(self._gather_moats(features))
        items.extend(self._gather_switching_costs(features))
        return items

    def _gather_concentration(self, features: ExtractedFeatures) -> list[EvidenceItem]:
        if not features.market_concentration:
            return []
        key = features.market_concentration.lower()
        entries = MARKET_CONCENTRATION_EVIDENCE.get(key, [])
        return [
            EvidenceItem(
                domain="competition",
                category=entry["category"],
                statement=entry["statement"],
                source=f"knowledge/competition.py:concentration:{key}",
                relevance_score=1.0,
            )
            for entry in entries
        ]

    def _gather_density(self, features: ExtractedFeatures) -> list[EvidenceItem]:
        if not features.competitive_density:
            return []
        key = features.competitive_density.lower()
        entries = COMPETITIVE_DENSITY_EVIDENCE.get(key, [])
        return [
            EvidenceItem(
                domain="competition",
                category=entry["category"],
                statement=entry["statement"],
                source=f"knowledge/competition.py:density:{key}",
                relevance_score=1.0,
            )
            for entry in entries
        ]

    def _gather_network_effects(self, features: ExtractedFeatures) -> list[EvidenceItem]:
        if not features.network_effect_competition:
            return []
        key = features.network_effect_competition.lower()
        entries = NETWORK_EFFECT_EVIDENCE.get(key, [])
        return [
            EvidenceItem(
                domain="competition",
                category=entry["category"],
                statement=entry["statement"],
                source=f"knowledge/competition.py:network_effects:{key}",
                relevance_score=1.0,
            )
            for entry in entries
        ]

    def _gather_barriers(self, features: ExtractedFeatures) -> list[EvidenceItem]:
        if not features.barriers_to_entry:
            return []
        return [
            EvidenceItem(
                domain="competition",
                category="barriers_to_entry",
                statement=f"Detected barrier to entry: {signal}",
                source="knowledge/competition.py:barriers",
                relevance_score=0.8,
            )
            for signal in features.barriers_to_entry
        ]

    def _gather_moats(self, features: ExtractedFeatures) -> list[EvidenceItem]:
        if not features.competitive_moat_indicators:
            return []
        return [
            EvidenceItem(
                domain="competition",
                category="moat",
                statement=f"Detected competitive moat indicator: {signal}",
                source="knowledge/competition.py:moats",
                relevance_score=0.8,
            )
            for signal in features.competitive_moat_indicators
        ]

    def _gather_switching_costs(self, features: ExtractedFeatures) -> list[EvidenceItem]:
        if not features.switching_cost_signals:
            return []
        return [
            EvidenceItem(
                domain="competition",
                category="switching_costs",
                statement=f"Detected switching cost signal: {signal}",
                source="knowledge/competition.py:switching_costs",
                relevance_score=0.8,
            )
            for signal in features.switching_cost_signals
        ]
