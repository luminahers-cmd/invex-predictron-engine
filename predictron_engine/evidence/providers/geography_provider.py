"""Geography evidence provider — retrieves domain knowledge for geographic markets.

Gathers contextual facts about the startup's geographic market from the
knowledge base. Each fact is an objective observation about the region,
not a conclusion about the startup.
"""

from __future__ import annotations

from predictron_engine.evidence.evidence_models import EvidenceItem

# Domain knowledge per geography — objective facts about the market
GEOGRAPHY_EVIDENCE: dict[str, list[dict[str, str]]] = {
    "north_america": [
        {"category": "market_size", "statement": "North America has the largest venture capital market globally."},
        {"category": "competition", "statement": "North American markets are highly competitive with many well-funded competitors."},
        {"category": "talent", "statement": "North America has deep talent pools in technology and business."},
        {"category": "regulatory", "statement": "Regulatory environment varies by state and federal jurisdiction."},
    ],
    "europe": [
        {"category": "market_size", "statement": "European tech market is growing but remains fragmented across countries."},
        {"category": "regulatory", "statement": "GDPR and EU regulations impose strict data privacy requirements."},
        {"category": "competition", "statement": "European markets have fewer mega-rounds but strong deep-tech ecosystems."},
    ],
    "asia_pacific": [
        {"category": "market_size", "statement": "Asia Pacific has the fastest-growing technology markets globally."},
        {"category": "competition", "statement": "Asian markets have dominant local champions in most technology sectors."},
        {"category": "regulatory", "statement": "Regulatory requirements vary significantly across Asian markets."},
    ],
    "latin_america": [
        {"category": "market_size", "statement": "Latin American tech investment has grown significantly in recent years."},
        {"category": "competition", "statement": "Latin American markets are less saturated than North American markets."},
        {"category": "risk", "statement": "Currency volatility and political instability are market considerations."},
    ],
    "middle_east_africa": [
        {"category": "market_size", "statement": "Middle East and Africa represent emerging technology markets."},
        {"category": "competition", "statement": "Digital infrastructure gaps create opportunities for first movers."},
        {"category": "risk", "statement": "Political and regulatory uncertainty varies across the region."},
    ],
    "global": [
        {"category": "market_size", "statement": "Global market focus provides access to the largest addressable market."},
        {"category": "competition", "statement": "Global companies compete with local leaders in every market."},
        {"category": "operations", "statement": "Operating globally requires multi-market regulatory compliance."},
    ],
}


class GeographyEvidenceProvider:
    """Gathers domain evidence based on the startup's geographic market.

    Returns objective factual statements about the geographic region
    the startup operates in, covering market size, competition, and
    regulatory factors.
    """

    def gather(self, features: object) -> list[EvidenceItem]:
        from predictron_engine.models.extracted_features import ExtractedFeatures

        if not isinstance(features, ExtractedFeatures):
            return []

        region = features.geography or features.headquarters_region
        if region is None:
            return []

        region_key = region.lower()

        if region_key not in GEOGRAPHY_EVIDENCE:
            return []

        items = GEOGRAPHY_EVIDENCE[region_key]
        return [
            EvidenceItem(
                domain="geography",
                category=item["category"],
                statement=item["statement"],
                source=f"knowledge/geography:{region_key}",
                relevance_score=1.0,
            )
            for item in items
        ]
