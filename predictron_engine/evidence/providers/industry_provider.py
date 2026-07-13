"""Industry evidence provider — retrieves domain knowledge for industries.

Gathers contextual facts about the startup's industry from the knowledge
base. Each fact is an objective observation about the industry, not a
conclusion about the startup.
"""

from __future__ import annotations

from predictron_engine.evidence.evidence_models import EvidenceItem
from predictron_engine.knowledge.taxonomies import Industry

INDUSTRY_EVIDENCE: dict[Industry, list[dict[str, str]]] = {
    Industry.FINTECH: [
        {"category": "regulatory", "statement": "Financial services require regulatory compliance."},
        {"category": "sales_cycle", "statement": "Enterprise financial institutions have long procurement cycles."},
        {"category": "switching_costs", "statement": "Financial infrastructure changes carry integration risk."},
        {"category": "market_size", "statement": "Global financial services market exceeds $25 trillion."},
        {"category": "trust", "statement": "Trust and security track record are critical buying factors."},
    ],
    Industry.HEALTHTECH: [
        {"category": "regulatory", "statement": "Healthcare products face extensive regulatory approval."},
        {"category": "sales_cycle", "statement": "Hospital sales cycles typically exceed 12 months."},
        {"category": "switching_costs", "statement": "Clinical workflow integration creates high switching costs."},
        {"category": "market_size", "statement": "Digital health market is projected to exceed $300B by 2030."},
        {"category": "data_privacy", "statement": "HIPAA imposes strict data handling requirements."},
    ],
    Industry.EDTECH: [
        {"category": "sales_cycle", "statement": "Institutional sales follow academic procurement calendars."},
        {"category": "market_size", "statement": "Global edtech market exceeds $250 billion."},
        {"category": "adoption", "statement": "Education institutions are traditionally slow adopters."},
        {"category": "competition", "statement": "Edtech market has low barriers to entry and high fragmentation."},
    ],
    Industry.ENTERPRISE_SAAS: [
        {"category": "sales_cycle", "statement": "Enterprise SaaS sales cycles range from 3 to 12 months."},
        {"category": "switching_costs", "statement": "Enterprise integrations create meaningful switching costs."},
        {"category": "market_size", "statement": "Global enterprise software market exceeds $500 billion."},
        {"category": "unit_economics", "statement": "SaaS benefits from recurring revenue and high gross margins."},
        {"category": "competition", "statement": "Enterprise SaaS markets tend toward winner-take-most dynamics."},
    ],
    Industry.CONSUMER_TECH: [
        {"category": "adoption", "statement": "Consumer adoption is driven by network effects and virality."},
        {"category": "market_size", "statement": "Consumer tech market is vast but highly competitive."},
        {"category": "competition", "statement": "Consumer markets have low switching costs and high churn risk."},
        {"category": "monetization", "statement": "Consumer monetization typically requires large user bases."},
    ],
    Industry.ECOMMERCE: [
        {"category": "competition", "statement": "Ecommerce faces intense platform competition."},
        {"category": "unit_economics", "statement": "Ecommerce margins are thinner than software businesses."},
        {"category": "logistics", "statement": "Physical product fulfillment adds operational complexity."},
        {"category": "market_size", "statement": "Global ecommerce market exceeds $6 trillion."},
    ],
    Industry.AI_ML: [
        {"category": "competition", "statement": "AI/ML talent is scarce and expensive to recruit."},
        {"category": "market_size", "statement": "AI market is projected to exceed $2 trillion by 2030."},
        {"category": "moat", "statement": "Proprietary data and model performance create defensible moats."},
        {"category": "regulatory", "statement": "AI regulation is evolving with new compliance requirements."},
    ],
    Industry.CYBERSECURITY: [
        {"category": "market_size", "statement": "Global cybersecurity market exceeds $200 billion."},
        {"category": "regulatory", "statement": "Data breach regulations drive mandatory security spending."},
        {"category": "switching_costs", "statement": "Security tool integration creates significant lock-in."},
        {"category": "sales_cycle", "statement": "Security purchases are often compliance-triggered."},
    ],
    Industry.CLIMATE_TECH: [
        {"category": "market_size", "statement": "Climate tech investment exceeds $100 billion annually."},
        {"category": "regulatory", "statement": "Government incentives and carbon policies drive adoption."},
        {"category": "capital_intensity", "statement": "Climate tech solutions require significant capital."},
        {"category": "sales_cycle", "statement": "Energy projects have multi-year development timelines."},
    ],
    Industry.BIOTECH: [
        {"category": "regulatory", "statement": "Biotech products face lengthy FDA approval processes."},
        {"category": "capital_intensity", "statement": "Drug development requires substantial R&D investment."},
        {"category": "market_size", "statement": "Global biotech market exceeds $1.5 trillion."},
        {"category": "risk", "statement": "Clinical trial failure rates create binary outcomes."},
    ],
    Industry.HARDWARE: [
        {"category": "capital_intensity", "statement": "Hardware requires significant manufacturing investment."},
        {"category": "competition", "statement": "Hardware margins are typically lower than software."},
        {"category": "supply_chain", "statement": "Global supply chain dependencies create operational risk."},
    ],
    Industry.MARKETPLACE: [
        {"category": "competition", "statement": "Marketplaces face chicken-and-egg cold start problems."},
        {"category": "unit_economics", "statement": "Marketplace take rates vary significantly by vertical."},
        {"category": "network_effects", "statement": "Successful marketplaces benefit from strong network effects."},
    ],
    Industry.LOGISTICS: [
        {"category": "capital_intensity", "statement": "Logistics requires substantial capital investment."},
        {"category": "competition", "statement": "Logistics markets are dominated by established players."},
        {"category": "operations", "statement": "Operational efficiency is the primary differentiator."},
    ],
    Industry.GAMING: [
        {"category": "market_size", "statement": "Global gaming market exceeds $200 billion."},
        {"category": "competition", "statement": "Hit-driven dynamics create high outcome variance."},
        {"category": "monetization", "statement": "Free-to-play with in-app purchases dominates."},
    ],
    Industry.MEDIA_ENTERTAINMENT: [
        {"category": "competition", "statement": "Content costs rise while distribution is commoditized."},
        {"category": "monetization", "statement": "Subscription and advertising models dominate."},
        {"category": "market_size", "statement": "Global media market exceeds $2 trillion."},
    ],
}


class IndustryEvidenceProvider:
    """Gathers domain evidence based on the startup's industry classification."""

    def gather(self, features: object) -> list[EvidenceItem]:
        from predictron_engine.models.extracted_features import ExtractedFeatures

        if not isinstance(features, ExtractedFeatures):
            return []
        if features.industry is None:
            return []

        industry_key = features.industry.lower()

        matched: Industry | None = None
        for ind in Industry:
            if ind.value == industry_key:
                matched = ind
                break

        if matched is None or matched not in INDUSTRY_EVIDENCE:
            return []

        return [
            EvidenceItem(
                domain="industry",
                category=item["category"],
                statement=item["statement"],
                source=f"knowledge/taxonomies.py:{matched.value}",
                relevance_score=1.0,
            )
            for item in INDUSTRY_EVIDENCE[matched]
        ]
