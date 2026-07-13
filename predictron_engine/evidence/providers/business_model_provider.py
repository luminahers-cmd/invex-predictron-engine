"""Business model evidence provider — retrieves domain knowledge for business models.

Gathers contextual facts about the startup's business model from the
knowledge base. Each fact is an objective observation about the model,
not a conclusion about the startup.
"""

from __future__ import annotations

from predictron_engine.evidence.evidence_models import EvidenceItem
from predictron_engine.knowledge.taxonomies import BusinessModel

# Domain knowledge per business model — objective facts
BUSINESS_MODEL_EVIDENCE: dict[BusinessModel, list[dict[str, str]]] = {
    BusinessModel.SAAS: [
        {"category": "unit_economics", "statement": "SaaS businesses typically achieve 70-90% gross margins."},
        {"category": "revenue_model", "statement": "Subscription revenue provides predictable recurring cash flows."},
        {"category": "retention", "statement": "Net revenue retention above 100% indicates strong product stickiness."},
        {"category": "competition", "statement": "SaaS markets consolidate rapidly around category leaders."},
    ],
    BusinessModel.PAAS: [
        {"category": "unit_economics", "statement": "Platform businesses benefit from usage-based scaling."},
        {"category": "switching_costs", "statement": "Developer platform adoption creates deep integration lock-in."},
        {"category": "competition", "statement": "Platform markets tend toward monopoly or duopoly outcomes."},
    ],
    BusinessModel.MARKETPLACE: [
        {"category": "unit_economics", "statement": "Marketplace take rates vary from 5-30% depending on vertical."},
        {"category": "competition", "statement": "Marketplace businesses benefit from strong network effects."},
        {"category": "growth", "statement": "Two-sided marketplaces face chicken-and-egg cold start challenges."},
    ],
    BusinessModel.ECOMMERCE: [
        {"category": "unit_economics", "statement": "Ecommerce gross margins are typically 20-50% depending on category."},
        {"category": "operations", "statement": "Inventory management and fulfillment are critical operational factors."},
        {"category": "competition", "statement": "Direct-to-consumer brands compete on brand and customer experience."},
    ],
    BusinessModel.ADVERTISING: [
        {"category": "unit_economics", "statement": "Ad-supported models require large audiences to generate meaningful revenue."},
        {"category": "competition", "statement": "Digital advertising is dominated by a few large platforms."},
        {"category": "growth", "statement": "User growth and engagement metrics directly drive revenue potential."},
    ],
    BusinessModel.TRANSACTIONAL: [
        {"category": "unit_economics", "statement": "Per-transaction revenue scales linearly with volume."},
        {"category": "competition", "statement": "Payment processing is a low-margin, high-volume business."},
        {"category": "regulatory", "statement": "Financial transactions require compliance with payment regulations."},
    ],
    BusinessModel.LICENSING: [
        {"category": "unit_economics", "statement": "License revenue is lumpy and harder to predict than subscriptions."},
        {"category": "sales_cycle", "statement": "Enterprise license deals involve extended negotiation cycles."},
        {"category": "retention", "statement": "License renewals depend on demonstrated ongoing value."},
    ],
    BusinessModel.HARDWARE_PLUS_SOFTWARE: [
        {"category": "unit_economics", "statement": "Hardware margins are typically lower than software-only businesses."},
        {"category": "operations", "statement": "Hardware production requires supply chain and manufacturing management."},
        {"category": "switching_costs", "statement": "Hardware-software integration creates strong ecosystem lock-in."},
    ],
    BusinessModel.SERVICES: [
        {"category": "unit_economics", "statement": "Services businesses scale linearly with headcount."},
        {"category": "competition", "statement": "Services differentiation is driven by expertise and reputation."},
        {"category": "growth", "statement": "Productization of services is key to achieving scale."},
    ],
}


class BusinessModelEvidenceProvider:
    """Gathers domain evidence based on the startup's business model.

    Looks up the business model in the knowledge base and returns
    objective factual statements about that model's characteristics.
    """

    def gather(self, features: object) -> list[EvidenceItem]:
        from predictron_engine.models.extracted_features import ExtractedFeatures

        if not isinstance(features, ExtractedFeatures):
            return []
        if features.business_model is None:
            return []

        model_key = features.business_model.lower()

        matched_model: BusinessModel | None = None
        for model in BusinessModel:
            if model.value == model_key:
                matched_model = model
                break

        if matched_model is None or matched_model not in BUSINESS_MODEL_EVIDENCE:
            return []

        items = BUSINESS_MODEL_EVIDENCE[matched_model]
        return [
            EvidenceItem(
                domain="business_model",
                category=item["category"],
                statement=item["statement"],
                source=f"knowledge/taxonomies.py:{matched_model.value}",
                relevance_score=1.0,
            )
            for item in items
        ]
