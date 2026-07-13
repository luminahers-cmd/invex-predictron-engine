"""Stage evidence provider — retrieves domain knowledge for funding stages.

Gathers contextual facts about the startup's funding stage from the
knowledge base. Each fact is an objective observation about the stage,
not a conclusion about the startup.
"""

from __future__ import annotations

from predictron_engine.evidence.evidence_models import EvidenceItem
from predictron_engine.knowledge.stages import STAGE_CONTEXT, FundingStage

STAGE_EVIDENCE: dict[FundingStage, list[dict[str, str]]] = {
    FundingStage.PRE_SEED: [
        {"category": "expectations", "statement": "Pre-seed companies are typically pre-product."},
        {"category": "risk", "statement": "Pre-seed investments carry the highest risk."},
        {"category": "diligence", "statement": "Founding team quality is the primary criterion."},
    ],
    FundingStage.SEED: [
        {"category": "expectations", "statement": "Seed companies are building MVP and initial customers."},
        {"category": "risk", "statement": "Seed investments have high failure rates but high upside."},
        {"category": "diligence", "statement": "Product-market fit signals are the key diligence focus."},
    ],
    FundingStage.SERIES_A: [
        {"category": "expectations", "statement": "Series A companies have early product-market fit."},
        {"category": "metrics", "statement": "Revenue growth and unit economics become important."},
        {"category": "diligence", "statement": "Go-to-market scalability is the primary focus."},
    ],
    FundingStage.SERIES_B: [
        {"category": "expectations", "statement": "Series B companies have a proven model."},
        {"category": "metrics", "statement": "Revenue growth and retention are key metrics."},
        {"category": "diligence", "statement": "Path to profitability and positioning are priorities."},
    ],
    FundingStage.SERIES_C: [
        {"category": "expectations", "statement": "Series C companies are accelerating growth."},
        {"category": "metrics", "statement": "Revenue scale and margin trajectory are key."},
        {"category": "diligence", "statement": "International expansion and M&A are topics."},
    ],
    FundingStage.SERIES_D: [
        {"category": "expectations", "statement": "Series D companies are preparing for IPO."},
        {"category": "metrics", "statement": "Profitability and IPO readiness are key criteria."},
    ],
    FundingStage.SERIES_E_PLUS: [
        {"category": "expectations", "statement": "Late-stage rounds indicate IPO preparation."},
        {"category": "risk", "statement": "Late-stage investors expect near-term liquidity."},
    ],
    FundingStage.GROWTH: [
        {"category": "expectations", "statement": "Growth companies optimize profitability and position."},
        {"category": "metrics", "statement": "EBITDA and cash flow are primary metrics."},
    ],
    FundingStage.IPO_READY: [
        {"category": "expectations", "statement": "IPO-ready companies are pursuing public listing."},
        {"category": "diligence", "statement": "Public markets demand rigorous governance and reporting."},
    ],
}


class StageEvidenceProvider:
    """Gathers domain evidence based on the startup's funding stage."""

    def gather(self, features: object) -> list[EvidenceItem]:
        from predictron_engine.models.extracted_features import ExtractedFeatures

        if not isinstance(features, ExtractedFeatures):
            return []
        if features.funding_stage is None:
            return []

        stage_key = features.funding_stage.lower()

        matched: FundingStage | None = None
        for stage in FundingStage:
            if stage.name.lower() == stage_key:
                matched = stage
                break

        if matched is None:
            return []

        context = STAGE_CONTEXT.get(matched, {})
        stage_label = context.get("label", matched.name)

        items: list[EvidenceItem] = [
            EvidenceItem(
                domain="funding_stage",
                category=item["category"],
                statement=item["statement"],
                source=f"knowledge/stages.py:{stage_label}",
                relevance_score=1.0,
            )
            for item in STAGE_EVIDENCE.get(matched, [])
        ]

        if context:
            items.append(
                EvidenceItem(
                    domain="funding_stage",
                    category="context",
                    statement=context.get("description", ""),
                    source=f"knowledge/stages.py:{stage_label}",
                    relevance_score=1.0,
                )
            )

        return items
