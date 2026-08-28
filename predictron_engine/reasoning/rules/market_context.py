"""Market context reasoning rule — observes market conditions.

Evaluates industry and geographic evidence to produce observations
about the market environment the startup operates in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from predictron_engine.reasoning.rules.base import evidence_ref, feature_ref, filter_evidence

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation
    from predictron_engine.reasoning.context import ReasoningContext


class MarketContextRule:
    """Produces observations about the startup's market context.

    Evaluates industry and geography evidence to describe the market
    environment, including size, competition, and regulatory factors.
    """

    @property
    def name(self) -> str:
        return "market_context"

    def evaluate(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
    ) -> list[Observation]:
        from predictron_engine.knowledge.concepts import AnalysisDimension
        from predictron_engine.models.report import Observation as Obs

        observations: list[Obs] = []
        industry_evidence = filter_evidence(evidence, "industry")
        geo_evidence = filter_evidence(evidence, "geography")

        if industry_evidence:
            refs = [evidence_ref(e) for e in industry_evidence]
            if features.industry:
                refs.append(feature_ref("industry", features.industry))

            categories = {e.category for e in industry_evidence}
            key_factors = ", ".join(sorted(categories)[:3])

            observations.append(
                Obs(
                    dimension=AnalysisDimension.MARKET_OPPORTUNITY.value,
                    category="market_context",
                    statement=(
                        f"The {features.industry} market presents domain-specific "
                        f"factors including {key_factors}."
                    ),
                    evidence=refs,
                    confidence=min(0.5 + len(industry_evidence) * 0.1, 1.0),
                    importance=0.8,
                    source_rule="MarketContextRule",
                )
            )

        if geo_evidence:
            refs = [evidence_ref(e) for e in geo_evidence]
            region = features.geography or features.headquarters_region
            if region:
                refs.append(feature_ref("geography", region))

            observations.append(
                Obs(
                    dimension=AnalysisDimension.MARKET_OPPORTUNITY.value,
                    category="market_context",
                    statement=(
                        f"The {region or 'target'} geographic market has "
                        f"{len(geo_evidence)} documented domain factors."
                    ),
                    evidence=refs,
                    confidence=min(0.5 + len(geo_evidence) * 0.1, 1.0),
                    importance=0.6,
                    source_rule="MarketContextRule",
                )
            )

        return observations

    def evaluate_context(self, context: ReasoningContext) -> list[Observation]:
        """Context-aware evaluation using pre-indexed ReasoningContext lookups.

        Produces the same observations as :meth:`evaluate` but resolves
        domain evidence through the context's indexes instead of scanning
        the raw evidence list.
        """
        return self.evaluate(context.features, context.evidence_items)
