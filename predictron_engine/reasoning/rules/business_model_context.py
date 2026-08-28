"""Business model context reasoning rule.

Evaluates business model evidence to produce observations about
the startup's revenue model and its characteristics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from predictron_engine.reasoning.rules.base import evidence_ref, feature_ref, filter_evidence

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation
    from predictron_engine.reasoning.context import ReasoningContext


class BusinessModelContextRule:
    """Produces observations about the startup's business model viability.

    Evaluates business model evidence to describe the characteristics
    and implications of the chosen revenue model.
    """

    @property
    def name(self) -> str:
        return "business_model_context"

    def evaluate(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
    ) -> list[Observation]:
        from predictron_engine.knowledge.concepts import AnalysisDimension
        from predictron_engine.models.report import Observation as Obs

        if features.business_model is None:
            return []

        bm_evidence = filter_evidence(evidence, "business_model")
        refs: list[str] = [feature_ref("business_model", features.business_model)]

        if features.industry:
            refs.append(feature_ref("industry", features.industry))

        for e in bm_evidence:
            refs.append(evidence_ref(e))

        confidence = 0.5
        if bm_evidence:
            confidence = min(0.5 + len(bm_evidence) * 0.1, 1.0)

        statement = (
            f"The startup follows a {features.business_model} model"
        )
        if features.industry:
            statement += f" within the {features.industry} industry"
        statement += "."

        if bm_evidence:
            categories = {e.category for e in bm_evidence}
            statement += (
                f" Domain knowledge covers {len(categories)} factors "
                f"including {', '.join(sorted(categories)[:3])}."
            )

        return [
            Obs(
                dimension=AnalysisDimension.BUSINESS_MODEL_VIABILITY.value,
                category="business_model_assessment",
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=0.7,
                source_rule="BusinessModelContextRule",
            )
        ]

    def evaluate_context(self, context: ReasoningContext) -> list[Observation]:
        """Context-aware evaluation using pre-indexed ReasoningContext lookups.

        Produces the same observations as :meth:`evaluate` but resolves
        business-model evidence through the context's indexes instead of
        scanning the raw evidence list.
        """
        return self.evaluate(context.features, context.evidence_items)
