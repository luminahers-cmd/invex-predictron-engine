"""Technology context reasoning rule.

Evaluates technology evidence to produce observations about
the startup's technical stack and its implications.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from predictron_engine.reasoning.rules.base import evidence_ref, feature_ref, filter_evidence

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation


class TechnologyContextRule:
    """Produces observations about technology stack implications.

    Evaluates technology evidence to describe ecosystem, talent, and
    operational factors related to the startup's technical choices.
    """

    @property
    def name(self) -> str:
        return "technology_context"

    def evaluate(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
    ) -> list[Observation]:
        from predictron_engine.knowledge.concepts import AnalysisDimension
        from predictron_engine.models.report import Observation as Obs

        if not features.technology_stack:
            return []

        tech_evidence = filter_evidence(evidence, "technology")
        refs: list[str] = [
            feature_ref("technology_stack", ",".join(features.technology_stack))
        ]

        for e in tech_evidence:
            refs.append(evidence_ref(e))

        confidence = 0.5
        if tech_evidence:
            confidence = min(0.5 + len(tech_evidence) * 0.05, 1.0)

        tech_count = len(features.technology_stack)
        evidence_count = len(tech_evidence)

        statement = (
            f"The technology stack includes {tech_count} identified "
            f"technologies with {evidence_count} documented domain factors."
        )

        return [
            Obs(
                dimension=AnalysisDimension.PRODUCT_STRENGTH.value,
                category="technology_assessment",
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=0.6,
                source_rule="TechnologyContextRule",
            )
        ]

    def evaluate_context(self, context) -> list[Observation]:
        """Context-aware evaluation using pre-indexed ReasoningContext lookups.

        Produces the same observations as :meth:`evaluate` but resolves
        technology evidence through the context's indexes instead of
        scanning the raw evidence list.
        """
        return self.evaluate(context.features, context.evidence_items)
