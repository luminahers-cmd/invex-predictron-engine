"""Stage expectation reasoning rule.

Evaluates funding stage evidence to produce observations about
what is typically expected at the startup's current stage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from predictron_engine.reasoning.rules.base import evidence_ref, feature_ref, filter_evidence

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation


class StageExpectationRule:
    """Produces observations about funding stage expectations.

    Evaluates funding stage evidence to describe typical expectations,
    risk profiles, and diligence focus at the current stage.
    """

    @property
    def name(self) -> str:
        return "stage_expectation"

    def evaluate(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
    ) -> list[Observation]:
        from predictron_engine.knowledge.concepts import AnalysisDimension
        from predictron_engine.models.report import Observation as Obs

        if features.funding_stage is None:
            return []

        stage_evidence = filter_evidence(evidence, "funding_stage")
        refs: list[str] = [feature_ref("funding_stage", features.funding_stage)]

        for e in stage_evidence:
            refs.append(evidence_ref(e))

        confidence = 0.5
        if stage_evidence:
            confidence = min(0.5 + len(stage_evidence) * 0.1, 1.0)

        categories = {e.category for e in stage_evidence} if stage_evidence else set()
        key_areas = ", ".join(sorted(categories)[:3]) if categories else "standard expectations"

        return [
            Obs(
                dimension=AnalysisDimension.TRACTION_SIGNALS.value,
                category="stage_assessment",
                statement=(
                    f"At the {features.funding_stage} stage, evaluation should "
                    f"consider {key_areas}."
                ),
                evidence=refs,
                confidence=confidence,
                importance=0.7,
                source_rule="StageExpectationRule",
            )
        ]
