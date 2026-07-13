"""Risk indicator reasoning rule.

Identifies risk signals from missing data, gaps in key fields,
and other observable patterns in the extracted features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from predictron_engine.reasoning.rules.base import feature_ref

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation


class RiskIndicatorRule:
    """Produces observations about risk signals in available data.

    Identifies missing key fields, low data completeness, and other
    observable gaps that could affect analysis quality.
    """

    @property
    def name(self) -> str:
        return "risk_indicator"

    def evaluate(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
    ) -> list[Observation]:
        from predictron_engine.knowledge.concepts import AnalysisDimension
        from predictron_engine.models.report import Observation as Obs

        observations: list[Obs] = []

        if not features.has_pitch_deck:
            observations.append(
                Obs(
                    dimension=AnalysisDimension.PRODUCT_STRENGTH.value,
                    category="risk_indicator",
                    statement="No pitch deck available for analysis.",
                    evidence=[feature_ref("has_pitch_deck", False)],
                    confidence=0.9,
                    importance=0.5,
                    source_rule="RiskIndicatorRule",
                )
            )

        if features.founder_profile_count == 0:
            observations.append(
                Obs(
                    dimension=AnalysisDimension.FOUNDER_QUALITY.value,
                    category="risk_indicator",
                    statement="No founder profiles identified.",
                    evidence=[feature_ref("founder_profile_count", 0)],
                    confidence=0.9,
                    importance=0.6,
                    source_rule="RiskIndicatorRule",
                )
            )

        if features.data_completeness < 0.3:
            observations.append(
                Obs(
                    dimension="data_quality",
                    category="risk_indicator",
                    statement=(
                        f"Very low data completeness "
                        f"({features.data_completeness:.0%}) limits analysis depth."
                    ),
                    evidence=[
                        feature_ref("data_completeness", f"{features.data_completeness:.2f}"),
                    ],
                    confidence=0.9,
                    importance=0.7,
                    source_rule="RiskIndicatorRule",
                )
            )

        if features.industry is None and features.business_model is None:
            observations.append(
                Obs(
                    dimension=AnalysisDimension.MARKET_OPPORTUNITY.value,
                    category="risk_indicator",
                    statement="Neither industry nor business model could be classified.",
                    evidence=[
                        feature_ref("industry", None),
                        feature_ref("business_model", None),
                    ],
                    confidence=0.8,
                    importance=0.7,
                    source_rule="RiskIndicatorRule",
                )
            )

        return observations
