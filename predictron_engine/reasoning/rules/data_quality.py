"""Data quality reasoning rule.

Evaluates data completeness and coverage to produce observations
about the quality and reliability of available information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from predictron_engine.reasoning.rules.base import feature_ref

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import Observation


class DataQualityRule:
    """Produces observations about data completeness and coverage.

    Evaluates the data_completeness score and the presence of key
    fields to assess how well-supported the analysis is.
    """

    @property
    def name(self) -> str:
        return "data_quality"

    def evaluate(
        self,
        features: ExtractedFeatures,
        evidence: list[EvidenceItem],
    ) -> list[Observation]:
        from predictron_engine.models.report import Observation as Obs

        completeness = features.data_completeness
        refs: list[str] = [
            feature_ref("data_completeness", f"{completeness:.2f}"),
        ]

        populated_fields = sum([
            features.industry is not None,
            features.business_model is not None,
            features.funding_stage is not None,
            bool(features.technology_stack),
            features.founder_profile_count > 0,
            features.has_revenue is not None,
            features.founded_year is not None,
        ])
        refs.append(feature_ref("populated_key_fields", populated_fields))

        if completeness >= 0.7:
            level = "high"
            statement = "High data completeness — analysis is well-supported."
            confidence = min(completeness + 0.2, 1.0)
            importance = 0.4
        elif completeness >= 0.4:
            level = "moderate"
            statement = "Moderate data completeness — analysis has some gaps."
            confidence = completeness + 0.2
            importance = 0.5
        else:
            level = "low"
            statement = (
                "Low data completeness — analysis is limited by available data."
            )
            confidence = min(completeness + 0.3, 1.0)
            importance = 0.7

        refs.append(feature_ref("completeness_level", level))

        return [
            Obs(
                dimension="data_quality",
                category="data_assessment",
                statement=statement,
                evidence=refs,
                confidence=confidence,
                importance=importance,
                source_rule="DataQualityRule",
            )
        ]
