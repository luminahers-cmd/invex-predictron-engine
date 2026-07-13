"""Evidence Engine — gathers contextual domain knowledge from extracted features.

The evidence layer transforms extracted facts into structured evidence by
querying domain knowledge providers. It sits between extraction and reasoning
in the pipeline:

  extract → evidence → reason

Evidence items are objective, domain-specific facts retrieved from the
knowledge base. They are NOT conclusions about the startup. They are
observations about the domain the startup operates in.

Key principles:
  - Each provider is independent and single-responsibility
  - Evidence is deterministic and reproducible
  - No scoring, no reasoning, no recommendations
  - Providers are injectable and replaceable via protocol
  - The evidence engine aggregates provider outputs into an EvidenceSet
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from predictron_engine.evidence.evidence_models import EvidenceItem, EvidenceSet
from predictron_engine.evidence.providers.business_model_provider import (
    BusinessModelEvidenceProvider,
)
from predictron_engine.evidence.providers.geography_provider import (
    GeographyEvidenceProvider,
)
from predictron_engine.evidence.providers.industry_provider import (
    IndustryEvidenceProvider,
)
from predictron_engine.evidence.providers.stage_provider import StageEvidenceProvider
from predictron_engine.evidence.providers.technology_provider import (
    TechnologyEvidenceProvider,
)
from predictron_engine.models.extracted_features import ExtractedFeatures

logger = logging.getLogger(__name__)


def _default_providers() -> list[object]:
    """Return the default ordered list of evidence providers."""
    return [
        IndustryEvidenceProvider(),
        BusinessModelEvidenceProvider(),
        StageEvidenceProvider(),
        TechnologyEvidenceProvider(),
        GeographyEvidenceProvider(),
    ]


@dataclass
class DefaultEvidenceEngine:
    """Orchestrates multiple evidence providers into a single EvidenceSet.

    Runs all registered providers against the extracted features and
    aggregates the results into a single EvidenceSet.

    Parameters
    ----------
    providers:
        Ordered list of evidence providers to run.
    """

    providers: list[object] = field(default_factory=_default_providers)

    def gather(self, features: ExtractedFeatures) -> EvidenceSet:
        """Run all providers and aggregate into an EvidenceSet."""
        logger.info(
            "EvidenceEngine: running %d providers",
            len(self.providers),
        )

        all_items: list[EvidenceItem] = []
        contributing_providers = 0

        for provider in self.providers:
            try:
                items = provider.gather(features)  # type: ignore[union-attr]
                if items:
                    all_items.extend(items)
                    contributing_providers += 1
            except Exception:
                logger.warning(
                    "Provider %s failed, skipping",
                    type(provider).__name__,
                )

        coverage = self._compute_coverage(features)

        evidence_set = EvidenceSet(
            items=all_items,
            provider_count=contributing_providers,
            feature_coverage=coverage,
        )

        logger.info(
            "EvidenceEngine: collected %d items from %d providers",
            len(all_items),
            contributing_providers,
        )
        return evidence_set

    def _compute_coverage(self, features: ExtractedFeatures) -> float:
        """Fraction of populated feature fields that could produce evidence."""
        check_fields = [
            features.industry is not None,
            features.business_model is not None,
            features.funding_stage is not None,
            bool(features.technology_stack),
            features.geography is not None or features.headquarters_region is not None,
        ]
        if not check_fields:
            return 0.0
        return round(sum(check_fields) / len(check_fields), 2)
