"""Composite extractor — orchestrates specialized extractors into ExtractedFeatures.

The CompositeExtractor runs all registered domain extractors, then merges
their partial results into a single ExtractedFeatures object. This preserves
the FeatureExtractor protocol while enabling single-responsibility composition.

Merge strategy:
  - Scalar fields: last non-default value wins (order-determined)
  - List fields: union of all contributions, preserving insertion order
  - None / empty-list / 0 / False are treated as "no contribution"

Adding a new extractor:
  1. Create a class implementing extract(startup, data) -> ExtractedFeatures
  2. Add it to the default extractors list or inject via the constructor
  3. No other module changes required
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from predictron_engine.extraction.extractors.business_model import BusinessModelExtractor
from predictron_engine.extraction.extractors.company import CompanyExtractor
from predictron_engine.extraction.extractors.competition import CompetitionExtractor
from predictron_engine.extraction.extractors.founder import FounderExtractor
from predictron_engine.extraction.extractors.market import MarketExtractor
from predictron_engine.extraction.extractors.metadata import MetadataExtractor
from predictron_engine.extraction.extractors.product import ProductExtractor
from predictron_engine.extraction.extractors.risk import RiskExtractor
from predictron_engine.extraction.extractors.technology import TechnologyExtractor
from predictron_engine.extraction.extractors.traction import TractionExtractor
from predictron_engine.extraction.feature_models import NlpService
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

logger = logging.getLogger(__name__)

# Fields where non-empty list overrides empty list
_LIST_FIELDS: frozenset[str] = frozenset({
    "technology_stack",
    "key_keywords",
    "market_keywords",
    "market_signals",
    "market_characteristics",
    "domain_expertise_signals",
    "serial_founder_indicators",
    "leadership_roles",
    "hiring_signals",
    "advisor_mentions",
    "founder_market_fit_signals",
    "execution_signals",
    "primary_capabilities",
    "feature_signals",
    "integration_ecosystem",
    "differentiation_signals",
    "defensibility_signals",
    "scalability_indicators",
    "innovation_signals",
    "product_keywords",
    "programming_language_signals",
    "framework_signals",
    "cloud_infrastructure_signals",
    "data_architecture_signals",
    "security_signals",
    "open_source_signals",
    "developer_tooling_signals",
    "technology_keywords",
    "direct_competitor_signals",
    "indirect_competitor_signals",
    "incumbent_signals",
    "fragmentation_signals",
    "winner_take_most_signals",
    "switching_cost_signals",
    "differentiation_signals",
    "competitive_moat_indicators",
    "barriers_to_entry",
    "substitute_product_signals",
    "platform_dependency",
    "ecosystem_dependency",
    "open_source_competition",
    "regulatory_competition",
    "geographic_competition",
    "pricing_pressure",
    "competitive_keywords",
    "market_risk",
    "founder_risk",
    "execution_risk",
    "product_risk",
    "technology_risk",
    "business_model_risk",
    "traction_risk",
    "competitive_risk",
    "regulatory_risk",
    "operational_risk",
    "platform_dependency_risk",
    "customer_concentration_risk",
    "hiring_risk",
    "funding_risk",
    "scaling_risk",
    "security_risk",
    "compliance_risk",
    "risk_keywords",
})


def _default_extractors() -> list[object]:
    """Return the default ordered list of domain extractors."""
    return [
        CompanyExtractor(),
        ProductExtractor(),
        MarketExtractor(),
        FounderExtractor(),
        BusinessModelExtractor(),
        TechnologyExtractor(),
        TractionExtractor(),
        CompetitionExtractor(),
        RiskExtractor(),
        MetadataExtractor(),
    ]


@dataclass
class CompositeExtractor:
    """Orchestrates multiple domain extractors into a single ExtractedFeatures.

    Each extractor is independent and replaces a single concern.
    The composite merges all outputs using a deterministic overlay strategy.

    Parameters
    ----------
    extractors:
        Ordered list of extractors to run. Later extractors can override
        earlier ones for the same scalar field (last-write-wins).
    nlp_service:
        Optional NLP service injected into all BaseExtractor subclasses.
    """

    extractors: list[object] = field(default_factory=_default_extractors)
    nlp_service: NlpService | None = None

    def __post_init__(self) -> None:
        if self.nlp_service is not None:
            self._inject_nlp()

    def extract(self, startup: Startup, data: CollectedData) -> ExtractedFeatures:
        """Run all extractors and merge into a single ExtractedFeatures."""
        logger.info(
            "CompositeExtractor: running %d extractors for %s",
            len(self.extractors),
            startup.name,
        )

        merged = ExtractedFeatures()

        for extractor in self.extractors:
            partial = extractor.extract(startup, data)
            merged = self._merge(merged, partial)

        return merged

    def _merge(
        self, base: ExtractedFeatures, overlay: ExtractedFeatures
    ) -> ExtractedFeatures:
        """Merge overlay onto base. Overlay wins for scalars; lists are unioned."""
        base_dict = base.model_dump()
        overlay_dict = overlay.model_dump()

        merged_data: dict[str, object] = {}

        for field_name in base_dict:
            base_val = base_dict[field_name]
            overlay_val = overlay_dict[field_name]

            if field_name in _LIST_FIELDS:
                base_list = base_val if isinstance(base_val, list) else []
                overlay_list = overlay_val if isinstance(overlay_val, list) else []
                combined = list(dict.fromkeys(base_list + overlay_list))
                merged_data[field_name] = combined
            elif overlay_val is not None and overlay_val != 0 and overlay_val is not False:
                merged_data[field_name] = overlay_val
            else:
                merged_data[field_name] = base_val

        return ExtractedFeatures(**merged_data)

    def _inject_nlp(self) -> None:
        """Inject NlpService into all BaseExtractor subclasses."""
        from predictron_engine.extraction.extractors.base import BaseExtractor

        for ext in self.extractors:
            if isinstance(ext, BaseExtractor):
                ext._nlp = self.nlp_service  # noqa: SLF001
