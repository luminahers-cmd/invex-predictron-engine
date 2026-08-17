"""Feature Extractor — legacy-compatible wrapper around CompositeExtractor.

The DefaultFeatureExtractor preserves the original FeatureExtractor protocol
while delegating all extraction to the CompositeExtractor. This ensures
backward compatibility while enabling the new composition-based architecture.

For new code, prefer importing CompositeExtractor directly and configuring
individual extractors.
"""

from __future__ import annotations

import logging

from predictron_engine.evidence.models import EvidenceBundle
from predictron_engine.extraction.composite import CompositeExtractor
from predictron_engine.extraction.feature_models import NlpService
from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup

logger = logging.getLogger(__name__)


class DefaultFeatureExtractor:
    """Legacy-compatible FeatureExtractor that delegates to CompositeExtractor.

    Satisfies the FeatureExtractor protocol while using the new
    composition-based architecture internally.
    """

    def __init__(self, nlp_service: NlpService | None = None) -> None:
        self._composite = CompositeExtractor(nlp_service=nlp_service)

    def extract(
        self,
        startup: Startup,
        data: CollectedData,
        evidence: EvidenceBundle | None = None,
    ) -> ExtractedFeatures:
        """Extract structured features via the composite extractor pipeline."""
        logger.info(
            "DefaultFeatureExtractor delegating to CompositeExtractor for: %s",
            startup.name,
        )
        return self._composite.extract(startup, data, evidence)
