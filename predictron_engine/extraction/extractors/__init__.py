"""Domain-specific extractors — one per single-responsibility concern.

Each extractor produces only objective, structured facts. No reasoning,
no scoring, no recommendations, no predictions.

The CompositeExtractor orchestrates all extractors into a single
ExtractedFeatures object. Individual extractors can be replaced or
augmented independently without affecting the rest of the pipeline.
"""

from predictron_engine.extraction.extractors.base import BaseExtractor
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

__all__ = [
    "BaseExtractor",
    "BusinessModelExtractor",
    "CompanyExtractor",
    "CompetitionExtractor",
    "FounderExtractor",
    "MarketExtractor",
    "MetadataExtractor",
    "ProductExtractor",
    "RiskExtractor",
    "TechnologyExtractor",
    "TractionExtractor",
]
