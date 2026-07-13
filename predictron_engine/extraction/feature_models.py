"""Feature extraction models — service protocols for text analysis.

Defines the DomainExtractor protocol for individual extractors and the
NlpService protocol for pluggable NLP backends.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.extracted_features import ExtractedFeatures
from predictron_engine.models.startup import Startup


@runtime_checkable
class DomainExtractor(Protocol):
    """Protocol for individual domain-specific extractors.

    Each domain extractor implements this protocol independently.
    The CompositeExtractor orchestrates multiple DomainExtractors
    into a single FeatureExtractor-compatible interface.
    """

    def extract(self, startup: Startup, data: CollectedData) -> ExtractedFeatures: ...


@runtime_checkable
class NlpService(Protocol):
    """Protocol for NLP services used during feature extraction.

    Implementations might wrap spaCy, a transformer model,
    or a remote NLP API. Individual extractors delegate text analysis
    to this service when one is provided.
    """

    def extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text."""
        ...

    def extract_keywords(self, text: str, top_n: int = 10) -> list[str]:
        """Extract the most relevant keywords from text."""
        ...
