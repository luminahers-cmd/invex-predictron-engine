"""Feature extraction models — service protocols for text analysis.

Defines the DomainExtractor protocol for individual extractors and the
NlpService protocol for pluggable NLP backends.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

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


def _extract_target(extractor: object) -> Callable[..., ExtractedFeatures] | None:
    """Resolve the callable to inspect/invoke for an extractor.

    Accepts either a DomainExtractor instance (with an ``extract`` method)
    or a bare callable such as a bound method or plain function.
    """
    target = getattr(extractor, "extract", None)
    if target is None and callable(extractor):
        target = extractor
    return target


def extractor_accepts_evidence(extractor: object) -> bool:
    """Return True when an extractor's ``extract`` accepts an ``evidence`` kwarg.

    Inspects the runtime signature so custom extractors that only implement
    ``extract(startup, data)`` remain fully backward compatible and are
    simply called without evidence.
    """
    target = _extract_target(extractor)
    if target is None:
        return False
    signature = inspect.signature(target)
    return "evidence" in signature.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def call_extractor_with_evidence(
    extractor: object,
    startup: Startup,
    data: CollectedData,
    evidence: Any,
) -> ExtractedFeatures:
    """Call an extractor, passing ``evidence`` only when it accepts it."""
    target = _extract_target(extractor)
    if target is None:
        raise TypeError("Extractor is not callable and has no extract method")
    if extractor_accepts_evidence(extractor):
        return target(startup, data, evidence=evidence)
    return target(startup, data)


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
