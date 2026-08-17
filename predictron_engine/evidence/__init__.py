"""Evidence module — contextual domain knowledge and web evidence collection.

The evidence layer contains two independent subsystems:

1. Domain knowledge providers (extract → evidence → reason), implemented in
   ``evidence_engine.py`` and ``providers/``. Evidence items are objective,
   domain-specific facts retrieved from the knowledge base — observations
   about the domain, not conclusions about the startup.

2. The Evidence Collection Layer (Sprint 1) — deterministic, asynchronous
   collection of structured evidence from a company's website, implemented
   in ``orchestrator.py``, ``provider_contracts.py``,
   ``website_provider.py``, ``discover.py``, ``fetcher.py``, ``cleaner.py``,
   and ``models.py``.
"""

from predictron_engine.evidence.cleaner import HtmlCleaner
from predictron_engine.evidence.discover import DefaultPageDiscoverer
from predictron_engine.evidence.evidence_engine import DefaultEvidenceEngine
from predictron_engine.evidence.evidence_models import EvidenceItem, EvidenceSet
from predictron_engine.evidence.exceptions import (
    CleanError,
    DiscoveryError,
    EvidenceCollectionError,
    FetchError,
    InvalidWebsiteError,
)
from predictron_engine.evidence.fetcher import FetcherSettings, HttpPageFetcher
from predictron_engine.evidence.models import (
    CleanResult,
    DocumentStatus,
    EvidenceBundle,
    EvidenceDocument,
    EvidenceSource,
    FetchResult,
    PageCandidate,
    PageType,
    ProviderRun,
    RetrievalMethod,
    make_document_id,
)
from predictron_engine.evidence.orchestrator import EvidenceOrchestrator
from predictron_engine.evidence.provider_contracts import (
    CollectContext,
    EvidenceProvider,
    ProviderResult,
)
from predictron_engine.evidence.website_provider import (
    WebsiteEvidenceProvider,
    WebsiteProviderSettings,
)

__all__ = [
    "CleanError",
    "CleanResult",
    "CollectContext",
    "DefaultEvidenceEngine",
    "DefaultPageDiscoverer",
    "DiscoveryError",
    "DocumentStatus",
    "EvidenceBundle",
    "EvidenceCollectionError",
    "EvidenceDocument",
    "EvidenceItem",
    "EvidenceOrchestrator",
    "EvidenceProvider",
    "EvidenceSet",
    "EvidenceSource",
    "FetchError",
    "FetchResult",
    "FetcherSettings",
    "HtmlCleaner",
    "HttpPageFetcher",
    "InvalidWebsiteError",
    "PageCandidate",
    "PageType",
    "ProviderResult",
    "ProviderRun",
    "RetrievalMethod",
    "WebsiteEvidenceProvider",
    "WebsiteProviderSettings",
    "make_document_id",
]
