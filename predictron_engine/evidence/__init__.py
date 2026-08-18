"""Evidence module — contextual domain knowledge and web evidence collection.

The evidence layer contains three independent subsystems:

1. Domain knowledge providers (extract → evidence → reason), implemented in
   ``evidence_engine.py`` and ``providers/``. Evidence items are objective,
   domain-specific facts retrieved from the knowledge base — observations
   about the domain, not conclusions about the startup.

2. The Evidence Collection Layer (Sprint 1) — deterministic, asynchronous
   collection of structured evidence from a company's website, implemented
   in ``orchestrator.py``, ``provider_contracts.py``,
   ``website_provider.py``, ``discover.py``, ``fetcher.py``, ``cleaner.py``,
   and ``models.py``.

3. The Internet Search Discovery Layer (Sprint 3B / 4A / 4B) — discovers,
   ranks, prioritises, and selects candidate URLs across the public
   internet via a pluggable search backend and an evidence prioritization
   layer, implemented in ``search_interfaces.py``, ``ranking.py``,
   ``prioritization.py``, ``search_provider.py``, and ``search_backends.py``.

4. The Document Intelligence Layer (Sprint 4C) — classifies documents,
   computes quality metrics, estimates authority, and detects duplicates.
   Implemented in ``document_intelligence.py``.
"""

from predictron_engine.evidence.cleaner import HtmlCleaner
from predictron_engine.evidence.discover import DefaultPageDiscoverer
from predictron_engine.evidence.document_intelligence import (
    DocumentIntelligence,
    QualityMetrics,
    classify_document,
    compute_content_hash,
    compute_quality_metrics,
    detect_duplicates,
    enrich_documents,
    estimate_authority,
)
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
    DocumentMetadata,
    DocumentStatus,
    DocumentType,
    EvidenceBundle,
    EvidenceDocument,
    EvidenceSource,
    FetchResult,
    IntelligenceSummary,
    PageCandidate,
    PageType,
    ProviderRun,
    RetrievalMethod,
    make_document_id,
)
from predictron_engine.evidence.orchestrator import EvidenceOrchestrator
from predictron_engine.evidence.prioritization import (
    OfficialWebsiteResult,
    PrioritizationSettings,
    PrioritizedPage,
    identify_official_website,
    prioritise_pages,
    score_evidence_quality,
    select_pages_for_fetch,
)
from predictron_engine.evidence.provider_contracts import (
    CollectContext,
    EvidenceProvider,
    PrioritizationSummary,
    ProviderResult,
)
from predictron_engine.evidence.ranking import RankedUrl, rank_urls, score_url
from predictron_engine.evidence.search_backends import TavilySearchBackend
from predictron_engine.evidence.search_interfaces import (
    SearchBackend,
    SearchResult,
    SearchSettings,
)
from predictron_engine.evidence.search_provider import SearchEvidenceProvider
from predictron_engine.evidence.url_utils import (
    ensure_scheme,
    extract_host,
    has_valid_scheme,
    normalise_url_for_dedup,
    normalise_website,
    parse_http_url,
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
    "DocumentIntelligence",
    "DocumentMetadata",
    "DocumentStatus",
    "DocumentType",
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
    "IntelligenceSummary",
    "InvalidWebsiteError",
    "OfficialWebsiteResult",
    "PageCandidate",
    "PageType",
    "PrioritizationSettings",
    "PrioritizationSummary",
    "PrioritizedPage",
    "ProviderResult",
    "ProviderRun",
    "QualityMetrics",
    "RankedUrl",
    "RetrievalMethod",
    "SearchBackend",
    "SearchEvidenceProvider",
    "SearchResult",
    "SearchSettings",
    "TavilySearchBackend",
    "WebsiteEvidenceProvider",
    "WebsiteProviderSettings",
    "classify_document",
    "compute_content_hash",
    "compute_quality_metrics",
    "detect_duplicates",
    "ensure_scheme",
    "enrich_documents",
    "estimate_authority",
    "extract_host",
    "has_valid_scheme",
    "identify_official_website",
    "make_document_id",
    "normalise_url_for_dedup",
    "normalise_website",
    "parse_http_url",
    "prioritise_pages",
    "rank_urls",
    "score_evidence_quality",
    "score_url",
    "select_pages_for_fetch",
]
