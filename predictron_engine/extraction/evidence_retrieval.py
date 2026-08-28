"""Domain-specific evidence retrieval strategies for extractors.

Provides deterministic, pure functions that select the most relevant
evidence documents and items for each extraction domain.  Each strategy
implements a ``RetrievalStrategy`` protocol that filters evidence by
document type, keyword relevance, and trust score.

Design
------
* :class:`RetrievalStrategy` — protocol defining the retrieval interface.
* :class:`MarketRetrievalStrategy` — selects market-relevant evidence.
* :class:`FounderRetrievalStrategy` — selects founder/team evidence.
* :class:`TechnologyRetrievalStrategy` — selects technology evidence.
* :class:`BusinessModelRetrievalStrategy` — selects business model evidence.
* :class:`ProductRetrievalStrategy` — selects product-relevant evidence.
* :func:`get_strategy_for_domain` — maps domain name to strategy.

All functions are pure and side-effect-free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from predictron_engine.evidence.models import DocumentType

if TYPE_CHECKING:
    from predictron_engine.evidence.models import (
        EvidenceBundle,
        EvidenceDocument,
    )
    from predictron_engine.models.report import EvidenceItem


class RetrievalStrategy(ABC):
    """Protocol for domain-specific evidence retrieval.

    Each strategy selects evidence documents and items relevant to a
    specific extraction domain, applying trust-weighted prioritization.
    """

    @abstractmethod
    def retrieve_documents(
        self,
        bundle: EvidenceBundle,
    ) -> list[EvidenceDocument]:
        """Select and rank documents relevant to this domain.

        Returns documents sorted by relevance (highest trust first).
        """
        ...

    @abstractmethod
    def retrieve_evidence_items(
        self,
        items: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        """Filter evidence items relevant to this domain.

        Returns items sorted by relevance score descending.
        """
        ...

    @abstractmethod
    def get_keywords(self) -> list[str]:
        """Return domain-specific keywords for signal matching."""
        ...

    @abstractmethod
    def get_document_types(self) -> list[DocumentType]:
        """Return preferred document types for this domain."""
        ...


def _sort_by_trust(docs: list[EvidenceDocument]) -> list[EvidenceDocument]:
    """Sort documents by trust score descending, then by quality score."""
    return sorted(
        docs,
        key=lambda d: (
            d.metadata.trust_score.overall if d.metadata and d.metadata.trust_score else 0.0,
            d.metadata.quality_score if d.metadata else 0.0,
        ),
        reverse=True,
    )


def _filter_by_keywords(
    docs: list[EvidenceDocument],
    keywords: list[str],
) -> list[EvidenceDocument]:
    """Filter documents whose text contains at least one keyword."""
    if not keywords:
        return docs
    keywords_lower = [kw.lower() for kw in keywords]
    return [
        doc for doc in docs
        if doc.text and any(kw in doc.text.lower() for kw in keywords_lower)
    ]


def _filter_evidence_by_keywords(
    items: list[EvidenceItem],
    keywords: list[str],
) -> list[EvidenceItem]:
    """Filter evidence items whose statement contains at least one keyword."""
    if not keywords:
        return items
    keywords_lower = [kw.lower() for kw in keywords]
    return [
        item for item in items
        if any(kw in item.statement.lower() for kw in keywords_lower)
    ]


class MarketRetrievalStrategy(RetrievalStrategy):
    """Retrieves evidence relevant to market intelligence extraction.

    Prioritizes about pages, blog posts, and news — documents that
    typically contain market positioning and industry context.
    """

    _DOMAIN_KEYWORDS: list[str] = [
        "market", "industry", "industry vertical", "customer", "segment",
        "enterprise", "mid-market", "smb", "tam", "sam", "som",
        "competitor", "incumbent", "disrupt", "emerging", "growth",
        "mature", "b2b", "b2c", "target market", "market size",
    ]

    _PREFERRED_DOC_TYPES: list[DocumentType] = [
        DocumentType.HOMEPAGE,
        DocumentType.ABOUT,
        DocumentType.BLOG,
        DocumentType.NEWS,
        DocumentType.PRESS_RELEASE,
    ]

    def retrieve_documents(
        self,
        bundle: EvidenceBundle,
    ) -> list[EvidenceDocument]:
        from predictron_engine.evidence.models import DocumentStatus

        successful = [
            doc for doc in bundle.documents
            if doc.status == DocumentStatus.SUCCESS
        ]

        preferred = _filter_by_keywords(successful, self._DOMAIN_KEYWORDS)
        rest = [d for d in successful if d not in preferred]

        return _sort_by_trust(preferred + rest)

    def retrieve_evidence_items(
        self,
        items: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        market_items = [i for i in items if i.domain == "market"]
        other_items = [i for i in items if i.domain != "market"]

        return sorted(market_items + other_items, key=lambda i: i.relevance_score, reverse=True)

    def get_keywords(self) -> list[str]:
        return self._DOMAIN_KEYWORDS

    def get_document_types(self) -> list[DocumentType]:
        return self._PREFERRED_DOC_TYPES


class FounderRetrievalStrategy(RetrievalStrategy):
    """Retrieves evidence relevant to founder/team intelligence extraction.

    Prioritizes about pages, careers pages, and repository pages —
    documents that contain team and leadership information.
    """

    _DOMAIN_KEYWORDS: list[str] = [
        "founder", "ceo", "cto", "team", "engineering", "leader",
        "hiring", "recruiting", "experience", "background", "serial",
        "domain expert", "advisor", "board", "funding", "investor",
        "raised", "series", "employee", "staff",
    ]

    _PREFERRED_DOC_TYPES: list[DocumentType] = [
        DocumentType.ABOUT,
        DocumentType.CAREERS,
        DocumentType.REPOSITORY,
        DocumentType.BLOG,
    ]

    def retrieve_documents(
        self,
        bundle: EvidenceBundle,
    ) -> list[EvidenceDocument]:
        from predictron_engine.evidence.models import DocumentStatus

        successful = [
            doc for doc in bundle.documents
            if doc.status == DocumentStatus.SUCCESS
        ]

        preferred = _filter_by_keywords(successful, self._DOMAIN_KEYWORDS)
        rest = [d for d in successful if d not in preferred]

        return _sort_by_trust(preferred + rest)

    def retrieve_evidence_items(
        self,
        items: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        founder_items = [i for i in items if i.domain == "founder"]
        other_items = [i for i in items if i.domain != "founder"]

        return sorted(founder_items + other_items, key=lambda i: i.relevance_score, reverse=True)

    def get_keywords(self) -> list[str]:
        return self._DOMAIN_KEYWORDS

    def get_document_types(self) -> list[DocumentType]:
        return self._PREFERRED_DOC_TYPES


class TechnologyRetrievalStrategy(RetrievalStrategy):
    """Retrieves evidence relevant to technology intelligence extraction.

    Prioritizes documentation pages, API docs, repository pages, and
    product pages — documents that contain technology stack details.
    """

    _DOMAIN_KEYWORDS: list[str] = [
        "technology", "tech stack", "framework", "library", "api",
        "database", "cloud", "aws", "gcp", "azure", "kubernetes",
        "docker", "python", "javascript", "typescript", "react",
        "machine learning", "ai", "infrastructure", "architecture",
        "microservices", "serverless", "open source", "github",
        "developer", "sdk", "cli", "pipeline", "ci/cd",
    ]

    _PREFERRED_DOC_TYPES: list[DocumentType] = [
        DocumentType.DOCUMENTATION,
        DocumentType.API_DOCS,
        DocumentType.REPOSITORY,
        DocumentType.PRODUCT,
    ]

    def retrieve_documents(
        self,
        bundle: EvidenceBundle,
    ) -> list[EvidenceDocument]:
        from predictron_engine.evidence.models import DocumentStatus

        successful = [
            doc for doc in bundle.documents
            if doc.status == DocumentStatus.SUCCESS
        ]

        preferred = _filter_by_keywords(successful, self._DOMAIN_KEYWORDS)
        rest = [d for d in successful if d not in preferred]

        return _sort_by_trust(preferred + rest)

    def retrieve_evidence_items(
        self,
        items: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        tech_items = [i for i in items if i.domain == "technology"]
        other_items = [i for i in items if i.domain != "technology"]

        return sorted(tech_items + other_items, key=lambda i: i.relevance_score, reverse=True)

    def get_keywords(self) -> list[str]:
        return self._DOMAIN_KEYWORDS

    def get_document_types(self) -> list[DocumentType]:
        return self._PREFERRED_DOC_TYPES


class BusinessModelRetrievalStrategy(RetrievalStrategy):
    """Retrieves evidence relevant to business model extraction.

    Prioritizes pricing pages, about pages, and blog posts —
    documents that contain revenue, pricing, and business model info.
    """

    _DOMAIN_KEYWORDS: list[str] = [
        "saas", "subscription", "pricing", "revenue", "mrr", "arr",
        "marketplace", "commission", "licensing", "freemium",
        "business model", "monetization", "customer acquisition",
        "sales", "enterprise contract", "per-seat", "usage-based",
        "take rate", "gmv", "network effect", "platform",
    ]

    _PREFERRED_DOC_TYPES: list[DocumentType] = [
        DocumentType.PRICING,
        DocumentType.ABOUT,
        DocumentType.BLOG,
        DocumentType.HOMEPAGE,
    ]

    def retrieve_documents(
        self,
        bundle: EvidenceBundle,
    ) -> list[EvidenceDocument]:
        from predictron_engine.evidence.models import DocumentStatus

        successful = [
            doc for doc in bundle.documents
            if doc.status == DocumentStatus.SUCCESS
        ]

        preferred = _filter_by_keywords(successful, self._DOMAIN_KEYWORDS)
        rest = [d for d in successful if d not in preferred]

        return _sort_by_trust(preferred + rest)

    def retrieve_evidence_items(
        self,
        items: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        bm_items = [i for i in items if i.domain == "business_model"]
        other_items = [i for i in items if i.domain != "business_model"]

        return sorted(bm_items + other_items, key=lambda i: i.relevance_score, reverse=True)

    def get_keywords(self) -> list[str]:
        return self._DOMAIN_KEYWORDS

    def get_document_types(self) -> list[DocumentType]:
        return self._PREFERRED_DOC_TYPES


class ProductRetrievalStrategy(RetrievalStrategy):
    """Retrieves evidence relevant to product intelligence extraction.

    Prioritizes product pages, documentation pages, and homepage —
    documents that describe product capabilities and features.
    """

    _DOMAIN_KEYWORDS: list[str] = [
        "product", "platform", "feature", "capability", "deployment",
        "cloud", "saas", "api", "integration", "automation",
        "analytics", "compliance", "security", "mobile", "web app",
        "enterprise", "workflow", "real-time", "ai-powered", "diagnostic",
        "model serving", "carbon accounting", "robotics",
    ]

    _PREFERRED_DOC_TYPES: list[DocumentType] = [
        DocumentType.PRODUCT,
        DocumentType.DOCUMENTATION,
        DocumentType.API_DOCS,
        DocumentType.HOMEPAGE,
    ]

    def retrieve_documents(
        self,
        bundle: EvidenceBundle,
    ) -> list[EvidenceDocument]:
        from predictron_engine.evidence.models import DocumentStatus

        successful = [
            doc for doc in bundle.documents
            if doc.status == DocumentStatus.SUCCESS
        ]

        preferred = _filter_by_keywords(successful, self._DOMAIN_KEYWORDS)
        rest = [d for d in successful if d not in preferred]

        return _sort_by_trust(preferred + rest)

    def retrieve_evidence_items(
        self,
        items: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        product_items = [i for i in items if i.domain == "product"]
        other_items = [i for i in items if i.domain != "product"]

        return sorted(product_items + other_items, key=lambda i: i.relevance_score, reverse=True)

    def get_keywords(self) -> list[str]:
        return self._DOMAIN_KEYWORDS

    def get_document_types(self) -> list[DocumentType]:
        return self._PREFERRED_DOC_TYPES


_STRATEGY_MAP: dict[str, RetrievalStrategy] = {
    "market": MarketRetrievalStrategy(),
    "founder": FounderRetrievalStrategy(),
    "technology": TechnologyRetrievalStrategy(),
    "business_model": BusinessModelRetrievalStrategy(),
    "product": ProductRetrievalStrategy(),
}


def get_strategy_for_domain(domain: str) -> RetrievalStrategy | None:
    """Return the retrieval strategy for a given domain, or None."""
    return _STRATEGY_MAP.get(domain)


def get_all_strategies() -> dict[str, RetrievalStrategy]:
    """Return a copy of all registered domain strategies."""
    return dict(_STRATEGY_MAP)
