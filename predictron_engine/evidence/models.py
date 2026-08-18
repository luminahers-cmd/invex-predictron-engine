"""Pydantic models for the Evidence Collection Layer.

These models are intentionally free of any reasoning, scoring, or domain
semantics. They represent raw and cleaned web content together with the
provenance metadata required to trace any downstream fact back to its
original source (URL, retrieval timestamp, page type, HTTP status, and
retrieval method).

The collection subsystem is deterministic: document ids are derived from
the final URL using UUIDv5, so re-collecting the same page always yields
the same id.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl, computed_field

# ── Document Intelligence types (Sprint 4C) ────────────────────────


class DocumentType(str, Enum):
    """Classification of an evidence document's content type."""

    HOMEPAGE = "homepage"
    ABOUT = "about"
    PRODUCT = "product"
    PRICING = "pricing"
    DOCUMENTATION = "documentation"
    API_DOCS = "api_docs"
    BLOG = "blog"
    CAREERS = "careers"
    SECURITY = "security"
    PRIVACY = "privacy"
    TERMS = "terms"
    FAQ = "faq"
    CONTACT = "contact"
    NEWS = "news"
    PRESS_RELEASE = "press_release"
    INVESTOR = "investor"
    REPOSITORY = "repository"
    UNKNOWN = "unknown"


class DocumentMetadata(BaseModel):
    """Intelligence metadata attached to an :class:`EvidenceDocument`.

    All fields have defaults so the model can be constructed incrementally
    and remains backward-compatible with existing consumers.
    """

    document_type: DocumentType = Field(
        default=DocumentType.UNKNOWN,
        description="Classified document type",
    )
    authority_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Authority confidence [0, 1]",
    )
    quality_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Content quality score [0, 1]",
    )
    priority: int = Field(
        default=6, ge=1, le=6,
        description="Priority tier (1 = highest)",
    )
    canonical_url: str | None = Field(
        default=None,
        description="Canonical URL after deduplication",
    )
    language: str = Field(
        default="en",
        description="Detected ISO 639-1 language code",
    )
    word_count: int = Field(default=0, description="Word count of cleaned text")
    heading_count: int = Field(default=0, description="Number of headings")
    table_count: int = Field(default=0, description="Number of table rows")
    list_count: int = Field(default=0, description="Number of list items")
    content_hash: str = Field(
        default="",
        description="SHA-256 hex digest of normalised text",
    )
    duplicate_of: str | None = Field(
        default=None,
        description="Document id of the canonical document when this is a duplicate",
    )
    is_duplicate: bool = Field(
        default=False,
        description="True when this document is a duplicate of another",
    )
    trust_level: str = Field(
        default="unknown",
        description="Trust classification: official, third_party, unknown",
    )
    source_provider: str = Field(
        default="",
        description="Name of the provider that collected this document",
    )


class IntelligenceSummary(BaseModel):
    """Diagnostics for the Document Intelligence stage."""

    documents_input: int = Field(default=0, description="Documents before intelligence")
    documents_classified: int = Field(default=0, description="Documents classified")
    duplicates_removed: int = Field(default=0, description="Duplicate documents removed")
    average_authority: float = Field(default=0.0, description="Mean authority score")
    average_quality: float = Field(default=0.0, description="Mean quality score")
    document_type_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Count per document type",
    )
    processing_duration_ms: int = Field(default=0, description="Processing wall-clock time")
    classification_confidence: float = Field(
        default=0.0,
        description="Fraction of documents classified as non-UNKNOWN",
    )


def make_document_id(url: str) -> str:
    """Return a stable UUIDv5 document id derived from a URL."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, url))


class PageType(str, Enum):
    """Semantic classification of a page discovered on a company website."""

    HOMEPAGE = "homepage"
    ABOUT = "about"
    COMPANY = "company"
    PRODUCTS = "products"
    SERVICES = "services"
    PLATFORM = "platform"
    TECHNOLOGY = "technology"
    UNKNOWN = "unknown"


class RetrievalMethod(str, Enum):
    """Transport used to retrieve a page."""

    HTTP_GET = "http_get"


class DocumentStatus(str, Enum):
    """Outcome for a single document within an EvidenceBundle."""

    SUCCESS = "success"
    EMPTY = "empty"  # fetched but no meaningful text was extractable
    FAILED = "failed"  # fetched but cleaning failed


class PageCandidate(BaseModel):
    """A URL to attempt collection for, with its semantic classification."""

    url: HttpUrl = Field(..., description="Absolute URL to fetch")
    page_type: PageType = Field(..., description="Semantic classification of the page")
    path: str = Field(..., description="Site path used to derive this candidate")


class FetchResult(BaseModel):
    """Raw HTTP retrieval result, before any cleaning is applied."""

    requested_url: HttpUrl = Field(..., description="URL that was requested")
    final_url: HttpUrl = Field(..., description="URL after any redirects")
    status: int = Field(..., description="HTTP status code (0 if transport failure)")
    response_time_ms: int = Field(..., description="End-to-end latency in milliseconds")
    html: str | None = Field(default=None, description="Raw HTML body when fetch succeeded")
    content_type: str | None = Field(default=None, description="Content-Type header when available")
    truncated: bool = Field(default=False, description="True if the body exceeded the size cap")
    retry_count: int = Field(default=0, description="Number of retries performed")
    error: str | None = Field(default=None, description="Human-readable failure description")

    @property
    def ok(self) -> bool:
        """True when usable HTML was retrieved."""
        return self.error is None and self.html is not None


class CleanResult(BaseModel):
    """Cleaned, normalized textual content extracted from raw HTML."""

    title: str = Field(default="", description="Document title or best-effort heading")
    text: str = Field(default="", description="Whitespace-normalized readable text")
    headings: list[str] = Field(default_factory=list, description="Extracted headings")
    paragraphs: list[str] = Field(default_factory=list, description="Extracted paragraphs")
    list_items: list[str] = Field(default_factory=list, description="Extracted list items")
    table_rows: list[str] = Field(default_factory=list, description="Extracted table rows")
    word_count: int = Field(default=0, description="Number of words in the cleaned text")
    empty: bool = Field(default=False, description="True when no meaningful text was found")


class EvidenceDocument(BaseModel):
    """One cleaned page of evidence with full provenance metadata."""

    id: str = Field(..., description="Stable UUIDv5 id derived from the final URL")
    original_url: HttpUrl = Field(..., description="URL that was requested")
    url: HttpUrl = Field(..., description="Final URL after redirects")
    page_type: PageType = Field(..., description="Semantic classification of the page")
    status: DocumentStatus = Field(..., description="Collection outcome")
    fetched_at: datetime = Field(..., description="UTC timestamp of retrieval")
    response_time_ms: int = Field(..., description="HTTP latency in milliseconds")
    http_status: int | None = Field(..., description="HTTP status code")
    title: str = Field(default="", description="Document title")
    text: str = Field(default="", description="Cleaned, normalized readable text")
    retrieval_method: RetrievalMethod = Field(
        default=RetrievalMethod.HTTP_GET, description="Transport used"
    )
    content_type: str | None = Field(default=None, description="Response Content-Type")
    truncated: bool = Field(default=False, description="True if body exceeded the size cap")
    error: str | None = Field(default=None, description="Failure description if any")
    metadata: DocumentMetadata | None = Field(
        default=None,
        description="Document Intelligence metadata (populated after enrichment)",
    )


class EvidenceSource(BaseModel):
    """Provenance record for one attempted retrieval (success or failure).

    A source is recorded for every candidate page whether it fetched
    successfully or not, guaranteeing that every failure is preserved in
    the bundle for downstream traceability.
    """

    original_url: HttpUrl = Field(..., description="URL that was requested")
    page_type: PageType = Field(..., description="Semantic classification of the page")
    fetched_at: datetime = Field(..., description="UTC timestamp of the attempt")
    success: bool = Field(..., description="True when usable HTML was retrieved")
    url: HttpUrl | None = Field(
        default=None, description="Final URL after redirects, when available"
    )
    http_status: int | None = Field(default=None, description="HTTP status code, when available")
    response_time_ms: int = Field(default=0, description="HTTP latency in milliseconds")
    retrieval_method: RetrievalMethod = Field(
        default=RetrievalMethod.HTTP_GET, description="Transport used"
    )
    error: str | None = Field(default=None, description="Failure description if any")


class ProviderRun(BaseModel):
    """Diagnostic record of one evidence provider's collection attempt."""

    provider: str = Field(..., description="Name of the evidence provider")
    duration_ms: int = Field(default=0, description="Wall-clock time spent by the provider")
    success: bool = Field(default=True, description="Whether the provider completed successfully")
    documents: int = Field(default=0, description="Number of documents the provider produced")
    attempted_pages: int = Field(
        default=0, description="Number of sources the provider attempted"
    )
    failure_reason: str | None = Field(
        default=None, description="Why the provider failed, if it did"
    )


class EvidenceBundle(BaseModel):
    """Complete output of one evidence collection run.

    ``website`` is optional so an empty bundle can represent the
    "no website provided" case without fabricating a URL.
    """

    startup_name: str = Field(..., description="Company name")
    website: HttpUrl | None = Field(
        default=None, description="Website URL that was collected, if any"
    )
    documents: list[EvidenceDocument] = Field(
        default_factory=list, description="Collected documents"
    )
    sources: list[EvidenceSource] = Field(
        default_factory=list, description="Provenance for every attempt"
    )
    collected_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="UTC completion timestamp"
    )
    duration_ms: int = Field(default=0, description="Total wall-clock time of the run")
    attempted_pages: int = Field(default=0, description="Number of candidate pages attempted")
    providers: list[ProviderRun] = Field(
        default_factory=list, description="Per-provider diagnostic records"
    )
    intelligence: IntelligenceSummary | None = Field(
        default=None,
        description="Document Intelligence diagnostics (populated after enrichment)",
    )

    @classmethod
    def empty(cls, startup_name: str) -> EvidenceBundle:
        """Return an empty bundle for the "no website" or "collection skipped" cases."""
        return cls(startup_name=startup_name, website=None)

    @computed_field
    @property
    def total_pages(self) -> int:
        """Number of collected documents (pages retrieved and processed)."""
        return len(self.documents)

    @property
    def failures(self) -> list[EvidenceSource]:
        """Provenance records for every failed retrieval attempt."""
        return [source for source in self.sources if not source.success]
