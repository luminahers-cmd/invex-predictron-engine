"""Structural contracts (ports) for the Evidence Collection Layer.

The collector depends on these protocols so that each stage — discovery,
fetching, and cleaning — can be swapped for testing, caching, or richer
implementations (e.g. link-based crawling) without modifying the
orchestration service.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import HttpUrl

from predictron_engine.evidence.models import CleanResult, FetchResult, PageCandidate


@runtime_checkable
class PageDiscoverer(Protocol):
    """Discovers candidate pages for a company website."""

    def discover(self, website: HttpUrl) -> list[PageCandidate]:
        """Return the ordered list of pages to attempt collection for."""
        ...


@runtime_checkable
class PageFetcher(Protocol):
    """Retrieves raw HTML for a single page."""

    async def fetch(self, url: HttpUrl) -> FetchResult:
        """Fetch a page and return a FetchResult; never raises for page failures."""
        ...


@runtime_checkable
class HtmlCleaner(Protocol):
    """Converts raw HTML into normalized, readable text."""

    def clean(self, html: str, *, url: str | None = None) -> CleanResult:
        """Clean raw HTML and return normalized text content."""
        ...
