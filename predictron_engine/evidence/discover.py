"""Deterministic page discovery for a company website.

Attempts a fixed, ordered set of well-known paths, always starting with
the homepage. Discovery is intentionally deterministic and does not depend
on the homepage content; link-based crawling is a Sprint 2 extension point.
"""

from __future__ import annotations

from urllib.parse import urljoin

from pydantic import HttpUrl

from predictron_engine.evidence.models import PageCandidate, PageType


class DefaultPageDiscoverer:
    """Discover candidate pages using a fixed list of well-known paths."""

    DEFAULT_PATHS: tuple[tuple[str, PageType], ...] = (
        ("", PageType.HOMEPAGE),
        ("about", PageType.ABOUT),
        ("about-us", PageType.ABOUT),
        ("company", PageType.COMPANY),
        ("products", PageType.PRODUCTS),
        ("services", PageType.SERVICES),
        ("platform", PageType.PLATFORM),
        ("technology", PageType.TECHNOLOGY),
    )

    def __init__(self, paths: list[tuple[str, PageType]] | None = None) -> None:
        self._paths = paths if paths is not None else list(self.DEFAULT_PATHS)

    def discover(self, website: HttpUrl) -> list[PageCandidate]:
        """Return ordered, de-duplicated candidate pages for the website."""
        seen: set[str] = set()
        candidates: list[PageCandidate] = []
        base = str(website)
        for path, page_type in self._paths:
            url = HttpUrl(urljoin(base, path))
            key = str(url)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(PageCandidate(url=url, page_type=page_type, path=path))
        return candidates
