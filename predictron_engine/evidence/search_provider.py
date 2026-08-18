"""Search evidence provider — discovers candidate URLs via a pluggable search backend.

The provider queries a configurable :class:`SearchBackend` to discover
high-quality candidate URLs for a company across the public internet,
scores and ranks them using deterministic heuristics, identifies the
official website, prioritises evidence quality, and selects only the
highest-value pages for downstream fetching.

Design constraints
------------------
* **No fetching.**  This provider never retrieves HTML or creates
  :class:`EvidenceDocument` objects.  Its responsibility ends at
  producing prioritised candidate URLs and provider diagnostics.
* **No hardcoding.**  The actual search backend is injected; the
  ``backend`` name in :class:`SearchSettings` is purely a diagnostics
  label.
* **Deterministic.**  Identical inputs always produce identical outputs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from datetime import UTC, datetime

from pydantic import HttpUrl, ValidationError

from predictron_engine.evidence.models import EvidenceSource, PageType
from predictron_engine.evidence.prioritization import (
    PrioritizationSettings,
    PrioritizedPage,
    identify_official_website,
    prioritise_pages,
    select_pages_for_fetch,
)
from predictron_engine.evidence.provider_contracts import (
    CollectContext,
    PrioritizationSummary,
    ProviderResult,
)
from predictron_engine.evidence.ranking import rank_urls
from predictron_engine.evidence.search_interfaces import SearchBackend, SearchSettings
from predictron_engine.evidence.url_utils import extract_host, parse_http_url

logger = logging.getLogger(__name__)


class SearchEvidenceProvider:
    """Discovers and prioritises candidate URLs via a pluggable search backend.

    Parameters
    ----------
    backend:
        A :class:`SearchBackend` implementation to query.
    settings:
        Optional configuration (max results, timeouts, retry policy).
    prioritization_settings:
        Optional prioritization configuration (official website confidence,
        max fetch pages).
    """

    name = "search"

    def __init__(
        self,
        *,
        backend: SearchBackend | None = None,
        settings: SearchSettings | None = None,
        prioritization_settings: PrioritizationSettings | None = None,
    ) -> None:
        self._backend: SearchBackend | None = backend
        self._settings = settings or SearchSettings()
        self._prioritization_settings = prioritization_settings or PrioritizationSettings()

    # ------------------------------------------------------------------
    # EvidenceProvider protocol
    # ------------------------------------------------------------------

    def can_collect(self, context: CollectContext) -> bool:
        """Return True when a search backend is available."""
        return self._backend is not None

    async def collect(self, context: CollectContext) -> ProviderResult:
        """Discover, prioritise, and select candidate URLs for *context.startup_name*.

        Returns a :class:`ProviderResult` whose ``sources`` carry only
        the highest-value prioritised URLs, and whose ``documents`` list
        is always empty (this provider does not fetch pages).
        """
        started = time.monotonic()

        if self._backend is None:
            return self._empty_result(
                reason="No search backend configured",
                started=started,
            )

        website_host = self._extract_website_host(context.website)

        # Build one or more search queries
        queries = self._build_queries(context.startup_name)
        all_results = []

        for query in queries:
            try:
                results = await asyncio.wait_for(
                    self._backend.search(query, self._settings.max_results),
                    timeout=self._settings.timeout,
                )
                all_results.extend(results)
            except TimeoutError:
                logger.warning(
                    "Search backend timed out for query %r after %.1fs",
                    query,
                    self._settings.timeout,
                )
                break
            except Exception as exc:  # noqa: BLE001 — provider must not crash
                logger.warning("Search backend failed for query %r: %s", query, exc)
                if self._settings.retry_count > 0:
                    all_results = await self._retry(query, all_results)
                    if all_results is None:
                        return self._failure_result(
                            reason=f"Backend failed after retries: {exc}",
                            started=started,
                        )
                    break
                return self._failure_result(
                    reason=f"Backend failed: {exc}",
                    started=started,
                )

        # Rank, deduplicate, and filter
        ranked = rank_urls(
            all_results,
            website_host=website_host,
        )

        # Build title/snippet lookup for prioritization
        search_map: dict[str, tuple[str, str]] = {}
        for r in all_results:
            if r.url not in search_map:
                search_map[r.url] = (r.title, r.snippet)

        # Identify official website
        official = identify_official_website(
            ranked,
            startup_name=context.startup_name,
            known_website_host=website_host,
        )

        # Prioritise pages
        prioritised = prioritise_pages(
            ranked,
            search_results=search_map,
            official_host=extract_host(official.url) if official.url else None,
            settings=self._prioritization_settings,
        )

        # Select pages for fetch
        selected = select_pages_for_fetch(
            prioritised,
            settings=self._prioritization_settings,
        )

        # Build provenance sources from selected pages only
        sources = self._build_sources(selected)

        # Build diagnostics
        skipped_count = len(prioritised) - len(selected)
        evidence_counts: Counter[str] = Counter()
        for page in selected:
            evidence_counts[page.evidence_type] += 1

        prioritization_summary = PrioritizationSummary(
            detected_official_url=official.url,
            official_confidence=official.confidence,
            selected_count=len(selected),
            skipped_count=max(0, skipped_count),
            evidence_types=dict(sorted(evidence_counts.items())),
        )

        duration_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "Search evidence provider completed: %d candidates discovered, "
            "%d prioritised, %d selected for %s in %d ms",
            len(all_results),
            len(prioritised),
            len(selected),
            context.startup_name,
            duration_ms,
        )
        return ProviderResult(
            provider=self.name,
            website=self._parse_website(context.website),
            documents=[],
            sources=sources,
            attempted_pages=1,
            duration_ms=duration_ms,
            success=True,
            prioritization=prioritization_summary,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_website_host(website: str | None) -> str | None:
        """Extract a clean hostname from a raw website string."""
        if not website:
            return None
        host = extract_host(website)
        return host if host else None

    @staticmethod
    def _parse_website(website: str | None) -> HttpUrl | None:
        """Parse and return an :class:`HttpUrl` if possible."""
        if not website:
            return None
        return parse_http_url(website)

    @staticmethod
    def _build_queries(startup_name: str) -> list[str]:
        """Build search queries for a startup name.

        Returns a list so callers can fan out across multiple queries in
        the future.  The primary query is always first.
        """
        return [f"{startup_name}"]

    async def _retry(
        self,
        query: str,
        previous_results: list,
    ) -> list | None:
        """Retry a failed query per the configured retry policy."""
        for attempt in range(self._settings.retry_count):
            try:
                await asyncio.sleep(self._settings.retry_delay)
                results = await asyncio.wait_for(
                    self._backend.search(query, self._settings.max_results),  # type: ignore[union-attr]
                    timeout=self._settings.timeout,
                )
                return results
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Retry %d/%d failed for query %r: %s",
                    attempt + 1,
                    self._settings.retry_count,
                    query,
                    exc,
                )
        return None

    @staticmethod
    def _build_sources(pages: list[PrioritizedPage]) -> list[EvidenceSource]:
        """Convert prioritised pages into provenance :class:`EvidenceSource` records."""
        now = datetime.now(UTC)
        sources: list[EvidenceSource] = []
        for page in pages:
            try:
                url = HttpUrl(page.url)
            except (ValidationError, Exception):  # noqa: BLE001
                logger.debug("Skipping malformed URL in prioritised results: %s", page.url)
                continue
            sources.append(
                EvidenceSource(
                    original_url=url,
                    url=url,
                    page_type=PageType.UNKNOWN,
                    fetched_at=now,
                    success=True,
                )
            )
        return sources

    def _empty_result(
        self,
        *,
        reason: str,
        started: float,
    ) -> ProviderResult:
        """Return a successful but empty result."""
        return ProviderResult(
            provider=self.name,
            documents=[],
            sources=[],
            attempted_pages=0,
            duration_ms=int((time.monotonic() - started) * 1000),
            success=True,
            failure_reason=reason,
        )

    def _failure_result(
        self,
        *,
        reason: str,
        started: float,
    ) -> ProviderResult:
        """Return a failed result."""
        return ProviderResult(
            provider=self.name,
            documents=[],
            sources=[],
            attempted_pages=0,
            duration_ms=int((time.monotonic() - started) * 1000),
            success=False,
            failure_reason=reason,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release resources held by the search backend."""
        if self._backend is not None:
            close = getattr(self._backend, "close", None)
            if close is not None:
                await close()

    async def __aenter__(self) -> SearchEvidenceProvider:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()
