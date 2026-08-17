"""Website evidence provider — collects structured evidence from a company website.

Encapsulates the website-specific pipeline (discover candidate pages,
fetch them concurrently, clean the raw HTML) behind the
:class:`EvidenceProvider` interface so the orchestrator and the rest of
the pipeline never depend on website details. Collection is best-effort:
individual page failures are recorded as provenance and never abort the
run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime

from pydantic import BaseModel, Field, HttpUrl

from predictron_engine.evidence.cleaner import HtmlCleaner
from predictron_engine.evidence.discover import DefaultPageDiscoverer
from predictron_engine.evidence.fetcher import HttpPageFetcher
from predictron_engine.evidence.interfaces import HtmlCleaner as HtmlCleanerProtocol
from predictron_engine.evidence.interfaces import PageDiscoverer, PageFetcher
from predictron_engine.evidence.models import (
    DocumentStatus,
    EvidenceDocument,
    EvidenceSource,
    FetchResult,
    PageCandidate,
    make_document_id,
)
from predictron_engine.evidence.provider_contracts import CollectContext, ProviderResult
from predictron_engine.evidence.url_utils import normalise_website

logger = logging.getLogger(__name__)


class WebsiteProviderSettings(BaseModel):
    """Settings for the :class:`WebsiteEvidenceProvider`."""

    max_concurrency: int = Field(default=8, ge=1, description="Maximum concurrent page fetches")


class WebsiteEvidenceProvider:
    """Collects structured evidence from a company website.

    Parameters
    ----------
    discoverer:
        Optional page discoverer; defaults to :class:`DefaultPageDiscoverer`.
    fetcher:
        Optional page fetcher; defaults to :class:`HttpPageFetcher`.
    cleaner:
        Optional HTML cleaner; defaults to :class:`HtmlCleaner`.
    settings:
        Optional provider settings.
    """

    name = "website"

    def __init__(
        self,
        *,
        discoverer: PageDiscoverer | None = None,
        fetcher: PageFetcher | None = None,
        cleaner: HtmlCleanerProtocol | None = None,
        settings: WebsiteProviderSettings | None = None,
    ) -> None:
        self._discoverer = discoverer or DefaultPageDiscoverer()
        self._fetcher = fetcher or HttpPageFetcher()
        self._cleaner = cleaner or HtmlCleaner()
        self._settings = settings or WebsiteProviderSettings()

    def can_collect(self, context: CollectContext) -> bool:
        """Return True when a website is available to collect."""
        return bool(context.website)

    async def collect(self, context: CollectContext) -> ProviderResult:
        """Collect website evidence for the company in ``context``.

        Never raises for page-level failures; returns a
        :class:`ProviderResult` with partial results and full provenance.
        Raises :class:`InvalidWebsiteError` for an invalid website URL.
        """
        started = time.monotonic()
        website_url = self._normalize_website(context.website)
        sources: list[EvidenceSource] = []
        documents: list[EvidenceDocument] = []

        logger.info(
            "Website evidence provider starting for %s (%s)",
            context.startup_name,
            website_url,
        )
        try:
            candidates = self._discoverer.discover(website_url)
        except Exception as exc:  # noqa: BLE001 - collection must never crash the pipeline
            logger.exception("Page discovery failed for %s: %s", website_url, exc)
            candidates = []

        logger.info("Discovered %d candidate pages for %s", len(candidates), website_url)
        semaphore = asyncio.Semaphore(self._settings.max_concurrency)

        async def collect_candidate(candidate: PageCandidate) -> None:
            async with semaphore:
                await self._collect_page(candidate, sources, documents)

        await asyncio.gather(
            *(collect_candidate(candidate) for candidate in candidates),
            return_exceptions=True,
        )

        documents.sort(key=lambda doc: str(doc.original_url))
        sources.sort(key=lambda src: str(src.original_url))

        duration_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "Website evidence provider completed: %d documents from %d attempted pages in %d ms",
            len(documents),
            len(candidates),
            duration_ms,
        )
        return ProviderResult(
            provider=self.name,
            website=website_url,
            documents=documents,
            sources=sources,
            attempted_pages=len(candidates),
            duration_ms=duration_ms,
            success=True,
        )

    async def _collect_page(
        self,
        candidate: PageCandidate,
        sources: list[EvidenceSource],
        documents: list[EvidenceDocument],
    ) -> None:
        try:
            logger.info("Collecting %s page: %s", candidate.page_type.value, candidate.url)
            fetch = await self._fetcher.fetch(candidate.url)
        except Exception as exc:  # noqa: BLE001 - a broken fetcher must not kill the run
            logger.warning("Fetch failed for %s: %s", candidate.url, exc)
            sources.append(self._source(candidate, error=f"{type(exc).__name__}: {exc}"))
            return

        sources.append(self._source(candidate, fetch=fetch))

        if not fetch.ok:
            logger.info("%s page returned HTTP %d", candidate.page_type.value.title(), fetch.status)
            return

        try:
            cleaned = self._cleaner.clean(fetch.html or "", url=str(fetch.final_url))
        except Exception as exc:  # noqa: BLE001 - cleaning failure must not kill the run
            logger.warning("Cleaning failed for %s: %s", candidate.url, exc)
            documents.append(
                EvidenceDocument(
                    id=make_document_id(str(fetch.final_url)),
                    original_url=candidate.url,
                    url=fetch.final_url,
                    page_type=candidate.page_type,
                    status=DocumentStatus.FAILED,
                    fetched_at=datetime.now(UTC),
                    response_time_ms=fetch.response_time_ms,
                    http_status=fetch.status,
                    content_type=fetch.content_type,
                    truncated=fetch.truncated,
                    error=f"Cleaning failed: {exc}",
                )
            )
            return

        status = DocumentStatus.SUCCESS if not cleaned.empty else DocumentStatus.EMPTY
        if cleaned.empty:
            logger.warning("No meaningful text extracted from %s", candidate.url)
        else:
            logger.info(
                "Collected %s page (%d chars)",
                candidate.page_type.value.title(),
                len(cleaned.text),
            )

        documents.append(
            EvidenceDocument(
                id=make_document_id(str(fetch.final_url)),
                original_url=candidate.url,
                url=fetch.final_url,
                page_type=candidate.page_type,
                title=cleaned.title,
                text=cleaned.text,
                status=status,
                fetched_at=datetime.now(UTC),
                response_time_ms=fetch.response_time_ms,
                http_status=fetch.status,
                content_type=fetch.content_type,
                truncated=fetch.truncated,
            )
        )

    def _source(
        self,
        candidate: PageCandidate,
        fetch: FetchResult | None = None,
        *,
        error: str | None = None,
    ) -> EvidenceSource:
        if fetch is None:
            return EvidenceSource(
                original_url=candidate.url,
                page_type=candidate.page_type,
                fetched_at=datetime.now(UTC),
                success=False,
                error=error or "Fetch failed",
            )
        return EvidenceSource(
            original_url=candidate.url,
            url=fetch.final_url,
            page_type=candidate.page_type,
            http_status=fetch.status,
            response_time_ms=fetch.response_time_ms,
            fetched_at=datetime.now(UTC),
            success=fetch.ok,
            error=fetch.error if not fetch.ok else None,
        )

    @staticmethod
    def _normalize_website(website: HttpUrl | str | None) -> HttpUrl:
        return normalise_website(website)

    async def close(self) -> None:
        """Release any resources held by the provider's dependencies."""
        close = getattr(self._fetcher, "close", None)
        if close is not None:
            await close()

    async def __aenter__(self) -> WebsiteEvidenceProvider:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()
