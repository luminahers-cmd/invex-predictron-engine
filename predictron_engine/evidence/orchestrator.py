"""Orchestration service for the Evidence Collection Layer.

Runs every applicable evidence provider concurrently, merges their
documents and provenance into a single deterministic
:class:`EvidenceBundle`, and records per-provider diagnostics.
Providers are isolated: a failing provider is recorded and skipped
without aborting the run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import HttpUrl

from predictron_engine.evidence.document_intelligence import (
    enrich_documents,
)
from predictron_engine.evidence.models import (
    EvidenceBundle,
    EvidenceDocument,
    EvidenceSource,
    IntelligenceSummary,
    ProviderRun,
)
from predictron_engine.evidence.provider_contracts import (
    CollectContext,
    EvidenceProvider,
    ProviderResult,
)
from predictron_engine.evidence.website_provider import WebsiteEvidenceProvider

logger = logging.getLogger(__name__)


def _run_from_result(result: ProviderResult) -> ProviderRun:
    """Derive a lightweight diagnostic record from a provider result."""
    return ProviderRun(
        provider=result.provider,
        duration_ms=result.duration_ms,
        success=result.success,
        documents=len(result.documents),
        attempted_pages=result.attempted_pages,
        failure_reason=result.failure_reason,
    )


class EvidenceOrchestrator:
    """Runs evidence providers and assembles a single EvidenceBundle.

    All applicable providers execute concurrently via
    :func:`asyncio.gather`.  The final output is deterministically ordered
    by the original provider sequence regardless of completion order.

    After provider collection, Document Intelligence enriches every
    document with classification, quality metrics, authority scores,
    and duplicate resolution.

    Parameters
    ----------
    providers:
        Optional ordered sequence of evidence providers. Defaults to a
        single :class:`WebsiteEvidenceProvider`.
    official_host:
        Optional lowercased hostname for official website identification.
    """

    def __init__(
        self,
        providers: Sequence[EvidenceProvider] | None = None,
        official_host: str | None = None,
    ) -> None:
        self._providers = (
            list(providers) if providers is not None else [WebsiteEvidenceProvider()]
        )
        self._official_host = official_host

    async def collect(
        self,
        startup_name: str,
        website: str | HttpUrl | None,
    ) -> EvidenceBundle:
        """Collect evidence from every applicable provider concurrently.

        Returns an :class:`EvidenceBundle` merging all provider results.
        Provider failures (including invalid website URLs) are isolated
        and recorded in diagnostics; they never abort the run.
        """
        started = time.monotonic()
        context = CollectContext(
            startup_name=startup_name,
            website=str(website) if website else None,
        )
        collected_at = datetime.now(UTC)

        logger.info("Evidence collection starting for %s", startup_name)

        # Phase 1: determine which providers are applicable (sync, fast)
        applicable: list[EvidenceProvider] = []
        for provider in self._providers:
            try:
                is_applicable = provider.can_collect(context)
            except Exception:  # noqa: BLE001 - a broken can_collect must not kill the run
                logger.exception(
                    "Provider %s can_collect failed for %s", provider.name, startup_name
                )
                continue
            if is_applicable:
                applicable.append(provider)
            else:
                logger.info("Provider %s skipped for %s", provider.name, startup_name)

        # Phase 2: run all applicable providers concurrently
        logger.info(
            "Running %d provider(s) concurrently for %s",
            len(applicable),
            startup_name,
        )
        raw_results: list[object] = list(
            await asyncio.gather(
                *(
                    self._safe_collect(provider, context, startup_name)
                    for provider in applicable
                ),
                return_exceptions=True,
            )
        )

        # Phase 3: merge results in deterministic provider order
        documents: list[EvidenceDocument] = []
        sources: list[EvidenceSource] = []
        provider_runs: list[ProviderRun] = []
        attempted_pages = 0
        bundle_website: HttpUrl | None = None

        for provider, raw in zip(applicable, raw_results, strict=True):
            if isinstance(raw, BaseException):
                logger.exception(
                    "Provider %s failed for %s", provider.name, startup_name
                )
                provider_runs.append(
                    ProviderRun(
                        provider=provider.name,
                        success=False,
                        failure_reason=f"{type(raw).__name__}: {raw}",
                    )
                )
                continue

            result: ProviderResult = raw
            provider_runs.append(_run_from_result(result))
            documents.extend(result.documents)
            sources.extend(result.sources)
            attempted_pages += result.attempted_pages
            if bundle_website is None and result.website is not None:
                bundle_website = result.website

        documents.sort(key=lambda doc: str(doc.original_url))
        sources.sort(key=lambda src: str(src.original_url))

        # Phase 4: Document Intelligence enrichment
        intelligence_summary: IntelligenceSummary | None = None
        if documents:
            logger.info(
                "Running Document Intelligence on %d documents for %s",
                len(documents),
                startup_name,
            )
            documents, intelligence_summary = enrich_documents(
                documents,
                official_host=self._official_host,
                source_provider="evidence",
            )
            logger.info(
                "Document Intelligence completed: %d classified, "
                "%d duplicates removed for %s",
                intelligence_summary.documents_classified,
                intelligence_summary.duplicates_removed,
                startup_name,
            )

        duration_ms = int((time.monotonic() - started) * 1000)
        bundle = EvidenceBundle(
            startup_name=startup_name,
            website=bundle_website,
            documents=documents,
            sources=sources,
            collected_at=collected_at,
            duration_ms=duration_ms,
            attempted_pages=attempted_pages,
            providers=provider_runs,
            intelligence=intelligence_summary,
        )
        logger.info(
            "Evidence bundle completed: %d documents from %d attempted pages in %d ms",
            len(documents),
            attempted_pages,
            duration_ms,
        )
        return bundle

    @staticmethod
    async def _safe_collect(
        provider: EvidenceProvider,
        context: CollectContext,
        startup_name: str,
    ) -> ProviderResult:
        """Run a single provider, converting all exceptions into results."""
        logger.info("Provider %s collecting for %s", provider.name, startup_name)
        try:
            return await provider.collect(context)
        except Exception as exc:  # noqa: BLE001 - provider failures are isolated
            logger.exception("Provider %s failed for %s", provider.name, startup_name)
            return ProviderResult(
                provider=provider.name,
                success=False,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

    async def close(self) -> None:
        """Release resources held by any provider that supports closing."""
        for provider in self._providers:
            close = getattr(provider, "close", None)
            if close is not None:
                await close()

    async def __aenter__(self) -> EvidenceOrchestrator:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()
