"""Tests for the provider-based Evidence Collection Layer architecture.

Covers the EvidenceProvider contract, EvidenceOrchestrator composition,
provider merging, failure isolation, per-provider diagnostics, and
resource cleanup. Website collection itself is exercised with httpx
MockTransport (no network).
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
from pydantic import HttpUrl

from predictron_engine.evidence.fetcher import FetcherSettings, HttpPageFetcher
from predictron_engine.evidence.models import (
    DocumentStatus,
    EvidenceDocument,
    EvidenceSource,
    PageType,
)
from predictron_engine.evidence.orchestrator import EvidenceOrchestrator
from predictron_engine.evidence.provider_contracts import (
    CollectContext,
    EvidenceProvider,
    ProviderResult,
)
from predictron_engine.evidence.website_provider import WebsiteEvidenceProvider

SITE = HttpUrl("https://example.com")

_STATIC_URL = "https://static.example.com/"


def _website_provider(handler) -> WebsiteEvidenceProvider:
    fetcher = HttpPageFetcher(
        settings=FetcherSettings(retry_backoff_ms=0),
        transport=httpx.MockTransport(handler),
    )
    return WebsiteEvidenceProvider(fetcher=fetcher)


def _ok_page(path: str) -> httpx.Response:
    return httpx.Response(
        200,
        text=(
            f"<html><head><title>{path} page</title></head>"
            f"<body><main><h1>{path}</h1><p>Content for {path}.</p></main></body></html>"
        ),
        headers={"content-type": "text/html; charset=utf-8"},
    )


def _ok_handler(request: httpx.Request) -> httpx.Response:
    return _ok_page(request.url.path)


def _static_document() -> EvidenceDocument:
    return EvidenceDocument(
        id="static-1",
        original_url=HttpUrl(_STATIC_URL),
        url=HttpUrl(_STATIC_URL),
        page_type=PageType.UNKNOWN,
        status=DocumentStatus.SUCCESS,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        response_time_ms=5,
        http_status=200,
        text="Static evidence content.",
    )


def _static_source() -> EvidenceSource:
    return EvidenceSource(
        original_url=HttpUrl(_STATIC_URL),
        page_type=PageType.UNKNOWN,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        success=True,
    )


class StaticProvider:
    """A minimal provider that contributes one static document."""

    name = "static"

    def can_collect(self, context: CollectContext) -> bool:
        return True

    async def collect(self, context: CollectContext) -> ProviderResult:
        return ProviderResult(
            provider=self.name,
            documents=[_static_document()],
            sources=[_static_source()],
            attempted_pages=1,
            duration_ms=2,
            success=True,
        )


class FailingProvider:
    """A provider that always fails at collection time."""

    name = "failing"

    def can_collect(self, context: CollectContext) -> bool:
        return True

    async def collect(self, context: CollectContext) -> ProviderResult:
        raise RuntimeError("boom")


class SkippedProvider:
    """A provider that opts out of every collection context."""

    name = "skipped"

    def can_collect(self, context: CollectContext) -> bool:
        return False

    async def collect(self, context: CollectContext) -> ProviderResult:
        raise AssertionError("must not be called")


class ClosableProvider(StaticProvider):
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class TestEvidenceProviderProtocol:
    def test_implementations_are_runtime_checkable(self) -> None:
        assert isinstance(WebsiteEvidenceProvider(), EvidenceProvider)
        assert isinstance(StaticProvider(), EvidenceProvider)

    def test_website_provider_applies_only_when_website_present(self) -> None:
        provider = WebsiteEvidenceProvider()
        assert provider.can_collect(CollectContext(startup_name="X", website="example.com")) is True
        assert provider.can_collect(CollectContext(startup_name="X", website="")) is False
        assert provider.can_collect(CollectContext(startup_name="X")) is False


class TestEvidenceOrchestrator:
    async def test_collects_from_website_provider(self) -> None:
        orchestrator = EvidenceOrchestrator(providers=[_website_provider(_ok_handler)])
        bundle = await orchestrator.collect("ExampleCo", SITE)

        assert bundle.website == SITE
        assert bundle.total_pages == 8
        assert len(bundle.sources) == 8
        assert bundle.attempted_pages == 8
        assert len(bundle.providers) == 1
        assert bundle.providers[0].provider == "website"
        assert bundle.providers[0].documents == 8
        assert bundle.providers[0].attempted_pages == 8
        assert bundle.providers[0].success is True
        assert bundle.providers[0].failure_reason is None
        assert bundle.providers[0].duration_ms >= 0

    async def test_merges_multiple_providers_and_records_metadata(self) -> None:
        orchestrator = EvidenceOrchestrator(
            providers=[_website_provider(_ok_handler), StaticProvider()]
        )
        bundle = await orchestrator.collect("ExampleCo", SITE)

        assert bundle.total_pages == 9
        assert len(bundle.sources) == 9
        assert bundle.attempted_pages == 9
        assert [run.provider for run in bundle.providers] == ["website", "static"]
        assert bundle.providers[1].documents == 1
        assert bundle.providers[1].attempted_pages == 1

    async def test_failing_provider_is_isolated(self) -> None:
        orchestrator = EvidenceOrchestrator(
            providers=[FailingProvider(), _website_provider(_ok_handler)]
        )
        bundle = await orchestrator.collect("ExampleCo", SITE)

        assert bundle.total_pages == 8
        assert len(bundle.providers) == 2
        failed = bundle.providers[0]
        assert failed.provider == "failing"
        assert failed.success is False
        assert failed.documents == 0
        assert "RuntimeError" in (failed.failure_reason or "")
        assert bundle.providers[1].provider == "website"
        assert bundle.providers[1].success is True

    async def test_skipped_provider_is_not_called(self) -> None:
        orchestrator = EvidenceOrchestrator(
            providers=[SkippedProvider(), _website_provider(_ok_handler)]
        )
        bundle = await orchestrator.collect("ExampleCo", SITE)

        assert bundle.total_pages == 8
        assert [run.provider for run in bundle.providers] == ["website"]

    async def test_empty_website_returns_empty_bundle(self) -> None:
        orchestrator = EvidenceOrchestrator(providers=[_website_provider(_ok_handler)])
        bundle = await orchestrator.collect("ExampleCo", "")

        assert bundle.website is None
        assert bundle.total_pages == 0
        assert bundle.providers == []

    async def test_default_orchestrator_skips_without_website(self) -> None:
        orchestrator = EvidenceOrchestrator()
        bundle = await orchestrator.collect("ExampleCo", "")

        assert bundle.website is None
        assert bundle.total_pages == 0
        assert bundle.providers == []

    async def test_default_orchestrator_records_invalid_website(self) -> None:
        orchestrator = EvidenceOrchestrator()
        bundle = await orchestrator.collect("ExampleCo", "not a url")
        assert bundle.providers[0].success is False
        assert "InvalidWebsiteError" in (bundle.providers[0].failure_reason or "")
        assert bundle.total_pages == 0

    async def test_close_releases_provider_resources(self) -> None:
        provider = ClosableProvider()
        orchestrator = EvidenceOrchestrator(providers=[provider])

        async with orchestrator:
            pass

        assert provider.closed is True
