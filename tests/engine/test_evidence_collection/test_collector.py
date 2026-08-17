"""End-to-end tests for the EvidenceOrchestrator orchestration service."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from pydantic import HttpUrl

from predictron_engine.evidence.exceptions import InvalidWebsiteError
from predictron_engine.evidence.fetcher import FetcherSettings, HttpPageFetcher
from predictron_engine.evidence.models import DocumentStatus
from predictron_engine.evidence.orchestrator import EvidenceOrchestrator
from predictron_engine.evidence.website_provider import (
    WebsiteEvidenceProvider,
    WebsiteProviderSettings,
)

SITE = HttpUrl("https://example.com")


def _collector(handler, *, max_concurrency: int = 8) -> EvidenceOrchestrator:
    fetcher = HttpPageFetcher(
        settings=FetcherSettings(retry_backoff_ms=0),
        transport=httpx.MockTransport(handler),
    )
    provider = WebsiteEvidenceProvider(
        fetcher=fetcher, settings=WebsiteProviderSettings(max_concurrency=max_concurrency)
    )
    return EvidenceOrchestrator(providers=[provider])


def _ok_page(path: str) -> httpx.Response:
    return httpx.Response(
        200,
        text=(
            f"<html><head><title>{path} page</title></head>"
            f"<body><main><h1>{path}</h1><p>Content for {path}.</p></main></body></html>"
        ),
        headers={"content-type": "text/html; charset=utf-8"},
    )


ALL_PATHS = {
    "/",
    "/about",
    "/about-us",
    "/company",
    "/products",
    "/services",
    "/platform",
    "/technology",
}


class TestEvidenceOrchestrator:
    async def test_collects_all_default_pages(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _ok_page(request.url.path)

        collector = _collector(handler)
        bundle = await collector.collect("ExampleCo", SITE)

        assert bundle.startup_name == "ExampleCo"
        assert bundle.website == SITE
        assert bundle.total_pages == 8
        assert len(bundle.sources) == 8
        assert all(source.success for source in bundle.sources)
        assert bundle.attempted_pages == 8
        assert any(doc.page_type.value == "homepage" for doc in bundle.documents)
        assert any(doc.page_type.value == "about" for doc in bundle.documents)

    async def test_partial_success_records_failures(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path in ("/products", "/services"):
                return httpx.Response(404)
            if request.url.path == "/platform":
                return httpx.Response(403)
            return _ok_page(request.url.path)

        collector = _collector(handler)
        bundle = await collector.collect("ExampleCo", SITE)

        assert bundle.total_pages == 5
        assert len(bundle.sources) == 8
        assert len(bundle.failures) == 3
        assert {source.http_status for source in bundle.failures} == {404, 403}
        failed_types = {source.page_type.value for source in bundle.failures}
        assert failed_types == {"products", "services", "platform"}

    async def test_all_pages_fail_returns_empty_bundle(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        collector = _collector(handler)
        bundle = await collector.collect("ExampleCo", SITE)

        assert bundle.documents == []
        assert bundle.total_pages == 0
        assert len(bundle.sources) == 8
        assert len(bundle.failures) == 8

    async def test_empty_page_creates_empty_document(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/about":
                return httpx.Response(200, text="<html><body><nav>nav</nav></body></html>")
            return _ok_page(request.url.path)

        collector = _collector(handler)
        bundle = await collector.collect("ExampleCo", SITE)

        about = next(doc for doc in bundle.documents if doc.page_type.value == "about")
        assert about.status == DocumentStatus.EMPTY
        assert about.text == ""

    async def test_deterministic_document_ids_and_order(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _ok_page(request.url.path)

        collector = _collector(handler)
        first = await collector.collect("ExampleCo", SITE)
        second = await collector.collect("ExampleCo", SITE)

        assert [doc.id for doc in first.documents] == [doc.id for doc in second.documents]
        first_urls = [str(doc.original_url) for doc in first.documents]
        second_urls = [str(doc.original_url) for doc in second.documents]
        assert first_urls == second_urls
        assert len(first.documents) == 8

    async def test_requests_all_candidate_paths(self) -> None:
        requested: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(request.url.path)
            return _ok_page(request.url.path)

        collector = _collector(handler)
        bundle = await collector.collect("ExampleCo", SITE)

        assert set(requested) == ALL_PATHS
        assert len(requested) == 8
        assert bundle.total_pages == 8

    async def test_fetches_concurrently(self) -> None:
        inflight = 0
        max_inflight = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal inflight, max_inflight
            inflight += 1
            max_inflight = max(max_inflight, inflight)
            await asyncio.sleep(0.01)
            inflight -= 1
            return _ok_page(request.url.path)

        collector = _collector(handler, max_concurrency=3)
        bundle = await collector.collect("ExampleCo", SITE)

        assert max_inflight > 1
        assert bundle.total_pages == 8

    async def test_cleaning_failure_records_failed_document(self) -> None:
        class BrokenCleaner:
            def clean(self, html: str, *, url: str | None = None):
                raise RuntimeError("boom")

        fetcher = HttpPageFetcher(
            settings=FetcherSettings(retry_backoff_ms=0),
            transport=httpx.MockTransport(lambda request: _ok_page(request.url.path)),
        )
        collector = EvidenceOrchestrator(
            providers=[WebsiteEvidenceProvider(fetcher=fetcher, cleaner=BrokenCleaner())]
        )

        bundle = await collector.collect("ExampleCo", SITE)

        assert len(bundle.documents) == 8
        assert all(doc.status == DocumentStatus.FAILED for doc in bundle.documents)
        assert all("boom" in (doc.error or "") for doc in bundle.documents)

    async def test_invalid_website_raises(self) -> None:
        collector = _collector(lambda request: _ok_page(request.url.path))
        with pytest.raises(InvalidWebsiteError):
            await collector.collect("ExampleCo", "not a url")

    async def test_missing_scheme_is_normalized(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _ok_page(request.url.path)

        collector = _collector(handler)
        bundle = await collector.collect("ExampleCo", "example.com")

        assert bundle.website == HttpUrl("https://example.com")
        assert bundle.total_pages == 8

    async def test_duration_is_recorded(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _ok_page(request.url.path)

        collector = _collector(handler)
        bundle = await collector.collect("ExampleCo", SITE)

        assert bundle.duration_ms >= 0
        assert bundle.attempted_pages == 8
