"""Tests for search-backed activation in the default EvidenceOrchestrator.

Covers the config-gated provider factory: registration, ordering,
merging, the disabled path, the missing-API-key path, and determinism
when search is unavailable.  No real network calls are made — the
default website provider is replaced with a ``MockTransport``-backed
build and search backends are injected.
"""

from __future__ import annotations

import httpx
from pydantic import HttpUrl

import predictron_engine.evidence.orchestrator as orchestrator_module
from predictron_engine.evidence.fetcher import FetcherSettings, HttpPageFetcher
from predictron_engine.evidence.orchestrator import (
    EvidenceOrchestrator,
    default_collection_providers,
)
from predictron_engine.evidence.provider_contracts import CollectContext
from predictron_engine.evidence.search_interfaces import SearchResult
from predictron_engine.evidence.search_provider import SearchEvidenceProvider
from predictron_engine.evidence.website_provider import WebsiteEvidenceProvider

SITE = HttpUrl("https://example.com")

_SEARCH_ENV = "EVIDENCE_SEARCH_ENABLED"
_TAVILY_KEY_ENV = "TAVILY_API_KEY"


class FakeSearchBackend:
    """Deterministic search backend for testing provider merging."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self._queries: list[str] = []

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        self._queries.append(query)
        return self._results[:max_results]

    async def close(self) -> None:
        pass


def _clear_search_env(monkeypatch) -> None:
    monkeypatch.delenv(_SEARCH_ENV, raising=False)
    monkeypatch.delenv(_TAVILY_KEY_ENV, raising=False)


def _ok_page(path: str) -> httpx.Response:
    return httpx.Response(
        200,
        text=(
            f"<html><head><title>{path} page</title></head>"
            f"<body><main><h1>{path}</h1><p>Content for {path}.</p></main></body></html>"
        ),
        headers={"content-type": "text/html; charset=utf-8"},
    )


def _network_free_website_provider() -> WebsiteEvidenceProvider:
    """A default-shaped website provider whose fetcher never hits the network."""
    fetcher = HttpPageFetcher(
        settings=FetcherSettings(retry_backoff_ms=0),
        transport=httpx.MockTransport(lambda request: _ok_page(request.url.path)),
    )
    return WebsiteEvidenceProvider(fetcher=fetcher)


class TestDefaultCollectionProviders:
    def test_disabled_default_includes_only_website(self, monkeypatch) -> None:
        _clear_search_env(monkeypatch)
        monkeypatch.setenv(_TAVILY_KEY_ENV, "should-not-matter")

        providers = default_collection_providers()

        assert [p.name for p in providers] == ["website"]
        assert isinstance(providers[0], WebsiteEvidenceProvider)

    def test_explicit_disabled_flag_skips_search(self, monkeypatch) -> None:
        monkeypatch.setenv(_SEARCH_ENV, "false")
        monkeypatch.setenv(_TAVILY_KEY_ENV, "test-key")

        assert [p.name for p in default_collection_providers()] == ["website"]

    def test_enabled_without_api_key_includes_only_website(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv(_SEARCH_ENV, "true")
        monkeypatch.delenv(_TAVILY_KEY_ENV, raising=False)

        assert [p.name for p in default_collection_providers()] == ["website"]

    def test_enabled_with_blank_api_key_includes_only_website(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv(_SEARCH_ENV, "true")
        monkeypatch.setenv(_TAVILY_KEY_ENV, "  ")

        assert [p.name for p in default_collection_providers()] == ["website"]

    def test_enabled_with_api_key_registers_search_after_website(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv(_SEARCH_ENV, "true")
        monkeypatch.setenv(_TAVILY_KEY_ENV, "test-key")

        providers = default_collection_providers()

        assert [p.name for p in providers] == ["website", "search"]
        search = providers[1]
        assert isinstance(search, SearchEvidenceProvider)
        assert search.can_collect(CollectContext(startup_name="AcmeCorp")) is True

    def test_case_insensitive_true_values(self, monkeypatch) -> None:
        for value in ("1", "true", "TRUE", "yes", "on"):
            monkeypatch.setenv(_SEARCH_ENV, value)
            monkeypatch.setenv(_TAVILY_KEY_ENV, "test-key")
            assert [p.name for p in default_collection_providers()] == [
                "website",
                "search",
            ]

    def test_orchestrator_default_registers_website_then_search(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv(_SEARCH_ENV, "true")
        monkeypatch.setenv(_TAVILY_KEY_ENV, "test-key")

        orchestrator = EvidenceOrchestrator()

        assert [p.name for p in orchestrator._providers] == ["website", "search"]


class TestSearchActivationBehavior:
    async def test_disabled_default_behavior_is_deterministic(
        self, monkeypatch
    ) -> None:
        _clear_search_env(monkeypatch)
        monkeypatch.setattr(
            orchestrator_module,
            "WebsiteEvidenceProvider",
            _network_free_website_provider,
        )

        first = await EvidenceOrchestrator().collect("ExampleCo", SITE)
        second = await EvidenceOrchestrator().collect("ExampleCo", SITE)

        assert [run.provider for run in first.providers] == ["website"]
        assert [doc.id for doc in first.documents] == [doc.id for doc in second.documents]
        assert [str(doc.original_url) for doc in first.documents] == [
            str(doc.original_url) for doc in second.documents
        ]
        assert len(first.documents) == 8
        assert first.duration_ms >= 0

    async def test_disabled_search_provider_never_collects(
        self, monkeypatch
    ) -> None:
        _clear_search_env(monkeypatch)
        monkeypatch.setattr(
            orchestrator_module,
            "WebsiteEvidenceProvider",
            _network_free_website_provider,
        )

        bundle = await EvidenceOrchestrator().collect("ExampleCo", "example.com")

        assert [run.provider for run in bundle.providers] == ["website"]
        assert bundle.total_pages == 8

    async def test_enabled_default_orchestrator_merges_website_then_search(
        self, monkeypatch
    ) -> None:
        backend = FakeSearchBackend(
            results=[
                SearchResult(url="https://example.com/", title="ExampleCo", score=0.9),
                SearchResult(
                    url="https://crunchbase.com/organization/exampleco",
                    title="ExampleCo - Crunchbase",
                    score=0.5,
                ),
            ]
        )
        monkeypatch.setenv(_SEARCH_ENV, "true")
        monkeypatch.setenv(_TAVILY_KEY_ENV, "test-key")
        monkeypatch.setattr(
            orchestrator_module,
            "WebsiteEvidenceProvider",
            _network_free_website_provider,
        )
        monkeypatch.setattr(
            orchestrator_module, "TavilySearchBackend", lambda: backend
        )

        bundle = await EvidenceOrchestrator().collect("ExampleCo", SITE)

        assert [run.provider for run in bundle.providers] == ["website", "search"]
        assert bundle.total_pages == 8
        assert bundle.providers[1].documents == 0
        assert bundle.providers[1].success is True
        source_urls = {str(src.original_url) for src in bundle.sources}
        assert "https://example.com/" in source_urls
        assert "https://crunchbase.com/organization/exampleco" in source_urls

    async def test_merges_search_sources_and_website_documents(self) -> None:
        search = SearchEvidenceProvider(
            backend=FakeSearchBackend(
                results=[
                    SearchResult(url="https://example.com/", title="Home", score=0.9),
                    SearchResult(
                        url="https://crunchbase.com/organization/exampleco",
                        title="Crunchbase",
                        score=0.5,
                    ),
                ]
            )
        )
        orchestrator = EvidenceOrchestrator(
            providers=[_network_free_website_provider(), search]
        )

        bundle = await orchestrator.collect("ExampleCo", SITE)

        assert [run.provider for run in bundle.providers] == ["website", "search"]
        assert bundle.total_pages == 8
        source_urls = {str(src.original_url) for src in bundle.sources}
        assert "https://example.com/" in source_urls
        assert "https://crunchbase.com/organization/exampleco" in source_urls

    async def test_merge_order_is_identical_across_runs(self) -> None:
        search = SearchEvidenceProvider(
            backend=FakeSearchBackend(
                results=[
                    SearchResult(url="https://example.com/", title="Home", score=0.9)
                ]
            )
        )
        orchestrator = EvidenceOrchestrator(
            providers=[_network_free_website_provider(), search]
        )

        first = await orchestrator.collect("ExampleCo", SITE)
        second = await orchestrator.collect("ExampleCo", SITE)

        assert [run.provider for run in first.providers] == [
            run.provider for run in second.providers
        ]
        assert [str(doc.original_url) for doc in first.documents] == [
            str(doc.original_url) for doc in second.documents
        ]
        assert [str(src.original_url) for src in first.sources] == [
            str(src.original_url) for src in second.sources
        ]
