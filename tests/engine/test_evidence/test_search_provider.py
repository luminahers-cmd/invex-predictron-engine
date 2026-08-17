"""Comprehensive tests for the Search Discovery Layer.

Covers the :class:`SearchEvidenceProvider`, the ``ranking`` module, and
the ``SearchBackend`` protocol contract.  Every test uses deterministic
canned data — no real HTTP calls are made.
"""

from __future__ import annotations

import asyncio
import random

import pytest

from predictron_engine.evidence.models import EvidenceSource, PageType
from predictron_engine.evidence.provider_contracts import CollectContext
from predictron_engine.evidence.ranking import (
    RankedUrl,
    rank_urls,
    score_url,
)
from predictron_engine.evidence.search_interfaces import (
    SearchBackend,
    SearchResult,
    SearchSettings,
)
from predictron_engine.evidence.search_provider import SearchEvidenceProvider

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeSearchBackend:
    """Deterministic search backend for testing."""

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = results or []
        self._queries: list[str] = []
        self._call_count = 0
        self._should_fail = False
        self._fail_count = 0

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        self._queries.append(query)
        self._call_count += 1
        if self._should_fail and self._call_count <= self._fail_count:
            raise RuntimeError("Backend unavailable")
        return self._results[:max_results]

    async def close(self) -> None:
        pass


class FailingSearchBackend:
    """Backend that always raises."""

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        raise RuntimeError("Backend unavailable")

    async def close(self) -> None:
        pass


class SlowSearchBackend:
    """Backend that always times out."""

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        await asyncio.sleep(100)
        return []

    async def close(self) -> None:
        pass


class EmptySearchBackend:
    """Backend that always returns empty results."""

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        return []

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ctx(
    name: str = "AcmeCorp",
    website: str | None = "https://acme.example.com",
) -> CollectContext:
    return CollectContext(startup_name=name, website=website)


def _results(*urls: str) -> list[SearchResult]:
    return [SearchResult(url=u, title=f"Page {i}", score=0.5) for i, u in enumerate(urls)]


# ---------------------------------------------------------------------------
# Ranking module tests (synchronous — pure functions)
# ---------------------------------------------------------------------------


class TestScoreUrl:
    def test_base_score_from_backend(self):
        score, reasons = score_url(
            "https://example.com/page",
            backend_score=0.8,
        )
        assert score == pytest.approx(0.4, abs=0.01)
        assert any("backend score" in r for r in reasons)

    def test_official_website_match(self):
        score, reasons = score_url(
            "https://acme.example.com/about",
            website_host="acme.example.com",
        )
        # base=0.0 + official(0.30) + about(0.15) = 0.45
        assert score == pytest.approx(0.45, abs=0.01)
        assert any("official website" in r for r in reasons)

    def test_official_website_subdomain(self):
        score, reasons = score_url(
            "https://docs.acme.example.com/guide",
            website_host="acme.example.com",
        )
        # base=0.0 + subdomain(0.20) = 0.20
        assert score == pytest.approx(0.20, abs=0.01)
        assert any("subdomain" in r for r in reasons)

    def test_github_boost(self):
        score, reasons = score_url(
            "https://github.com/acme/project",
            website_host="acme.example.com",
        )
        assert any("GitHub" in r for r in reasons)

    def test_crunchbase_boost(self):
        score, reasons = score_url(
            "https://crunchbase.com/organization/acme",
            website_host="acme.example.com",
        )
        assert any("Crunchbase" in r for r in reasons)

    def test_yc_boost(self):
        score, reasons = score_url(
            "https://www.ycombinator.com/companies/acme",
        )
        assert any("Y Combinator" in r for r in reasons)

    def test_linkedin_boost(self):
        score, reasons = score_url(
            "https://www.linkedin.com/company/acme",
        )
        assert any("LinkedIn" in r for r in reasons)

    def test_about_path_boost(self):
        score, reasons = score_url(
            "https://acme.example.com/about",
            website_host="acme.example.com",
        )
        assert any("about" in r for r in reasons)

    def test_pricing_path_boost(self):
        score, reasons = score_url(
            "https://acme.example.com/pricing",
            website_host="acme.example.com",
        )
        assert any("pricing" in r for r in reasons)

    def test_docs_path_boost(self):
        score, reasons = score_url(
            "https://docs.acme.example.com/api",
            website_host="acme.example.com",
        )
        assert any("documentation" in r for r in reasons) or any("API" in r for r in reasons)

    def test_techcrunch_reputable(self):
        score, reasons = score_url(
            "https://techcrunch.com/2024/01/acme-raises",
        )
        assert any("reputable" in r for r in reasons)


class TestRankUrls:
    def test_deduplication_trailing_slash(self):
        results = _results(
            "https://acme.example.com/about",
            "https://acme.example.com/about/",
        )
        ranked = rank_urls(results, website_host="acme.example.com")
        urls = [r.url for r in ranked]
        assert len(urls) == 1

    def test_deduplication_www_prefix(self):
        results = _results(
            "https://www.acme.example.com/about",
            "https://acme.example.com/about",
        )
        ranked = rank_urls(results, website_host="acme.example.com")
        urls = [r.url for r in ranked]
        assert len(urls) == 1

    def test_deduplication_scheme_difference(self):
        results = _results(
            "https://acme.example.com/about",
            "http://acme.example.com/about",
        )
        ranked = rank_urls(results, website_host="acme.example.com")
        urls = [r.url for r in ranked]
        assert len(urls) == 1

    def test_deduplication_keeps_highest_score(self):
        r1 = SearchResult(url="https://acme.example.com/about", score=0.3)
        r2 = SearchResult(url="https://acme.example.com/about/", score=0.9)
        ranked = rank_urls([r1, r2], website_host="acme.example.com")
        assert len(ranked) == 1
        assert ranked[0].score >= 0.5

    def test_deterministic_ordering(self):
        results = _results(
            "https://acme.example.com/pricing",
            "https://acme.example.com/about",
            "https://github.com/acme/repo",
            "https://acme.example.com/",
        )
        first = rank_urls(results, website_host="acme.example.com")
        second = rank_urls(results, website_host="acme.example.com")
        assert [r.url for r in first] == [r.url for r in second]

    def test_empty_input(self):
        ranked = rank_urls([])
        assert ranked == []

    def test_filters_low_value_login(self):
        results = _results(
            "https://acme.example.com/login",
            "https://acme.example.com/about",
        )
        ranked = rank_urls(results, website_host="acme.example.com")
        urls = [r.url for r in ranked]
        assert not any("login" in u for u in urls)
        assert any("about" in u for u in urls)

    def test_filters_low_value_signup(self):
        results = _results(
            "https://acme.example.com/signup",
            "https://acme.example.com/about",
        )
        ranked = rank_urls(results, website_host="acme.example.com")
        urls = [r.url for r in ranked]
        assert not any("signup" in u for u in urls)

    def test_filters_low_value_cart(self):
        results = _results(
            "https://acme.example.com/cart",
            "https://acme.example.com/about",
        )
        ranked = rank_urls(results, website_host="acme.example.com")
        urls = [r.url for r in ranked]
        assert not any("cart" in u for u in urls)

    def test_filters_low_value_privacy(self):
        results = _results(
            "https://acme.example.com/privacy",
            "https://acme.example.com/about",
        )
        ranked = rank_urls(results, website_host="acme.example.com")
        urls = [r.url for r in ranked]
        assert not any("privacy" in u for u in urls)

    def test_filters_low_value_rss(self):
        results = _results(
            "https://acme.example.com/feed",
            "https://acme.example.com/about",
        )
        ranked = rank_urls(results, website_host="acme.example.com")
        urls = [r.url for r in ranked]
        assert not any("feed" in u for u in urls)

    def test_min_score_filter(self):
        r1 = SearchResult(url="https://unknown-domain.xyz/page", score=0.0)
        ranked = rank_urls([r1], min_score=0.3)
        assert len(ranked) == 0

    def test_all_results_filtered(self):
        results = _results(
            "https://acme.example.com/login",
            "https://acme.example.com/signup",
            "https://acme.example.com/cart",
        )
        ranked = rank_urls(results, website_host="acme.example.com")
        assert ranked == []

    def test_malformed_url_skipped(self):
        results = [
            SearchResult(url="not-a-valid-url", score=0.5),
            SearchResult(url="https://acme.example.com/about", score=0.5),
        ]
        ranked = rank_urls(results, website_host="acme.example.com")
        assert len(ranked) == 1
        assert "acme.example.com" in ranked[0].url

    def test_ranking_order_prefers_official(self):
        results = [
            SearchResult(url="https://techcrunch.com/acme-story", score=0.8),
            SearchResult(url="https://acme.example.com/", score=0.5),
        ]
        ranked = rank_urls(results, website_host="acme.example.com")
        assert ranked[0].url == "https://acme.example.com/"

    def test_ranking_order_prefers_github(self):
        results = [
            SearchResult(url="https://random-blog.com/acme", score=0.9),
            SearchResult(url="https://github.com/acme/repo", score=0.5),
        ]
        ranked = rank_urls(results, website_host="acme.example.com")
        assert ranked[0].url == "https://github.com/acme/repo"

    def test_ranked_url_comparison(self):
        a = RankedUrl(url="a.com", score=0.8, reasons=())
        b = RankedUrl(url="b.com", score=0.6, reasons=())
        assert a < b  # higher score sorts first via __lt__

    def test_ranked_url_tiebreak_lexicographic(self):
        a = RankedUrl(url="a.com", score=0.8, reasons=())
        b = RankedUrl(url="b.com", score=0.8, reasons=())
        assert a < b  # same score, 'a' sorts first

    def test_ranking_stable_under_reordering(self):
        results = _results(
            "https://github.com/acme/repo",
            "https://acme.example.com/about",
            "https://techcrunch.com/acme-story",
            "https://acme.example.com/pricing",
        )
        first = rank_urls(results, website_host="acme.example.com")
        shuffled = results[:]
        random.seed(42)
        random.shuffle(shuffled)
        second = rank_urls(shuffled, website_host="acme.example.com")
        assert [r.url for r in first] == [r.url for r in second]


# ---------------------------------------------------------------------------
# SearchEvidenceProvider tests (async)
# ---------------------------------------------------------------------------


class TestSearchEvidenceProvider:
    def test_can_collect_with_backend(self):
        provider = SearchEvidenceProvider(backend=FakeSearchBackend())
        assert provider.can_collect(_ctx()) is True

    def test_can_collect_without_backend(self):
        provider = SearchEvidenceProvider()
        assert provider.can_collect(_ctx()) is False

    def test_provider_name(self):
        provider = SearchEvidenceProvider(backend=FakeSearchBackend())
        assert provider.name == "search"

    def test_provider_satisfies_protocol(self):
        provider = SearchEvidenceProvider(backend=FakeSearchBackend())
        assert hasattr(provider, "name")
        assert hasattr(provider, "can_collect")
        assert hasattr(provider, "collect")

    async def test_collect_returns_provider_result(self):
        backend = FakeSearchBackend(
            results=[
                SearchResult(url="https://acme.example.com/", score=0.9),
                SearchResult(url="https://github.com/acme/repo", score=0.7),
            ]
        )
        provider = SearchEvidenceProvider(backend=backend)
        result = await provider.collect(_ctx())
        assert result.provider == "search"
        assert result.success is True
        assert isinstance(result.sources, list)

    async def test_collect_discoveres_urls(self):
        backend = FakeSearchBackend(
            results=[
                SearchResult(url="https://acme.example.com/", score=0.9),
                SearchResult(url="https://github.com/acme/repo", score=0.7),
            ]
        )
        provider = SearchEvidenceProvider(backend=backend)
        result = await provider.collect(_ctx())
        assert len(result.sources) >= 1

    async def test_collect_documents_always_empty(self):
        backend = FakeSearchBackend(
            results=[SearchResult(url="https://acme.example.com/", score=0.9)]
        )
        provider = SearchEvidenceProvider(backend=backend)
        result = await provider.collect(_ctx())
        assert result.documents == []

    async def test_collect_passes_query_to_backend(self):
        backend = FakeSearchBackend()
        provider = SearchEvidenceProvider(backend=backend)
        await provider.collect(_ctx(name="TestCo"))
        assert backend._queries == ["TestCo"]

    async def test_collect_deduplicates_results(self):
        backend = FakeSearchBackend(
            results=[
                SearchResult(url="https://acme.example.com/about", score=0.9),
                SearchResult(url="https://acme.example.com/about/", score=0.8),
                SearchResult(url="https://www.acme.example.com/about", score=0.7),
            ]
        )
        provider = SearchEvidenceProvider(backend=backend)
        result = await provider.collect(_ctx())
        assert len(result.sources) == 1

    async def test_collect_filters_low_value_pages(self):
        backend = FakeSearchBackend(
            results=[
                SearchResult(url="https://acme.example.com/login", score=0.9),
                SearchResult(url="https://acme.example.com/signup", score=0.9),
                SearchResult(url="https://acme.example.com/about", score=0.5),
            ]
        )
        provider = SearchEvidenceProvider(backend=backend)
        result = await provider.collect(_ctx())
        urls = [str(s.original_url) for s in result.sources]
        assert not any("login" in u for u in urls)
        assert not any("signup" in u for u in urls)

    async def test_collect_no_backend_returns_empty(self):
        provider = SearchEvidenceProvider()
        result = await provider.collect(_ctx())
        assert result.success is True
        assert result.sources == []
        assert result.documents == []
        assert result.failure_reason is not None

    async def test_collect_backend_failure_recorded(self):
        provider = SearchEvidenceProvider(backend=FailingSearchBackend())
        result = await provider.collect(_ctx())
        assert result.success is False
        assert result.failure_reason is not None

    async def test_collect_timeout_handled(self):
        backend = SlowSearchBackend()
        settings = SearchSettings(timeout=0.1)
        provider = SearchEvidenceProvider(backend=backend, settings=settings)
        result = await provider.collect(_ctx())
        assert result.success is True
        assert result.sources == []

    async def test_collect_empty_results(self):
        provider = SearchEvidenceProvider(backend=EmptySearchBackend())
        result = await provider.collect(_ctx())
        assert result.success is True
        assert result.sources == []

    async def test_collect_duration_populated(self):
        backend = FakeSearchBackend(
            results=[SearchResult(url="https://acme.example.com/", score=0.9)]
        )
        provider = SearchEvidenceProvider(backend=backend)
        result = await provider.collect(_ctx())
        assert result.duration_ms >= 0

    async def test_collect_attempted_pages_is_one(self):
        backend = FakeSearchBackend(
            results=[SearchResult(url="https://acme.example.com/", score=0.9)]
        )
        provider = SearchEvidenceProvider(backend=backend)
        result = await provider.collect(_ctx())
        assert result.attempted_pages == 1

    async def test_collect_website_parsed(self):
        backend = FakeSearchBackend()
        provider = SearchEvidenceProvider(backend=backend)
        result = await provider.collect(_ctx(website="https://acme.example.com"))
        assert str(result.website) == "https://acme.example.com/"

    async def test_collect_no_website(self):
        backend = FakeSearchBackend()
        provider = SearchEvidenceProvider(backend=backend)
        result = await provider.collect(_ctx(website=None))
        assert result.website is None

    async def test_collect_malformed_website(self):
        backend = FakeSearchBackend()
        provider = SearchEvidenceProvider(backend=backend)
        result = await provider.collect(_ctx(website="not-a-url"))
        assert result.website is not None  # gets https:// prepended

    async def test_retry_on_failure(self):
        backend = FakeSearchBackend(
            results=[SearchResult(url="https://acme.example.com/", score=0.9)]
        )
        backend._should_fail = True
        backend._fail_count = 1
        settings = SearchSettings(retry_count=2, retry_delay=0.01)
        provider = SearchEvidenceProvider(backend=backend, settings=settings)
        result = await provider.collect(_ctx())
        assert result.success is True
        assert len(result.sources) >= 1

    async def test_retry_exhausted(self):
        backend = FailingSearchBackend()
        settings = SearchSettings(retry_count=2, retry_delay=0.01)
        provider = SearchEvidenceProvider(backend=backend, settings=settings)
        result = await provider.collect(_ctx())
        assert result.success is False

    async def test_no_retry_when_count_zero(self):
        backend = FailingSearchBackend()
        settings = SearchSettings(retry_count=0)
        provider = SearchEvidenceProvider(backend=backend, settings=settings)
        result = await provider.collect(_ctx())
        assert result.success is False

    def test_default_settings(self):
        provider = SearchEvidenceProvider(backend=FakeSearchBackend())
        assert provider._settings.max_results == 20
        assert provider._settings.timeout == 10.0
        assert provider._settings.retry_count == 1

    def test_custom_settings(self):
        settings = SearchSettings(max_results=5, timeout=3.0, retry_count=3)
        provider = SearchEvidenceProvider(
            backend=FakeSearchBackend(), settings=settings
        )
        assert provider._settings.max_results == 5
        assert provider._settings.timeout == 3.0
        assert provider._settings.retry_count == 3

    async def test_sources_are_evidence_source_objects(self):
        backend = FakeSearchBackend(
            results=[SearchResult(url="https://acme.example.com/", score=0.9)]
        )
        provider = SearchEvidenceProvider(backend=backend)
        result = await provider.collect(_ctx())
        for source in result.sources:
            assert isinstance(source, EvidenceSource)

    async def test_sources_have_unknown_page_type(self):
        backend = FakeSearchBackend(
            results=[SearchResult(url="https://acme.example.com/", score=0.9)]
        )
        provider = SearchEvidenceProvider(backend=backend)
        result = await provider.collect(_ctx())
        for source in result.sources:
            assert source.page_type == PageType.UNKNOWN

    async def test_sources_have_urls(self):
        backend = FakeSearchBackend(
            results=[SearchResult(url="https://acme.example.com/", score=0.9)]
        )
        provider = SearchEvidenceProvider(backend=backend)
        result = await provider.collect(_ctx())
        for source in result.sources:
            assert source.url is not None
            assert source.original_url is not None

    async def test_sources_marked_successful(self):
        backend = FakeSearchBackend(
            results=[
                SearchResult(url="https://acme.example.com/", score=0.9),
                SearchResult(url="https://github.com/acme/repo", score=0.7),
            ]
        )
        provider = SearchEvidenceProvider(backend=backend)
        result = await provider.collect(_ctx())
        for source in result.sources:
            assert source.success is True

    async def test_async_context_manager(self):
        backend = FakeSearchBackend()
        async with SearchEvidenceProvider(backend=backend) as provider:
            result = await provider.collect(_ctx())
            assert result.success is True

    async def test_close_method(self):
        provider = SearchEvidenceProvider(backend=FakeSearchBackend())
        await provider.close()

    async def test_collect_no_website_provided(self):
        backend = FakeSearchBackend(
            results=[SearchResult(url="https://github.com/acme/repo", score=0.9)]
        )
        provider = SearchEvidenceProvider(backend=backend)
        result = await provider.collect(_ctx(website=None))
        assert result.success is True
        assert result.website is None

    async def test_deterministic_output(self):
        backend = FakeSearchBackend(
            results=[
                SearchResult(url="https://acme.example.com/pricing", score=0.9),
                SearchResult(url="https://github.com/acme/repo", score=0.7),
                SearchResult(url="https://techcrunch.com/acme-story", score=0.6),
            ]
        )
        provider = SearchEvidenceProvider(backend=backend)
        first = await provider.collect(_ctx())
        second = await provider.collect(_ctx())
        assert [str(s.original_url) for s in first.sources] == [
            str(s.original_url) for s in second.sources
        ]

    async def test_malformed_url_in_results_skipped(self):
        backend = FakeSearchBackend(
            results=[
                SearchResult(url="not-a-valid-url", score=0.9),
                SearchResult(url="https://acme.example.com/about", score=0.5),
            ]
        )
        provider = SearchEvidenceProvider(backend=backend)
        result = await provider.collect(_ctx())
        assert len(result.sources) == 1
        assert "acme.example.com" in str(result.sources[0].original_url)


# ---------------------------------------------------------------------------
# SearchBackend protocol tests
# ---------------------------------------------------------------------------


class TestSearchBackendProtocol:
    def test_fake_backend_is_protocol_compatible(self):
        backend = FakeSearchBackend()
        assert isinstance(backend, SearchBackend)

    def test_failing_backend_is_protocol_compatible(self):
        backend = FailingSearchBackend()
        assert isinstance(backend, SearchBackend)

    def test_empty_backend_is_protocol_compatible(self):
        backend = EmptySearchBackend()
        assert isinstance(backend, SearchBackend)

    def test_search_result_model(self):
        r = SearchResult(url="https://example.com", title="Example", score=0.5)
        assert r.url == "https://example.com"
        assert r.title == "Example"
        assert r.score == 0.5

    def test_search_result_defaults(self):
        r = SearchResult(url="https://example.com")
        assert r.title == ""
        assert r.snippet == ""
        assert r.score == 0.0

    def test_search_settings_defaults(self):
        s = SearchSettings()
        assert s.max_results == 20
        assert s.timeout == 10.0
        assert s.backend == "noop"
        assert s.retry_count == 1
        assert s.retry_delay == 1.0


# ---------------------------------------------------------------------------
# SearchSettings validation tests
# ---------------------------------------------------------------------------


class TestSearchSettings:
    def test_max_results_must_be_positive(self):
        with pytest.raises(Exception):
            SearchSettings(max_results=0)

    def test_timeout_must_be_positive(self):
        with pytest.raises(Exception):
            SearchSettings(timeout=0)

    def test_retry_count_non_negative(self):
        s = SearchSettings(retry_count=0)
        assert s.retry_count == 0

    def test_retry_delay_non_negative(self):
        s = SearchSettings(retry_delay=0.0)
        assert s.retry_delay == 0.0
