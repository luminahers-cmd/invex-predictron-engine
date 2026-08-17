"""Tests for the asynchronous HTTP page fetcher."""

from __future__ import annotations

import httpx
from pydantic import HttpUrl

from predictron_engine.evidence.fetcher import FetcherSettings, HttpPageFetcher


def _fetcher(handler, **settings) -> HttpPageFetcher:
    return HttpPageFetcher(
        settings=FetcherSettings(retry_backoff_ms=0, **settings),
        transport=httpx.MockTransport(handler),
    )


class TestHttpPageFetcher:
    async def test_fetches_successful_page(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, text="<p>Hello</p>", headers={"content-type": "text/html"})

        fetcher = _fetcher(handler)
        result = await fetcher.fetch(HttpUrl("https://example.com"))
        await fetcher.close()

        assert result.ok
        assert result.status == 200
        assert result.html == "<p>Hello</p>"
        assert result.content_type == "text/html"
        assert result.final_url == HttpUrl("https://example.com")
        assert result.response_time_ms >= 0
        assert result.retry_count == 0
        assert calls == ["https://example.com/"]

    async def test_404_returns_error_without_retry(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(404, text="not found")

        fetcher = _fetcher(handler, max_retries=3)
        result = await fetcher.fetch(HttpUrl("https://example.com/missing"))
        await fetcher.close()

        assert not result.ok
        assert result.status == 404
        assert result.error == "HTTP 404"
        assert len(calls) == 1

    async def test_403_returns_error_without_retry(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(403)

        fetcher = _fetcher(handler, max_retries=3)
        result = await fetcher.fetch(HttpUrl("https://example.com/private"))
        await fetcher.close()

        assert not result.ok
        assert result.status == 403
        assert len(calls) == 1

    async def test_500_then_success_retries(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            if len(calls) == 1:
                return httpx.Response(500)
            return httpx.Response(200, text="<p>recovered</p>")

        fetcher = _fetcher(handler, max_retries=2)
        result = await fetcher.fetch(HttpUrl("https://example.com"))
        await fetcher.close()

        assert result.ok
        assert result.html == "<p>recovered</p>"
        assert result.retry_count == 1
        assert len(calls) == 2

    async def test_persistent_503_fails_after_retries(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(503)

        fetcher = _fetcher(handler, max_retries=2)
        result = await fetcher.fetch(HttpUrl("https://example.com"))
        await fetcher.close()

        assert not result.ok
        assert result.status == 503
        assert result.error == "HTTP 503"
        assert result.retry_count == 2
        assert len(calls) == 3

    async def test_transport_error_then_success_retries(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            if len(calls) == 1:
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200, text="<p>ok</p>")

        fetcher = _fetcher(handler, max_retries=1)
        result = await fetcher.fetch(HttpUrl("https://example.com"))
        await fetcher.close()

        assert result.ok
        assert len(calls) == 2

    async def test_transport_error_exhausted(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            raise httpx.ConnectError("connection refused", request=request)

        fetcher = _fetcher(handler, max_retries=2)
        result = await fetcher.fetch(HttpUrl("https://example.com"))
        await fetcher.close()

        assert not result.ok
        assert result.status == 0
        assert "ConnectError" in (result.error or "")
        assert len(calls) == 3

    async def test_redirects_followed_and_final_url_captured(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            if request.url.path == "/":
                return httpx.Response(302, headers={"location": "/about"})
            return httpx.Response(200, text="<p>About</p>")

        fetcher = _fetcher(handler)
        result = await fetcher.fetch(HttpUrl("https://example.com/"))
        await fetcher.close()

        assert result.ok
        assert str(result.final_url).endswith("/about")
        assert calls == ["https://example.com/", "https://example.com/about"]

    async def test_oversized_response_truncated(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text="x" * 1_000,
                headers={"content-type": "text/plain", "content-length": "50"},
            )

        fetcher = _fetcher(handler, max_bytes=100)
        result = await fetcher.fetch(HttpUrl("https://example.com"))
        await fetcher.close()

        assert result.ok
        assert result.truncated
        assert len(result.html or "") <= 100

    async def test_oversized_content_length_rejected(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-length": "999999"})

        fetcher = _fetcher(handler, max_bytes=100)
        result = await fetcher.fetch(HttpUrl("https://example.com"))
        await fetcher.close()

        assert not result.ok
        assert "size cap" in (result.error or "")

    async def test_custom_user_agent_sent(self) -> None:
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["ua"] = request.headers.get("user-agent", "")
            return httpx.Response(200, text="<p>ok</p>")

        fetcher = _fetcher(handler, user_agent="TestAgent/1.0")
        result = await fetcher.fetch(HttpUrl("https://example.com"))
        await fetcher.close()

        assert result.ok
        assert captured["ua"] == "TestAgent/1.0"
