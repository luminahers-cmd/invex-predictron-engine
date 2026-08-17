"""Tests for the TavilySearchBackend production search backend.

Every test uses ``httpx.MockTransport`` — no real HTTP calls are made.
Covers: successful search, configuration, error handling, response parsing,
protocol compliance, and integration with :class:`SearchEvidenceProvider`.
"""

from __future__ import annotations

import json

import httpx
import pytest

from predictron_engine.evidence.provider_contracts import CollectContext
from predictron_engine.evidence.search_backends import TavilySearchBackend
from predictron_engine.evidence.search_interfaces import SearchBackend, SearchResult
from predictron_engine.evidence.search_provider import SearchEvidenceProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_API_KEY = "tvly-test-key-12345"


def _tavily_response(
    results: list[dict] | None = None,
    *,
    status_code: int = 200,
) -> tuple[int, dict]:
    """Build a mock (status_code, body) for the Tavily API."""
    if results is not None:
        return status_code, {"results": results}
    return status_code, {"results": []}


def _tavily_result(
    url: str = "https://example.com",
    title: str = "Example",
    content: str = "Snippet text",
    score: float = 0.9,
) -> dict:
    return {"url": url, "title": title, "content": content, "score": score}


def _make_transport(
    status_code: int = 200,
    body: dict | None = None,
    *,
    raise_on_connect: type[Exception] | None = None,
    raise_on_request: type[Exception] | None = None,
) -> httpx.MockTransport:
    """Create a ``MockTransport`` with a canned response.

    Parameters
    ----------
    status_code:
        HTTP status to return.
    body:
        JSON body. Defaults to ``{"results": []}``.
    raise_on_connect:
        If set, raise this exception during connection.
    raise_on_request:
        If set, raise this exception during the request.
    """

    if body is None:
        body = {"results": []}

    app_json = {"content-type": "application/json"}

    def _handler(request: httpx.Request) -> httpx.Response:
        if raise_on_request:
            raise raise_on_request("simulated failure")
        content = json.dumps(body).encode()
        return httpx.Response(status_code, content=content, headers=app_json)

    def _handler_connect(request: httpx.Request) -> httpx.Response:
        if raise_on_connect:
            raise raise_on_connect("simulated connection failure")
        content = json.dumps(body).encode()
        return httpx.Response(status_code, content=content, headers=app_json)

    if raise_on_connect:
        return httpx.MockTransport(_handler_connect)
    return httpx.MockTransport(_handler)


def _make_timeout_transport() -> httpx.MockTransport:
    """Transport that always raises a timeout error."""
    return httpx.MockTransport(_raise_timeout)


def _raise_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("Simulated timeout")


# ---------------------------------------------------------------------------
# TavilySearchBackend — construction and configuration
# ---------------------------------------------------------------------------


class TestTavilySearchBackendConfig:
    def test_default_configuration(self):
        backend = TavilySearchBackend(api_key=_API_KEY)
        assert backend._api_key == _API_KEY
        assert backend._api_url == "https://api.tavily.com/search"
        assert backend._timeout == 10.0
        assert backend._default_max_results == 20

    def test_custom_configuration(self):
        backend = TavilySearchBackend(
            api_key="custom-key",
            api_url="https://custom.api/search",
            timeout=5.0,
            max_results=10,
        )
        assert backend._api_key == "custom-key"
        assert backend._api_url == "https://custom.api/search"
        assert backend._timeout == 5.0
        assert backend._default_max_results == 10

    def test_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TAVILY_API_KEY", "env-key")
        monkeypatch.setenv("TAVILY_API_URL", "https://env.api/search")
        monkeypatch.setenv("TAVILY_TIMEOUT", "3.0")
        monkeypatch.setenv("TAVILY_MAX_RESULTS", "5")
        backend = TavilySearchBackend()
        assert backend._api_key == "env-key"
        assert backend._api_url == "https://env.api/search"
        assert backend._timeout == 3.0
        assert backend._default_max_results == 5

    def test_explicit_overrides_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("TAVILY_API_KEY", "env-key")
        backend = TavilySearchBackend(api_key="explicit-key")
        assert backend._api_key == "explicit-key"

    async def test_empty_api_key_returns_empty(self):
        backend = TavilySearchBackend(api_key="")
        transport = _make_transport(200, {"results": [_tavily_result()]})
        backend._transport = transport
        result = await backend.search("test", 10)
        assert result == []

    def test_missing_api_key_returns_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        backend = TavilySearchBackend()
        assert backend._api_key == ""


# ---------------------------------------------------------------------------
# TavilySearchBackend — successful search
# ---------------------------------------------------------------------------


class TestTavilySearchSuccess:
    async def test_successful_search_returns_results(self):
        results_data = [
            _tavily_result(url="https://acme.com", title="Acme", score=0.95),
            _tavily_result(url="https://github.com/acme/repo", title="Repo", score=0.8),
            _tavily_result(
                url="https://techcrunch.com/acme", title="TC", score=0.7
            ),
        ]
        transport = _make_transport(200, {"results": results_data})
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        async with backend:
            results = await backend.search("Acme Corp", 10)
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].url == "https://acme.com"
        assert results[0].title == "Acme"
        assert results[0].score == pytest.approx(0.95)
        assert results[1].url == "https://github.com/acme/repo"
        assert results[2].url == "https://techcrunch.com/acme"

    async def test_empty_results(self):
        transport = _make_transport(200, {"results": []})
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        async with backend:
            results = await backend.search("nonexistent company xyz", 10)
        assert results == []

    async def test_missing_results_key(self):
        transport = _make_transport(200, {"answer": "some answer"})
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        async with backend:
            results = await backend.search("test", 10)
        assert results == []

    async def test_result_with_missing_fields(self):
        transport = _make_transport(200, {"results": [{"url": "https://a.com"}]})
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        async with backend:
            results = await backend.search("test", 10)
        assert len(results) == 1
        assert results[0].url == "https://a.com"
        assert results[0].title == ""
        assert results[0].snippet == ""
        assert results[0].score == 0.0

    async def test_score_clamped_to_one(self):
        transport = _make_transport(
            200, {"results": [{"url": "https://a.com", "score": 5.0}]}
        )
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        async with backend:
            results = await backend.search("test", 10)
        assert results[0].score == 1.0

    async def test_score_clamped_to_zero(self):
        transport = _make_transport(
            200, {"results": [{"url": "https://a.com", "score": -1.0}]}
        )
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        async with backend:
            results = await backend.search("test", 10)
        assert results[0].score == 0.0

    async def test_invalid_score_type_defaults_to_zero(self):
        transport = _make_transport(
            200, {"results": [{"url": "https://a.com", "score": "bad"}]}
        )
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        async with backend:
            results = await backend.search("test", 10)
        assert results[0].score == 0.0

    async def test_non_dict_results_skipped(self):
        transport = _make_transport(200, {"results": ["string", 42, None]})
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        async with backend:
            results = await backend.search("test", 10)
        assert results == []

    async def test_result_without_url_skipped(self):
        transport = _make_transport(200, {"results": [{"title": "No URL"}]})
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        async with backend:
            results = await backend.search("test", 10)
        assert results == []

    async def test_max_results_capped(self):
        items = [_tavily_result(url=f"https://a{i}.com") for i in range(50)]
        transport = _make_transport(200, {"results": items})
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        async with backend:
            results = await backend.search("test", 5)
        assert len(results) == 5

    async def test_request_payload(self):
        captured_requests: list[httpx.Request] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            return httpx.Response(
                200,
                json={"results": []},
                headers={"content-type": "application/json"},
            )

        transport = httpx.MockTransport(_handler)
        backend = TavilySearchBackend(
            api_key=_API_KEY, max_results=10, transport=transport
        )
        async with backend:
            await backend.search("Acme Corp", 10)
        assert len(captured_requests) == 1
        req = captured_requests[0]
        body = json.loads(req.content)
        assert body["query"] == "Acme Corp"
        assert body["max_results"] == 10
        assert body["search_depth"] == "basic"
        assert req.headers.get("authorization") == f"Bearer {_API_KEY}"


# ---------------------------------------------------------------------------
# TavilySearchBackend — error handling
# ---------------------------------------------------------------------------


class TestTavilySearchErrors:
    async def test_invalid_api_key_returns_empty(self):
        transport = _make_transport(401, {"error": "Unauthorized"})
        backend = TavilySearchBackend(api_key="bad-key", transport=transport)
        async with backend:
            results = await backend.search("test", 10)
        assert results == []

    async def test_rate_limit_returns_empty(self):
        transport = _make_transport(429, {"error": "Rate limited"})
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        async with backend:
            results = await backend.search("test", 10)
        assert results == []

    async def test_server_error_returns_empty(self):
        transport = _make_transport(500, {"error": "Internal error"})
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        async with backend:
            results = await backend.search("test", 10)
        assert results == []

    async def test_timeout_returns_empty(self):
        backend = TavilySearchBackend(
            api_key=_API_KEY,
            timeout=0.01,
            transport=_make_timeout_transport(),
        )
        async with backend:
            results = await backend.search("test", 10)
        assert results == []

    async def test_network_error_returns_empty(self):
        def _handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(_handler)
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        async with backend:
            results = await backend.search("test", 10)
        assert results == []

    async def test_malformed_json_returns_empty(self):
        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"not json at all",
                headers={"content-type": "text/plain"},
            )

        transport = httpx.MockTransport(_handler)
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        async with backend:
            results = await backend.search("test", 10)
        assert results == []


# ---------------------------------------------------------------------------
# TavilySearchBackend — protocol compliance
# ---------------------------------------------------------------------------


class TestTavilySearchProtocol:
    def test_satisfies_search_backend_protocol(self):
        backend = TavilySearchBackend(api_key=_API_KEY)
        assert isinstance(backend, SearchBackend)

    async def test_close_method(self):
        backend = TavilySearchBackend(api_key=_API_KEY, transport=_make_transport())
        await backend._ensure_client()
        assert backend._client is not None
        await backend.close()
        assert backend._client is None

    async def test_async_context_manager(self):
        transport = _make_transport()
        async with TavilySearchBackend(
            api_key=_API_KEY, transport=transport
        ) as backend:
            results = await backend.search("test", 10)
            assert isinstance(results, list)

    async def test_client_lazily_created(self):
        backend = TavilySearchBackend(api_key=_API_KEY, transport=_make_transport())
        assert backend._client is None
        await backend._ensure_client()
        assert backend._client is not None
        await backend.close()


# ---------------------------------------------------------------------------
# TavilySearchBackend — deterministic mapping
# ---------------------------------------------------------------------------


class TestTavilySearchDeterminism:
    async def test_deterministic_output(self):
        results_data = [
            _tavily_result(url="https://a.com", score=0.9),
            _tavily_result(url="https://b.com", score=0.8),
        ]
        transport = _make_transport(200, {"results": results_data})
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        async with backend:
            first = await backend.search("test", 10)
        transport2 = _make_transport(200, {"results": results_data})
        backend2 = TavilySearchBackend(api_key=_API_KEY, transport=transport2)
        async with backend2:
            second = await backend2.search("test", 10)
        assert [r.url for r in first] == [r.url for r in second]
        assert [r.score for r in first] == [r.score for r in second]

    async def test_content_mapped_to_snippet(self):
        transport = _make_transport(
            200, {"results": [{"url": "https://a.com", "content": "Hello world"}]}
        )
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        async with backend:
            results = await backend.search("test", 10)
        assert results[0].snippet == "Hello world"


# ---------------------------------------------------------------------------
# Integration with SearchEvidenceProvider
# ---------------------------------------------------------------------------


class TestTavilyWithProvider:
    async def test_provider_uses_tavily_backend(self):
        results_data = [
            _tavily_result(url="https://acme.com/", score=0.95),
            _tavily_result(url="https://github.com/acme/repo", score=0.8),
        ]
        transport = _make_transport(200, {"results": results_data})
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        provider = SearchEvidenceProvider(backend=backend)
        ctx = CollectContext(startup_name="AcmeCorp", website="https://acme.com")
        result = await provider.collect(ctx)
        assert result.success is True
        assert len(result.sources) >= 1
        assert result.documents == []

    async def test_provider_handles_tavily_failure(self):
        transport = _make_transport(500)
        backend = TavilySearchBackend(api_key=_API_KEY, transport=transport)
        provider = SearchEvidenceProvider(backend=backend)
        ctx = CollectContext(startup_name="AcmeCorp", website="https://acme.com")
        result = await provider.collect(ctx)
        assert result.success is True
        assert result.sources == []
