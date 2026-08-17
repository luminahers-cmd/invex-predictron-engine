"""Production search backend implementations.

Each backend wraps a specific internet search API and maps its responses
into the generic :class:`~predictron_engine.evidence.search_interfaces.SearchResult`
format.  Backends are stateless — all mutable state (HTTP client, API key)
lives inside the instance and is managed via ``close`` / async-context-manager
protocols.

Currently implemented
---------------------
* :class:`TavilySearchBackend` — Tavily Search API (https://tavily.com)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from predictron_engine.evidence.search_interfaces import SearchResult

logger = logging.getLogger(__name__)


class TavilySearchBackend:
    """Production :class:`SearchBackend` backed by the Tavily Search API.

    Configuration is read from environment variables:

    * ``TAVILY_API_KEY`` — **required**, your Tavily API key.
    * ``TAVILY_API_URL`` — endpoint override (default
      ``https://api.tavily.com/search``).
    * ``TAVILY_TIMEOUT`` — request timeout in seconds (default ``10.0``).
    * ``TAVILY_MAX_RESULTS`` — default max results per query (default ``20``).

    Parameters
    ----------
    api_key:
        Tavily API key.  Falls back to the ``TAVILY_API_KEY`` env var.
    api_url:
        Endpoint URL.  Falls back to ``TAVILY_API_URL`` env var, then
        ``"https://api.tavily.com/search"``.
    timeout:
        HTTP timeout in seconds.  Falls back to ``TAVILY_TIMEOUT`` env var,
        then ``10.0``.
    max_results:
        Default maximum results.  Falls back to ``TAVILY_MAX_RESULTS`` env
        var, then ``20``.
    transport:
        Optional ``httpx.AsyncBaseTransport`` for dependency injection (tests).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        timeout: float | None = None,
        max_results: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self._api_url = (
            api_url
            or os.environ.get("TAVILY_API_URL", "")
            or "https://api.tavily.com/search"
        )
        self._timeout = timeout or float(os.environ.get("TAVILY_TIMEOUT", "10.0"))
        self._default_max_results = max_results or int(
            os.environ.get("TAVILY_MAX_RESULTS", "20")
        )
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # SearchBackend protocol
    # ------------------------------------------------------------------

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Execute *query* via the Tavily Search API.

        Returns up to *max_results* :class:`SearchResult` objects mapped
        from the Tavily response.  Transient errors (network, timeout,
        rate limit, invalid key) are caught and logged; an empty list is
        returned so the provider layer never receives unexpected exceptions
        for recoverable failures.
        """
        if not self._api_key:
            logger.warning("TavilySearchBackend: no API key configured")
            return []

        client = await self._ensure_client()
        effective_max = min(max_results, self._default_max_results)

        payload: dict[str, Any] = {
            "query": query,
            "max_results": effective_max,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
        }

        try:
            response = await client.post(
                self._api_url,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.TimeoutException:
            logger.warning("TavilySearchBackend: request timed out for query %r", query)
            return []
        except httpx.TransportError as exc:
            logger.warning(
                "TavilySearchBackend: transport error for query %r: %s", query, exc
            )
            return []

        if response.status_code == 401:
            logger.warning("TavilySearchBackend: invalid API key")
            return []
        if response.status_code == 429:
            logger.warning("TavilySearchBackend: rate limited")
            return []
        if response.status_code != 200:
            logger.warning(
                "TavilySearchBackend: unexpected status %d for query %r",
                response.status_code,
                query,
            )
            return []

        try:
            data = response.json()
        except ValueError:
            logger.warning("TavilySearchBackend: malformed JSON response")
            return []

        return self._parse_results(data)[:max_results]

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> TavilySearchBackend:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Lazily create the shared HTTP client."""
        if self._client is None:
            timeout = httpx.Timeout(self._timeout, connect=min(5.0, self._timeout))
            self._client = httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                transport=self._transport,
                headers={"User-Agent": "InveX-SearchBackend/1.0"},
            )
        return self._client

    @staticmethod
    def _parse_results(data: dict[str, Any]) -> list[SearchResult]:
        """Map the Tavily response body to a list of :class:`SearchResult`.

        Tavily response shape::

            {
                "results": [
                    {
                        "url": "https://example.com",
                        "title": "Example Page",
                        "content": "Snippet text...",
                        "score": 0.95
                    }
                ]
            }
        """
        raw_results = data.get("results")
        if not isinstance(raw_results, list):
            logger.debug("TavilySearchBackend: no 'results' key in response")
            return []

        results: list[SearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = item.get("url", "")
            if not url or not isinstance(url, str):
                continue
            raw_score = item.get("score", 0.0)
            try:
                normalised = max(0.0, min(1.0, float(raw_score)))
            except (TypeError, ValueError):
                normalised = 0.0
            results.append(
                SearchResult(
                    url=url,
                    title=str(item.get("title", "")),
                    snippet=str(item.get("content", "")),
                    score=normalised,
                )
            )
        return results
