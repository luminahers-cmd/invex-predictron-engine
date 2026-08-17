"""Asynchronous HTTP page fetcher.

Implements deterministic retrieval with configurable timeouts, retries,
redirect following, a custom User-Agent, and an explicit response-size
cap. Page-level failures are returned as :class:`FetchResult` records
rather than raised, so collection can always continue with partial
evidence.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
from pydantic import BaseModel, Field, HttpUrl

from predictron_engine.evidence.models import FetchResult

logger = logging.getLogger(__name__)

_USER_AGENT = "InveX-EvidenceProvider/1.0 (+https://invex.ai; evidence collection)"


class FetcherSettings(BaseModel):
    """Tunable settings for the HTTP page fetcher."""

    timeout_seconds: float = Field(default=10.0, description="Overall request timeout")
    connect_timeout_seconds: float = Field(default=5.0, description="TCP/TLS connect timeout")
    max_retries: int = Field(
        default=2, ge=0, description="Number of retries for transient failures"
    )
    retry_backoff_ms: int = Field(default=250, ge=0, description="Sleep between retries")
    follow_redirects: bool = Field(default=True, description="Follow HTTP redirects")
    user_agent: str = Field(default=_USER_AGENT, description="User-Agent header")
    max_bytes: int = Field(default=5_000_000, gt=0, description="Hard cap on response body size")
    success_statuses: set[int] = Field(default_factory=lambda: {200, 201, 202, 203, 204, 205, 206})
    retryable_statuses: set[int] = Field(default_factory=lambda: {429, 500, 502, 503, 504})
    definitive_statuses: set[int] = Field(
        default_factory=lambda: {
            300, 301, 302, 303, 307, 308, 400, 401, 402, 403, 404, 405, 406, 410, 418
        }
    )


class HttpPageFetcher:
    """Async fetcher built on httpx with retries, redirects, and a size cap.

    Parameters
    ----------
    settings:
        Optional :class:`FetcherSettings`; sane defaults are used otherwise.
    transport:
        Optional httpx transport (e.g. ``httpx.MockTransport``) for tests and
        for swapping in custom connection strategies.
    """

    def __init__(
        self,
        settings: FetcherSettings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings or FetcherSettings()
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def fetch(self, url: HttpUrl) -> FetchResult:
        """Fetch a single page, returning a FetchResult instead of raising."""
        client = await self._ensure_client()
        last_error: str | None = None
        last_status: int | None = None
        last_final_url = str(url)
        last_content_type: str | None = None
        last_ms = 0

        for attempt in range(self._settings.max_retries + 1):
            started = time.monotonic()
            try:
                async with client.stream("GET", str(url)) as response:
                    elapsed = int((time.monotonic() - started) * 1000)
                    last_ms = elapsed
                    last_status = response.status_code
                    last_final_url = str(response.url)
                    last_content_type = response.headers.get("content-type")

                    if response.status_code in self._settings.success_statuses:
                        content_length = response.headers.get("content-length")
                        exceeded = (
                            content_length
                            and content_length.isdigit()
                            and int(content_length) > self._settings.max_bytes
                        )
                        if exceeded:
                            return FetchResult(
                                requested_url=url,
                                final_url=HttpUrl(last_final_url),
                                status=response.status_code,
                                response_time_ms=elapsed,
                                content_type=last_content_type,
                                truncated=True,
                                retry_count=attempt,
                                error="Response body exceeds the configured size cap",
                            )
                        body, truncated = await self._read_limited(response)
                        if truncated:
                            logger.warning("Truncated oversized response from %s", url)
                        return FetchResult(
                            requested_url=url,
                            final_url=HttpUrl(last_final_url),
                            status=response.status_code,
                            response_time_ms=elapsed,
                            html=self._decode(body, response.encoding),
                            content_type=last_content_type,
                            truncated=truncated,
                            retry_count=attempt,
                        )

                    if response.status_code in self._settings.definitive_statuses:
                        return FetchResult(
                            requested_url=url,
                            final_url=HttpUrl(last_final_url),
                            status=response.status_code,
                            response_time_ms=elapsed,
                            content_type=last_content_type,
                            retry_count=attempt,
                            error=f"HTTP {response.status_code}",
                        )

                    last_error = f"HTTP {response.status_code}"
            except httpx.TransportError as exc:
                elapsed = int((time.monotonic() - started) * 1000)
                last_ms = elapsed
                last_error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # pragma: no cover - defensive
                elapsed = int((time.monotonic() - started) * 1000)
                last_ms = elapsed
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < self._settings.max_retries:
                await asyncio.sleep(self._settings.retry_backoff_ms / 1000.0)

        return FetchResult(
            requested_url=url,
            final_url=HttpUrl(last_final_url),
            status=last_status or 0,
            response_time_ms=last_ms,
            content_type=last_content_type,
            retry_count=self._settings.max_retries,
            error=last_error or "Fetch failed",
        )

    async def _read_limited(self, response: httpx.Response) -> tuple[bytes, bool]:
        chunks: list[bytes] = []
        total = 0
        truncated = False
        async for chunk in response.aiter_bytes():
            remaining = self._settings.max_bytes - total
            if remaining <= 0:
                truncated = True
                break
            take = min(len(chunk), remaining)
            chunks.append(chunk[:take])
            total += take
            if take < len(chunk):
                truncated = True
                break
        return b"".join(chunks), truncated

    @staticmethod
    def _decode(body: bytes, encoding: str | None) -> str:
        try:
            return body.decode(encoding or "utf-8", errors="replace")
        except LookupError:
            return body.decode("utf-8", errors="replace")

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(
                self._settings.timeout_seconds, connect=self._settings.connect_timeout_seconds
            )
            self._client = httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=self._settings.follow_redirects,
                transport=self._transport,
                headers={"User-Agent": self._settings.user_agent},
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying client, releasing pooled connections."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> HttpPageFetcher:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()
