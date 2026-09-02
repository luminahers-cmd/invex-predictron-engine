"""Thread-safe, memory-bounded sliding-window rate limiter.

Design notes
------------
* **Thread safety.** All access to the shared in-memory window store is
  guarded by a :class:`threading.Lock`.  The critical sections never await,
  so the event loop is only blocked for a few microseconds per request and
  the store is safe even when the middleware is reached from multiple
  threads.

* **Bounded memory.** Each client's timestamp list is capped at the
  configured request limit (a sliding window never needs more than
  ``limit`` entries), and the number of distinct tracked clients is capped
  at ``RATE_LIMIT_MAX_TRACKED_CLIENTS``.  Keys are rotated to the end of
  the store on every hit (LRU order) and the least-recently-seen clients
  are evicted when the cap is reached.

* **Trusted proxy support.** ``X-Forwarded-For`` is completely ignored
  unless ``RATE_LIMIT_TRUSTED_PROXIES`` is configured.  When it is, the
  header is only honoursed when the socket peer is itself a trusted proxy,
  and the client address is resolved by walking the header from the nearest
  entry to the farthest, skipping trusted proxies.  Spoofed headers sent by
  non-proxy clients are therefore ignored.

Limits the number of requests per client IP within a configurable time
window.  When the limit is exceeded the middleware returns HTTP 429
*without* forwarding the request to the inner application.

Excluded paths (health, readiness, docs) are never rate-limited.
"""

from __future__ import annotations

import ipaddress
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from app.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


type _IPNetwork = "ipaddress._BaseNetwork[Any]"


@lru_cache(maxsize=8)
def _compiled_networks(raw: tuple[str, ...]) -> tuple[_IPNetwork, ...]:
    """Compile raw IP/CIDR strings into networks, ignoring invalid entries."""
    networks: list[_IPNetwork] = []
    for entry in raw:
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            logger.warning("Ignoring invalid trusted proxy entry: %r", entry)
    return tuple(networks)


def _is_parseable_ip(host: str) -> bool:
    """Return True when ``host`` is a valid IP literal."""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _is_trusted_proxy(host: str, networks: tuple[_IPNetwork, ...]) -> bool:
    """Return True when ``host`` falls inside one of the trusted networks."""
    try:
        peer_ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(peer_ip in network for network in networks)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory sliding-window rate limiter keyed by resolved client IP.

    The store is a class-level dict guarded by a lock.  ``reset_windows``
    is provided for tests.  X-Forwarded-For is only honoured when operated
    behind a configured trusted proxy (see module docstring).
    """

    _windows: dict[str, list[float]] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: AbstractSet[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._exclude_paths: AbstractSet[str] = frozenset(
            exclude_paths or {
                "/api/v1/health",
                "/api/v1/health/readiness",
                "/docs",
                "/redoc",
                "/openapi.json",
            }
        )

    @classmethod
    def reset_windows(cls) -> None:
        """Clear all rate-limit windows (useful for testing)."""
        with cls._lock:
            cls._windows.clear()

    def _client_ip(self, request: Request) -> str:
        """Resolve the true client address, guarding against XFF spoofing.

        Returns the socket peer unless the peer is a *trusted proxy*, in
        which case the right-most untrusted ``X-Forwarded-For`` entry is
        used.  When no trusted proxies are configured the header is never
        consulted.
        """
        peer = request.client.host if request.client else "127.0.0.1"

        raw_proxies = tuple(settings.RATE_LIMIT_TRUSTED_PROXIES)
        if not raw_proxies:
            return peer

        networks = _compiled_networks(raw_proxies)
        if not _is_trusted_proxy(peer, networks):
            return peer

        forwarded = request.headers.get("X-Forwarded-For", "")
        addresses = [addr.strip() for addr in forwarded.split(",") if addr.strip()]
        for address in reversed(addresses):
            if not _is_parseable_ip(address):
                return peer
            if not _is_trusted_proxy(address, networks):
                return address
        return peer

    def _record_hit(
        self,
        client_ip: str,
        max_requests: int,
        window_seconds: float,
    ) -> tuple[bool, float]:
        """Record a request for ``client_ip``.

        Returns ``(allowed, retry_after)``.  When ``allowed`` is False the
        request is over the limit and ``retry_after`` is the number of
        seconds until the oldest in-window request expires.  Runs entirely
        under the store lock and never awaits, so it is safe to call from
        the event loop.
        """
        now = time.monotonic()
        with self._lock:
            timestamps = self._windows.pop(client_ip, None)
            if timestamps is None:
                timestamps = []
            else:
                timestamps = [ts for ts in timestamps if now - ts < window_seconds]

            if len(timestamps) >= max_requests:
                retry_after = max(window_seconds - (now - timestamps[0]), 0.0)
                self._windows[client_ip] = timestamps
                return False, retry_after

            timestamps.append(now)
            if len(timestamps) > max_requests:
                del timestamps[0]
            self._windows[client_ip] = timestamps
            self._evict_if_needed()
            return True, 0.0

    def _evict_if_needed(self) -> None:
        """Evict the least-recently-seen clients when the key cap is exceeded."""
        max_clients = settings.RATE_LIMIT_MAX_TRACKED_CLIENTS
        while len(self._windows) > max_clients:
            # Keys are rotated to the end on every hit, so the first key in
            # insertion order is the least-recently-seen client.
            oldest = next(iter(self._windows))
            del self._windows[oldest]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)

        path = request.url.path.rstrip("/")
        if path in self._exclude_paths:
            return await call_next(request)

        client_ip = self._client_ip(request)
        max_requests = settings.RATE_LIMIT_REQUESTS
        window_seconds = float(settings.RATE_LIMIT_WINDOW_SECONDS)

        allowed, retry_after = self._record_hit(
            client_ip, max_requests, window_seconds
        )
        if not allowed:
            request_id = getattr(request.state, "request_id", None)
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "request_id": request_id,
                    "client_ip": client_ip,
                    "method": request.method,
                    "path": path,
                    "limit": max_requests,
                    "window_seconds": window_seconds,
                },
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Try again later.",
                    "code": "RATE_LIMIT_EXCEEDED",
                },
                headers={"Retry-After": str(int(retry_after) or 1)},
            )

        return await call_next(request)
