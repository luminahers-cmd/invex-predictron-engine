from __future__ import annotations

import logging
import time
from collections.abc import Callable
from collections.abc import Set as AbstractSet

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from app.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory sliding-window rate limiter.

    Limits the number of requests per client IP within a configurable time
    window.  When the limit is exceeded the middleware returns HTTP 429
    *without* forwarding the request to the inner application.

    Excluded paths (health, readiness, docs) are never rate-limited.
    """

    _windows: dict[str, list[float]] = {}

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
        cls._windows.clear()

    def _client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "127.0.0.1"

    def _clean_expired(self, client_ip: str, window_seconds: float) -> None:
        now = time.monotonic()
        timestamps = type(self)._windows.get(client_ip, [])
        type(self)._windows[client_ip] = [t for t in timestamps if now - t < window_seconds]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)

        path = request.url.path.rstrip("/")
        if path in self._exclude_paths:
            return await call_next(request)

        client_ip = self._client_ip(request)
        max_requests = settings.RATE_LIMIT_REQUESTS
        window_seconds = settings.RATE_LIMIT_WINDOW_SECONDS

        self._clean_expired(client_ip, float(window_seconds))

        window = type(self)._windows.setdefault(client_ip, [])
        if len(window) >= max_requests:
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
                headers={"Retry-After": str(window_seconds)},
            )

        window.append(time.monotonic())
        return await call_next(request)
