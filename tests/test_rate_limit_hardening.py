"""Unit tests for rate-limiter production hardening.

Covers memory bounding, LRU eviction, thread-safe store updates, and
trusted-proxy `X-Forwarded-For` resolution.  These tests exercise the
middleware's internals directly so they run fast and deterministically.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import Request

from app.core.config import get_settings
from app.middleware.rate_limit import RateLimitMiddleware


class _DummyApp:
    """Minimal ASGI app stub sufficient to construct the middleware."""

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        return None


@pytest.fixture()
def _middleware() -> RateLimitMiddleware:
    RateLimitMiddleware.reset_windows()
    return RateLimitMiddleware(_DummyApp())


def _request_with_peer(host: str, forwarded: str | None = None) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/analyze/some-id",
        "headers": [],
        "client": (host, 12345),
        "query_string": b"",
        "server": ("testserver", 80),
        "scheme": "http",
    }
    if forwarded is not None:
        scope["headers"].append(
            (b"x-forwarded-for", forwarded.encode("utf-8"))
        )
    return Request(scope)


class TestThreadSafety:
    def test_concurrent_record_hits_never_exceed_limit(
        self, _middleware: RateLimitMiddleware
    ) -> None:
        settings = get_settings()
        with patch.object(settings, "RATE_LIMIT_MAX_TRACKED_CLIENTS", 100):
            errors: list[str] = []

            def work() -> None:
                try:
                    for _ in range(50):
                        allowed, _ = _middleware._record_hit("1.2.3.4", 10, 60.0)
                        if not allowed:
                            break
                except Exception as exc:  # noqa: BLE001 - surfaced in main thread
                    errors.append(str(exc))

            threads = [threading.Thread(target=work) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors
            stored = RateLimitMiddleware._windows.get("1.2.3.4", [])
            # The sliding window must never exceed the configured limit.
            assert len(stored) == 10

    def test_clean_expired_entries(self, _middleware: RateLimitMiddleware) -> None:
        RateLimitMiddleware._windows["1.2.3.4"] = [
            time.monotonic() - 120.0,  # long expired
            time.monotonic(),          # fresh
        ]
        allowed, _ = _middleware._record_hit("1.2.3.4", 10, 60.0)
        assert allowed
        stored = RateLimitMiddleware._windows["1.2.3.4"]
        assert len(stored) == 2  # expired dropped, new hit appended


class TestMemoryBounds:
    def test_per_client_list_is_capped(self, _middleware: RateLimitMiddleware) -> None:
        for _ in range(100):
            _middleware._record_hit("9.9.9.9", 5, 60.0)
        stored = RateLimitMiddleware._windows["9.9.9.9"]
        assert len(stored) == 5

    def test_client_cap_evicts_lru_keys(
        self, _middleware: RateLimitMiddleware
    ) -> None:
        settings = get_settings()
        with patch.object(settings, "RATE_LIMIT_MAX_TRACKED_CLIENTS", 3):
            for ip in ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]:
                _middleware._record_hit(ip, 5, 60.0)
            assert len(RateLimitMiddleware._windows) == 3
            # The oldest-inserted key must have been evicted.
            assert "10.0.0.1" not in RateLimitMiddleware._windows

    def test_revisiting_a_client_keeps_it_tracked(
        self, _middleware: RateLimitMiddleware
    ) -> None:
        settings = get_settings()
        with patch.object(settings, "RATE_LIMIT_MAX_TRACKED_CLIENTS", 2):
            _middleware._record_hit("10.0.0.1", 5, 60.0)
            _middleware._record_hit("10.0.0.2", 5, 60.0)
            # Re-touch 10.0.0.1 so it becomes most-recently-used.
            _middleware._record_hit("10.0.0.1", 5, 60.0)
            _middleware._record_hit("10.0.0.3", 5, 60.0)
            assert "10.0.0.1" in RateLimitMiddleware._windows
            assert "10.0.0.2" not in RateLimitMiddleware._windows


class TestTrustedProxyResolution:
    def test_xff_ignored_without_trusted_proxies(
        self, _middleware: RateLimitMiddleware
    ) -> None:
        settings = get_settings()
        with patch.object(settings, "RATE_LIMIT_TRUSTED_PROXIES", []):
            request = _request_with_peer("198.51.100.7", forwarded="6.6.6.6")
            ip = _middleware._client_ip(request)
            assert ip == "198.51.100.7"

    def test_spoofed_xff_from_untrusted_peer_ignored(
        self, _middleware: RateLimitMiddleware
    ) -> None:
        settings = get_settings()
        with patch.object(settings, "RATE_LIMIT_TRUSTED_PROXIES", ["203.0.113.0/24"]):
            # The peer is NOT a trusted proxy, so the XFF header is spoofed.
            request = _request_with_peer("198.51.100.7", forwarded="6.6.6.6")
            ip = _middleware._client_ip(request)
            assert ip == "198.51.100.7"

    def test_trusted_proxy_resolves_nearest_untrusted_xff(
        self, _middleware: RateLimitMiddleware
    ) -> None:
        settings = get_settings()
        with patch.object(settings, "RATE_LIMIT_TRUSTED_PROXIES", ["203.0.113.0/24"]):
            # peer is a trusted proxy; the right-most *untrusted* hop is the client.
            request = _request_with_peer(
                "203.0.113.9",
                forwarded="6.6.6.6, 203.0.113.10, 203.0.113.11",
            )
            ip = _middleware._client_ip(request)
            assert ip == "6.6.6.6"

    def test_trusted_proxy_malformed_header_falls_back_to_peer(
        self, _middleware: RateLimitMiddleware
    ) -> None:
        settings = get_settings()
        with patch.object(settings, "RATE_LIMIT_TRUSTED_PROXIES", ["203.0.113.0/24"]):
            request = _request_with_peer("203.0.113.9", forwarded="garbage")
            ip = _middleware._client_ip(request)
            assert ip == "203.0.113.9"


class TestRetryAfter:
    def test_over_limit_returns_positive_retry_after(
        self, _middleware: RateLimitMiddleware
    ) -> None:
        RateLimitMiddleware._windows["5.5.5.5"] = [time.monotonic(), time.monotonic()]
        allowed, retry_after = _middleware._record_hit("5.5.5.5", 2, 60.0)
        assert not allowed
        assert retry_after > 0.0
