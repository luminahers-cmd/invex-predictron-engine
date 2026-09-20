"""Shared fixtures for the API test package.

``sqlite_engine`` / ``sqlite_session`` are provided by ``tests/conftest.py``
(root) so the API suite reuses the real store against an isolated in-memory
database.
"""

from __future__ import annotations

import pytest

from app.middleware.rate_limit import RateLimitMiddleware


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate-limiter window store before each test.

    The middleware keeps a module-level window store keyed by client IP,
    so without a reset the entire suite shares a single 100 req/60s bucket.
    """
    RateLimitMiddleware.reset_windows()
    yield
