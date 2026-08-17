"""Shared fixtures for Evidence Collection Layer tests."""

from __future__ import annotations

import pytest
from pydantic import HttpUrl


@pytest.fixture
def site_url() -> HttpUrl:
    """The canonical website used across collection tests."""
    return HttpUrl("https://example.com")
