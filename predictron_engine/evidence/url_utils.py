"""Shared URL utilities for the Evidence Collection Layer.

Centralizes URL normalization, canonicalization, parsing, and host
extraction so that ``WebsiteEvidenceProvider``, ``SearchEvidenceProvider``,
and ``ranking`` all use identical logic without duplication.

Every function in this module is a pure helper with no side effects.
"""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import HttpUrl, ValidationError

from predictron_engine.evidence.exceptions import InvalidWebsiteError


def has_valid_scheme(url: str) -> bool:
    """Return ``True`` if *url* looks like it has an HTTP or HTTPS scheme."""
    lower = url.strip().lower()
    return lower.startswith("http://") or lower.startswith("https://")


def ensure_scheme(raw: str) -> str:
    """Return *raw* with ``https://`` prepended when no scheme is present."""
    stripped = raw.strip()
    if not has_valid_scheme(stripped):
        return f"https://{stripped}"
    return stripped


def extract_host(url: str) -> str:
    """Return the lowercased, ``www.``-stripped hostname, or empty string."""
    try:
        host = urlparse(url).hostname or ""
    except Exception:  # noqa: BLE001 — malformed URL
        return ""
    return host.lower().removeprefix("www.")


def normalise_url_for_dedup(url: str) -> str:
    """Return a canonical form of *url* suitable for deduplication.

    The output strips the scheme, normalises the hostname (lowercase,
    ``www.``-stripped), and removes trailing slashes from the path.
    """
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001 — malformed URL, return as-is
        return url.strip().lower()
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/") or ""
    return f"{host}{path}"


def parse_http_url(url: str) -> HttpUrl | None:
    """Parse *url* into an :class:`HttpUrl`, returning ``None`` on failure."""
    try:
        return HttpUrl(ensure_scheme(url))
    except (ValidationError, Exception):  # noqa: BLE001
        return None


def normalise_website(website: HttpUrl | str | None) -> HttpUrl:
    """Parse and normalize a website value into an :class:`HttpUrl`.

    Raises :class:`InvalidWebsiteError` when the input is missing,
    blank, or cannot be parsed as a valid HTTP URL.
    """
    if isinstance(website, HttpUrl):
        return website
    if not isinstance(website, str) or not website.strip():
        raise InvalidWebsiteError("A website URL is required")
    try:
        return HttpUrl(ensure_scheme(website))
    except (ValidationError, Exception) as exc:
        raise InvalidWebsiteError(f"Invalid website URL: {website.strip()!r}") from exc
