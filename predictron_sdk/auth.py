"""Authentication strategies for the Predictron SDK.

The SDK supports bearer tokens, API-key headers and arbitrary static headers.
Custom headers passed by the caller always take precedence over the values an
auth provider would inject.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = [
    "AuthProvider",
    "NullAuth",
    "BearerTokenAuth",
    "ApiKeyAuth",
    "StaticHeaderAuth",
    "CompositeAuth",
    "auth_from_api_key",
]


class AuthProvider(ABC):
    """Strategy that contributes authentication headers to each request."""

    @abstractmethod
    def headers(self) -> dict[str, str]:
        """Return the authentication headers this provider contributes."""
        raise NotImplementedError

    def apply(self, headers: dict[str, str] | None = None) -> dict[str, str]:
        """Merge auth headers into ``headers`` without overriding existing values."""
        merged = dict(headers or {})
        for name, value in self.headers().items():
            merged.setdefault(name, value)
        return merged


class NullAuth(AuthProvider):
    """No authentication — sends no identity headers."""

    def headers(self) -> dict[str, str]:
        return {}


class BearerTokenAuth(AuthProvider):
    """Authenticate with an ``Authorization: Bearer <token>`` header."""

    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("Bearer token must not be empty")
        self.token = token

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


class ApiKeyAuth(AuthProvider):
    """Authenticate with a custom API-key header (default ``X-API-Key``)."""

    def __init__(self, api_key: str, header_name: str = "X-API-Key") -> None:
        if not api_key:
            raise ValueError("API key must not be empty")
        if not header_name:
            raise ValueError("header name must not be empty")
        self.api_key = api_key
        self.header_name = header_name

    def headers(self) -> dict[str, str]:
        return {self.header_name: self.api_key}


class StaticHeaderAuth(AuthProvider):
    """Authenticate (or add context) with an arbitrary static header."""

    def __init__(self, header_name: str, header_value: str) -> None:
        if not header_name:
            raise ValueError("header name must not be empty")
        self.header_name = header_name
        self.header_value = header_value

    def headers(self) -> dict[str, str]:
        return {self.header_name: self.header_value}


class CompositeAuth(AuthProvider):
    """Combine multiple auth providers into a single header set."""

    def __init__(self, *providers: AuthProvider) -> None:
        self.providers = list(providers)

    def headers(self) -> dict[str, str]:
        merged: dict[str, str] = {}
        for provider in self.providers:
            for name, value in provider.headers().items():
                merged.setdefault(name, value)
        return merged


def auth_from_api_key(api_key: str) -> AuthProvider:
    """Build a bearer-token auth provider from an API key."""
    return BearerTokenAuth(api_key)
