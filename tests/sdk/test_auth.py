"""Tests for SDK authentication providers."""

from __future__ import annotations

import pytest

from predictron_sdk.auth import (
    ApiKeyAuth,
    AuthProvider,
    BearerTokenAuth,
    CompositeAuth,
    NullAuth,
    StaticHeaderAuth,
    auth_from_api_key,
)


def test_null_auth_headers_empty() -> None:
    assert NullAuth().headers() == {}


def test_null_auth_apply_is_noop() -> None:
    assert NullAuth().apply({"A": "b"}) == {"A": "b"}


def test_bearer_token_headers() -> None:
    auth = BearerTokenAuth("tok-123")
    assert auth.headers() == {"Authorization": "Bearer tok-123"}


def test_bearer_token_apply() -> None:
    headers = BearerTokenAuth("tok").apply({"X": "y"})
    assert headers["Authorization"] == "Bearer tok"
    assert headers["X"] == "y"


def test_bearer_token_rejects_empty() -> None:
    with pytest.raises(ValueError):
        BearerTokenAuth("")


def test_api_key_headers_default_name() -> None:
    assert ApiKeyAuth("k").headers() == {"X-API-Key": "k"}


@pytest.mark.parametrize("name", ["X-Api-Key", "api-key", "Apikey"])
def test_api_key_custom_header_name(name) -> None:
    assert ApiKeyAuth("k", header_name=name).headers() == {name: "k"}


def test_api_key_rejects_empty_value() -> None:
    with pytest.raises(ValueError):
        ApiKeyAuth("")


def test_api_key_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ApiKeyAuth("k", header_name="")


def test_static_header_auth() -> None:
    auth = StaticHeaderAuth("X-Tenant", "acme")
    assert auth.headers() == {"X-Tenant": "acme"}


def test_static_header_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        StaticHeaderAuth("", "v")


def test_composite_auth_merges() -> None:
    auth = CompositeAuth(BearerTokenAuth("t"), ApiKeyAuth("k"))
    assert auth.headers() == {
        "Authorization": "Bearer t",
        "X-API-Key": "k",
    }


def test_composite_auth_first_wins_on_collision() -> None:
    auth = CompositeAuth(
        BearerTokenAuth("a"),
        StaticHeaderAuth("Authorization", "Custom a"),
    )
    assert auth.headers()["Authorization"] == "Bearer a"


def test_composite_auth_empty() -> None:
    assert CompositeAuth().headers() == {}


def test_auth_provider_apply_does_not_override_custom_headers() -> None:
    auth = BearerTokenAuth("t")
    merged = auth.apply({"Authorization": "Bearer custom"})
    assert merged["Authorization"] == "Bearer custom"


def test_auth_from_api_key_returns_bearer() -> None:
    provider = auth_from_api_key("key")
    assert isinstance(provider, BearerTokenAuth)
    assert provider.headers()["Authorization"] == "Bearer key"


def test_auth_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        AuthProvider()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "provider_factory,expected",
    [
        (lambda: BearerTokenAuth("x"), {"Authorization": "Bearer x"}),
        (lambda: ApiKeyAuth("x"), {"X-API-Key": "x"}),
        (lambda: StaticHeaderAuth("H", "v"), {"H": "v"}),
        (lambda: NullAuth(), {}),
    ],
)
def test_provider_apply_patterns(provider_factory, expected) -> None:
    provider = provider_factory()
    result = provider.apply()
    for key, value in expected.items():
        assert result[key] == value


def test_apply_returns_new_dict() -> None:
    base = {"a": "1"}
    result = NullAuth().apply(base)
    base["a"] = "2"
    assert result["a"] == "1"
