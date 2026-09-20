"""Tests for SDK configuration and environment handling."""

from __future__ import annotations

import pytest

from predictron_sdk import config as config_mod
from predictron_sdk.config import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT_SECONDS,
    SDKConfig,
    build_config,
    config_from_env,
    endpoint_url,
    join_url,
    normalize_base_url,
)
from predictron_sdk.retry import RetryPolicy


def test_default_config_values() -> None:
    cfg = SDKConfig()
    assert cfg.base_url == DEFAULT_BASE_URL
    assert cfg.api_prefix == "/api/v1"
    assert cfg.timeout == DEFAULT_TIMEOUT_SECONDS
    assert cfg.max_retries == 4
    assert cfg.api_key is None
    assert cfg.verify_tls is True
    assert cfg.trust_env is True


def test_sdk_config_is_immutable() -> None:
    cfg = SDKConfig(api_key="k")
    with pytest.raises(AttributeError):
        cfg.api_key = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"api_key": "abc"},
        {"base_url": "https://x.example.com"},
        {"timeout": 5.0},
        {"max_retries": 7},
        {"api_prefix": "/custom"},
        {"verify_tls": False},
        {"trust_env": False},
    ],
)
def test_sdk_config_kwargs(kwargs) -> None:
    cfg = SDKConfig(**kwargs)
    for key, value in kwargs.items():
        assert getattr(cfg, key) == value


@pytest.mark.parametrize("base_url", ["https://a.com/", "https://a.com///"])
def test_normalize_base_url_strips_trailing_slashes(base_url) -> None:
    assert normalize_base_url(base_url) == "https://a.com"


def test_normalize_base_url_no_slash() -> None:
    assert normalize_base_url("https://a.com") == "https://a.com"


@pytest.mark.parametrize(
    "base,path,expected",
    [
        ("https://a.com", "health", "https://a.com/health"),
        ("https://a.com/", "/health", "https://a.com/health"),
        ("https://a.com", "/api/v1", "https://a.com/api/v1"),
    ],
)
def test_join_url(base, path, expected) -> None:
    assert join_url(base, path) == expected


@pytest.mark.parametrize("path", ["http://other.example/x", "https://other.example/y"])
def test_join_url_passes_through_absolute_urls(path) -> None:
    assert join_url("https://a.com", path) == path


def test_endpoint_url_with_prefix() -> None:
    assert (
        endpoint_url("https://a.com", "/api/v1", "/venture")
        == "https://a.com/api/v1/venture"
    )


def test_endpoint_url_without_prefix_slash() -> None:
    assert (
        endpoint_url("https://a.com", "/api/v1", "health")
        == "https://a.com/api/v1/health"
    )


def test_endpoint_url_empty_prefix() -> None:
    assert endpoint_url("https://a.com", "", "health") == "https://a.com/health"


def test_endpoint_url_absolute_path_passthrough() -> None:
    assert (
        endpoint_url("https://a.com", "/api/v1", "https://b.com/health")
        == "https://b.com/health"
    )


def test_config_from_env_empty(monkeypatch) -> None:
    monkeypatch.delenv("PREDICTRON_API_KEY", raising=False)
    monkeypatch.delenv("PREDICTRON_BASE_URL", raising=False)
    monkeypatch.delenv("PREDICTRON_TIMEOUT", raising=False)
    monkeypatch.delenv("PREDICTRON_MAX_RETRIES", raising=False)
    cfg = config_from_env()
    assert cfg.api_key is None
    assert cfg.base_url == DEFAULT_BASE_URL
    assert cfg.timeout == DEFAULT_TIMEOUT_SECONDS
    assert cfg.max_retries == 4


def test_config_from_env_populated(monkeypatch) -> None:
    monkeypatch.setenv("PREDICTRON_API_KEY", "env-key")
    monkeypatch.setenv("PREDICTRON_BASE_URL", "https://env.example.com")
    monkeypatch.setenv("PREDICTRON_TIMEOUT", "12.5")
    monkeypatch.setenv("PREDICTRON_MAX_RETRIES", "9")
    cfg = config_from_env()
    assert cfg.api_key == "env-key"
    assert cfg.base_url == "https://env.example.com"
    assert cfg.timeout == 12.5
    assert cfg.max_retries == 9


@pytest.mark.parametrize(
    "raw,expected",
    [("not-a-number", DEFAULT_TIMEOUT_SECONDS), ("", DEFAULT_TIMEOUT_SECONDS), ("0", 0.0)],
)
def test_config_from_env_bad_timeout(monkeypatch, raw, expected) -> None:
    monkeypatch.setenv("PREDICTRON_TIMEOUT", raw)
    assert config_from_env().timeout == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("not-a-number", 4), ("", 4), ("3", 3), ("-1", -1)],
)
def test_config_from_env_bad_max_retries(monkeypatch, raw, expected) -> None:
    monkeypatch.setenv("PREDICTRON_MAX_RETRIES", raw)
    assert config_from_env().max_retries == expected


def test_build_config_explicit_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("PREDICTRON_API_KEY", "env-key")
    monkeypatch.setenv("PREDICTRON_BASE_URL", "https://env.example.com")
    cfg = build_config(api_key="explicit", base_url="https://explicit.example.com")
    assert cfg.api_key == "explicit"
    assert cfg.base_url == "https://explicit.example.com"


def test_build_config_env_fallback(monkeypatch) -> None:
    monkeypatch.setenv("PREDICTRON_API_KEY", "env-key")
    cfg = build_config()
    assert cfg.api_key == "env-key"


def test_build_config_defaults() -> None:
    monkeypatch = pytest.MonkeyPatch()
    for name in (
        "PREDICTRON_API_KEY",
        "PREDICTRON_BASE_URL",
        "PREDICTRON_TIMEOUT",
        "PREDICTRON_MAX_RETRIES",
    ):
        monkeypatch.delenv(name, raising=False)
    cfg = build_config(timeout=3.0, max_retries=2)
    assert cfg.timeout == 3.0
    assert cfg.max_retries == 2
    assert cfg.base_url == DEFAULT_BASE_URL


def test_build_config_retry_policy_passthrough() -> None:
    policy = RetryPolicy(max_retries=1)
    cfg = build_config(retry_policy=policy)
    assert cfg.retry_policy is policy


def test_build_config_additional_headers() -> None:
    cfg = build_config(additional_headers={"X-Custom": "v"})
    assert cfg.additional_headers == {"X-Custom": "v"}


def test_build_config_verify_tls_false() -> None:
    assert build_config(verify_tls=False).verify_tls is False


def test_default_retry_policy_is_shared() -> None:
    assert SDKConfig().retry_policy is SDKConfig().retry_policy


def test_config_module_constants() -> None:
    assert config_mod.PREDICTRON_API_KEY_ENV == "PREDICTRON_API_KEY"
    assert config_mod.PREDICTRON_BASE_URL_ENV == "PREDICTRON_BASE_URL"
    assert config_mod.PREDICTRON_TIMEOUT_ENV == "PREDICTRON_TIMEOUT"
    assert config_mod.PREDICTRON_MAX_RETRIES_ENV == "PREDICTRON_MAX_RETRIES"
    assert config_mod.MAX_LIMIT == 100


def test_max_retries_negative_raises() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(max_retries=-1)
