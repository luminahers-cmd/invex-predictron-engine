"""Configuration for the Predictron SDK.

Configuration can be supplied programmatically, via environment variables, or
both. Environment variables are read lazily through :func:`config_from_env`
and merged with explicit constructor arguments in :class:`PredictronClient`.

Supported environment variables:

* ``PREDICTRON_API_KEY`` — default API key / bearer token.
* ``PREDICTRON_BASE_URL`` — default API base URL.
* ``PREDICTRON_TIMEOUT`` — request timeout in seconds.
* ``PREDICTRON_MAX_RETRIES`` — maximum number of retries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field as dataclass_field

from predictron_sdk.retry import DEFAULT_MAX_RETRIES, RetryPolicy

__all__ = [
    "SDKConfig",
    "config_from_env",
    "build_config",
    "normalize_base_url",
    "join_url",
    "DEFAULT_BASE_URL",
    "DEFAULT_API_PREFIX",
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_LIMIT",
    "PREDICTRON_API_KEY_ENV",
    "PREDICTRON_BASE_URL_ENV",
]

PREDICTRON_API_KEY_ENV = "PREDICTRON_API_KEY"
PREDICTRON_BASE_URL_ENV = "PREDICTRON_BASE_URL"
PREDICTRON_TIMEOUT_ENV = "PREDICTRON_TIMEOUT"
PREDICTRON_MAX_RETRIES_ENV = "PREDICTRON_MAX_RETRIES"

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_API_PREFIX = "/api/v1"
DEFAULT_TIMEOUT_SECONDS = 30.0

#: Maximum page size accepted by list endpoints.
MAX_LIMIT = 100

_DEFAULT_RETRY_POLICY = RetryPolicy()


@dataclass(frozen=True, slots=True)
class SDKConfig:
    """Immutable SDK configuration."""

    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    api_prefix: str = DEFAULT_API_PREFIX
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_policy: RetryPolicy = dataclass_field(default_factory=lambda: _DEFAULT_RETRY_POLICY)
    additional_headers: dict[str, str] = dataclass_field(default_factory=dict)
    verify_tls: bool = True
    trust_env: bool = True


def normalize_base_url(base_url: str) -> str:
    """Strip trailing slashes so URL joining is deterministic."""
    return base_url.rstrip("/")


def join_url(base_url: str, path: str) -> str:
    """Join a normalized base URL with a path/prefix."""
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return f"{normalize_base_url(base_url)}/{path.lstrip('/')}"


def endpoint_url(base_url: str, api_prefix: str, path: str) -> str:
    """Build the fully-qualified URL for an API path."""
    if path.startswith("http://") or path.startswith("https://"):
        return path
    prefix = api_prefix.strip("/")
    if prefix:
        return f"{normalize_base_url(base_url)}/{prefix}/{path.lstrip('/')}"
    return f"{normalize_base_url(base_url)}/{path.lstrip('/')}"


def _parse_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def config_from_env() -> SDKConfig:
    """Build a :class:`SDKConfig` from the process environment."""
    return SDKConfig(
        api_key=os.environ.get(PREDICTRON_API_KEY_ENV) or None,
        base_url=os.environ.get(PREDICTRON_BASE_URL_ENV) or DEFAULT_BASE_URL,
        timeout=_parse_float_env(PREDICTRON_TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS),
        max_retries=_parse_int_env(PREDICTRON_MAX_RETRIES_ENV, DEFAULT_MAX_RETRIES),
    )


def build_config(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    retry_policy: RetryPolicy | None = None,
    additional_headers: dict[str, str] | None = None,
    verify_tls: bool | None = None,
    trust_env: bool | None = None,
) -> SDKConfig:
    """Merge explicit overrides, the environment, and defaults into a config."""
    env_config = config_from_env()
    effective_max_retries = (
        max_retries if max_retries is not None else env_config.max_retries
    )
    if retry_policy is not None:
        effective_policy = retry_policy
    elif max_retries is not None:
        effective_policy = RetryPolicy(max_retries=max_retries)
    else:
        effective_policy = env_config.retry_policy
    return SDKConfig(
        api_key=api_key if api_key is not None else env_config.api_key,
        base_url=base_url if base_url is not None else env_config.base_url,
        timeout=timeout if timeout is not None else env_config.timeout,
        max_retries=effective_max_retries,
        retry_policy=effective_policy,
        additional_headers=(
            dict(additional_headers or {})
            if additional_headers is not None
            else dict(env_config.additional_headers)
        ),
        verify_tls=verify_tls if verify_tls is not None else True,
        trust_env=trust_env if trust_env is not None else True,
    )
