"""HTTP transport for the Predictron SDK.

The transport wraps ``httpx.Client``, injects authentication headers, applies
the deterministic retry policy and converts HTTP failures into typed SDK
exceptions. Lower-level network problems (DNS, refused connections, timeouts)
are also converted to typed SDK exceptions.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from predictron_sdk.auth import AuthProvider, NullAuth
from predictron_sdk.config import SDKConfig, endpoint_url
from predictron_sdk.errors import (
    SDKError,
    SerializationError,
    error_for_status,
)
from predictron_sdk.errors import (
    TimeoutError as SDKTimeoutError,
)
from predictron_sdk.retry import (
    RetryPolicy,
    delay_for_attempt,
    retry_after_seconds,
    should_retry_error,
    should_retry_status,
)

__all__ = ["Transport", "APIResponse", "SleepFn", "parse_model"]

SleepFn = Callable[[float], None]


class APIResponse:
    """A parsed API response envelope."""

    __slots__ = (
        "status_code",
        "headers",
        "text",
        "json",
        "method",
        "url",
        "request_id",
    )

    def __init__(
        self,
        *,
        status_code: int,
        headers: Mapping[str, str],
        text: str,
        json: Any,
        method: str,
        url: str,
    ) -> None:
        self.status_code = status_code
        self.headers = dict(headers)
        self.text = text
        self.json = json
        self.method = method
        self.url = url
        self.request_id = (
            self.headers.get("X-Request-ID")
            or self.headers.get("x-request-id")
            or None
        )

    def __repr__(self) -> str:
        return f"<APIResponse {self.method} {self.url} -> {self.status_code}>"


def _parse_json(text: str) -> Any:
    if not text:
        return None
    try:
        import json

        return json.loads(text)
    except ValueError:
        return None


class Transport:
    """Send prepared HTTP requests to the Predictron API."""

    def __init__(
        self,
        *,
        config: SDKConfig,
        auth: AuthProvider | None = None,
        retry_policy: RetryPolicy | None = None,
        http_client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep_fn: SleepFn = time.sleep,
    ) -> None:
        self._config = config
        self._auth = auth if auth is not None else NullAuth()
        self._retry_policy = retry_policy if retry_policy is not None else config.retry_policy
        self._sleep = sleep_fn

        from predictron_sdk.version import USER_AGENT

        if http_client is not None:
            self._client = http_client
            self._owns_client = False
        else:
            self._client = httpx.Client(
                transport=transport,
                timeout=config.timeout,
                verify=config.verify_tls,
                trust_env=config.trust_env,
                headers={
                    "User-Agent": USER_AGENT,
                    **config.additional_headers,
                },
            )
            self._owns_client = True
        self._closed = False

    @property
    def auth(self) -> AuthProvider:
        return self._auth

    @property
    def retry_policy(self) -> RetryPolicy:
        return self._retry_policy

    @property
    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Close the underlying HTTP client if the transport owns it."""
        if self._owns_client:
            self._client.close()
        self._closed = True

    def __enter__(self) -> Transport:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _endpoint(self, path: str) -> str:
        return endpoint_url(
            self._config.base_url, self._config.api_prefix, path
        )

    def _send_once(
        self,
        *,
        method: str,
        url: str,
        params: Mapping[str, str] | None,
        json: Any,
        headers: Mapping[str, str] | None,
    ) -> httpx.Response:
        return self._client.request(
            method,
            url,
            params=params,
            json=json,
            headers=headers,
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        json: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> APIResponse:
        """Send a request and return a parsed API response.

        Raises the appropriate typed exception on failure.
        """
        if self._closed:
            raise RuntimeError("Transport is closed")
        url = self._endpoint(path)
        merged_headers = self._auth.apply(dict(headers or {}))
        attempt = 0
        policy = self._retry_policy
        while True:
            try:
                response = self._send_once(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    headers=merged_headers,
                )
            except httpx.TimeoutException as exc:
                if attempt >= policy.max_retries or not policy.retry_on_timeout:
                    raise SDKTimeoutError(
                        f"Request to {method} {url} timed out"
                    ) from exc
                delay = delay_for_attempt(attempt, policy)
                self._sleep(delay)
                attempt += 1
                continue
            except httpx.TransportError as exc:
                if (
                    attempt >= policy.max_retries
                    or not policy.retry_on_connect_error
                    or not should_retry_error(exc, policy)
                ):
                    raise SDKError(
                        f"Connection error during {method} {url}: {exc}"
                    ) from exc
                delay = delay_for_attempt(attempt, policy)
                self._sleep(delay)
                attempt += 1
                continue

            status = response.status_code
            response_headers = response.headers
            if should_retry_status(status, policy) and attempt < policy.max_retries:
                delay = delay_for_attempt(attempt, policy)
                if policy.use_retry_after and status == 429:
                    retry_after = retry_after_seconds(response_headers)
                    if retry_after is not None:
                        delay = max(delay, retry_after)
                self._sleep(delay)
                attempt += 1
                continue

            payload = _parse_json(response.text)
            if 400 <= status < 600:
                raise error_for_status(
                    status,
                    method=method,
                    url=url,
                    headers=dict(response_headers.items()),
                    response_body=payload,
                )

            return APIResponse(
                status_code=status,
                headers=dict(response_headers.items()),
                text=response.text,
                json=payload,
                method=method,
                url=url,
            )

    def get(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> APIResponse:
        return self.request("GET", path, params=params, headers=headers)

    def post(
        self,
        path: str,
        *,
        json: Any = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> APIResponse:
        return self.request(
            "POST", path, params=params, json=json, headers=headers
        )


M = TypeVar("M", bound=BaseModel)


def parse_model(model_type: type[M], payload: Any) -> M:
    """Parse a JSON payload into a pydantic model, raising a typed error."""
    try:
        return model_type.model_validate(payload)
    except PydanticValidationError as exc:
        raise SerializationError(
            f"Failed to deserialize {model_type.__name__}: {exc}"
        ) from exc
    except Exception as exc:
        raise SerializationError(
            f"Failed to deserialize {model_type.__name__}: {exc}"
        ) from exc
