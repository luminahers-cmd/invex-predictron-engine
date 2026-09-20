"""Typed exceptions raised by the Predictron SDK.

Every error the SDK raises derives from :class:`SDKError`. Errors returned by
the Predictron API itself additionally derive from :class:`APIError` and
carry the HTTP status code, request metadata and the raw response body.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "SDKError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
    "ConflictError",
    "ValidationError",
    "ServerError",
    "TimeoutError",
    "SerializationError",
    "error_for_status",
    "message_from_payload",
]


class SDKError(Exception):
    """Base class for all Predictron SDK exceptions."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class APIError(SDKError):
    """Base class for errors returned by the Predictron API."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        method: str | None = None,
        url: str | None = None,
        headers: Mapping[str, str] | None = None,
        response_body: Any = None,
    ) -> None:
        super().__init__(message, code=code)
        self.status_code = status_code
        self.method = method
        self.url = url
        self.headers = dict(headers or {})
        self.response_body = response_body

    def __str__(self) -> str:
        base = super().__str__()
        if self.status_code is not None:
            return f"{base} (HTTP {self.status_code})"
        return base


class AuthenticationError(APIError):
    """HTTP 401 — missing, invalid or expired credentials."""


class RateLimitError(APIError):
    """HTTP 429 — the API rate limit was exceeded."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        method: str | None = None,
        url: str | None = None,
        headers: Mapping[str, str] | None = None,
        response_body: Any = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            code=code,
            method=method,
            url=url,
            headers=headers,
            response_body=response_body,
        )
        self.retry_after = retry_after


class NotFoundError(APIError):
    """HTTP 404 — the requested resource does not exist."""


class ConflictError(APIError):
    """HTTP 409 — the request conflicts with the current resource state."""


class ValidationError(APIError):
    """HTTP 422 — the request payload failed server-side validation."""


class ServerError(APIError):
    """HTTP 5xx — the server failed to process the request."""


class TimeoutError(SDKError):
    """The server did not respond before the configured timeout."""


class SerializationError(SDKError):
    """An API response could not be parsed into its typed model."""


_STATUS_ERROR_TYPES: dict[int, type[APIError]] = {
    401: AuthenticationError,
    404: NotFoundError,
    409: ConflictError,
    422: ValidationError,
    429: RateLimitError,
}


def message_from_payload(payload: Any, status: int | None = None) -> str:
    """Extract a human-readable message from an API error payload."""
    if isinstance(payload, Mapping):
        detail = payload.get("detail")
        code = payload.get("code")
        if isinstance(detail, str) and detail:
            return detail
        if detail and not isinstance(detail, list | dict):
            return str(detail)
        if isinstance(detail, list):
            parts = []
            for item in detail:
                if isinstance(item, Mapping):
                    parts.append(str(item.get("msg", "")) or str(item))
                else:
                    parts.append(str(item))
            message = "; ".join(part for part in parts if part)
            if message:
                return message
        if isinstance(detail, Mapping):
            return str(detail)
        hint = code if isinstance(code, str) and code else None
        if hint:
            return hint
    if status is not None:
        return f"Predictron API error (HTTP {status})"
    return "Predictron API error"


def _extract_code(payload: Any) -> str | None:
    if isinstance(payload, Mapping):
        code = payload.get("code")
        if isinstance(code, str) and code:
            return code
    return None


def error_for_status(
    status: int,
    *,
    method: str | None = None,
    url: str | None = None,
    headers: Mapping[str, str] | None = None,
    response_body: Any = None,
) -> APIError:
    """Build the appropriate typed exception for an HTTP status code."""
    message = message_from_payload(response_body, status)
    code = _extract_code(response_body)
    error_type = _STATUS_ERROR_TYPES.get(status)
    if error_type is None:
        if 500 <= status < 600:
            error_type = ServerError
        else:
            error_type = APIError
    kwargs: dict[str, Any] = {
        "status_code": status,
        "code": code,
        "method": method,
        "url": url,
        "headers": headers,
        "response_body": response_body,
    }
    if error_type is RateLimitError:
        retry_after = _retry_after_seconds(headers)
        kwargs["retry_after"] = retry_after
    return error_type(message, **kwargs)


def _retry_after_seconds(headers: Mapping[str, str] | None) -> float | None:
    if not headers:
        return None
    header = None
    for name in ("Retry-After", "retry-after"):
        value = headers.get(name)
        if value:
            header = value
            break
    if not header:
        return None
    stripped = header.strip()
    if stripped.isdigit():
        return float(int(stripped))
    target = _parse_http_date(stripped)
    if target is None:
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)
    remaining = (target - datetime.now(UTC)).total_seconds()
    return max(0.0, remaining)


def _parse_http_date(value: str) -> datetime | None:
    """Parse an ISO-8601 or RFC 7231 HTTP-date string into a datetime."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
