"""Deterministic retry policy for the Predictron SDK.

The policy is fully deterministic: no cryptographic or random jitter is
applied, so identical failures produce an identical retry schedule. Retries
are only attempted for transient failures (HTTP 429/502/503/504, timeouts and
connection errors) and the total number of retries is bounded.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

__all__ = [
    "RetryPolicy",
    "RETRYABLE_STATUS_CODES",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_BASE_DELAY",
    "DEFAULT_MAX_DELAY",
    "DEFAULT_BACKOFF_FACTOR",
    "delay_for_attempt",
    "should_retry_status",
    "should_retry_error",
    "that_failed",
    "retry_after_seconds",
]

#: Status codes that are safe to retry idempotently.
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 502, 503, 504})

DEFAULT_MAX_RETRIES = 4
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 32.0
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_TIMEOUT_RETRIES = 2


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Deterministic exponential-backoff retry configuration."""

    max_retries: int = DEFAULT_MAX_RETRIES
    base_delay: float = DEFAULT_BASE_DELAY
    max_delay: float = DEFAULT_MAX_DELAY
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR
    retry_statuses: frozenset[int] = field(
        default_factory=lambda: RETRYABLE_STATUS_CODES
    )
    retry_on_timeout: bool = True
    retry_on_connect_error: bool = True
    use_retry_after: bool = True

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.backoff_factor < 1.0:
            raise ValueError("backoff_factor must be >= 1.0")


def that_failed(exc: BaseException) -> bool:
    """Return ``True`` when a thrown exception is a transient failure."""
    return isinstance(exc, httpx.TimeoutException | httpx.ConnectError)


def delay_for_attempt(attempt: int, policy: RetryPolicy) -> float:
    """Exponential backoff delay for the n-th retry (1-indexed ``attempt``)."""
    if attempt <= 0:
        return policy.base_delay
    computed = policy.base_delay * (policy.backoff_factor ** attempt)
    return min(computed, policy.max_delay)


def should_retry_status(status: int, policy: RetryPolicy) -> bool:
    """Return ``True`` when an HTTP status is worth retrying."""
    return status in policy.retry_statuses


def should_retry_error(exc: BaseException, policy: RetryPolicy) -> bool:
    """Return ``True`` when a transport exception is worth retrying."""
    if isinstance(exc, httpx.TimeoutException):
        return policy.retry_on_timeout
    if isinstance(exc, httpx.ConnectError | httpx.NetworkError):
        return policy.retry_on_connect_error
    return False


def retry_after_seconds(headers: Mapping[str, str] | None) -> float | None:
    """Parse the ``Retry-After`` header into seconds (if present)."""
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
