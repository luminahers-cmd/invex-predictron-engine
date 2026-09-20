"""Tests for the deterministic retry policy."""

from __future__ import annotations

import httpx
import pytest

from predictron_sdk.retry import (
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_DELAY,
    DEFAULT_MAX_RETRIES,
    RETRYABLE_STATUS_CODES,
    RetryPolicy,
    delay_for_attempt,
    retry_after_seconds,
    should_retry_error,
    should_retry_status,
)


def test_default_retry_statuses() -> None:
    assert RETRYABLE_STATUS_CODES == frozenset({429, 502, 503, 504})


def test_default_policy_values() -> None:
    policy = RetryPolicy()
    assert policy.max_retries == DEFAULT_MAX_RETRIES
    assert policy.base_delay == DEFAULT_BASE_DELAY
    assert policy.max_delay == DEFAULT_MAX_DELAY
    assert policy.backoff_factor == DEFAULT_BACKOFF_FACTOR
    assert policy.retry_on_timeout is True
    assert policy.retry_on_connect_error is True


@pytest.mark.parametrize(
    "status,expected",
    [
        (429, True),
        (502, True),
        (503, True),
        (504, True),
        (500, False),
        (501, False),
        (404, False),
        (401, False),
        (200, False),
    ],
)
def test_should_retry_status(status, expected) -> None:
    assert should_retry_status(status, RetryPolicy()) is expected


def test_should_retry_status_custom_set() -> None:
    policy = RetryPolicy(retry_statuses=frozenset({500}))
    assert should_retry_status(500, policy) is True
    assert should_retry_status(429, policy) is False


def test_should_retry_timeout() -> None:
    assert should_retry_error(httpx.ReadTimeout("t"), RetryPolicy()) is True
    assert (
        should_retry_error(
            httpx.ReadTimeout("t"), RetryPolicy(retry_on_timeout=False)
        )
        is False
    )


def test_should_retry_connect_error() -> None:
    assert should_retry_error(httpx.ConnectError("c"), RetryPolicy()) is True
    assert (
        should_retry_error(
            httpx.ConnectError("c"), RetryPolicy(retry_on_connect_error=False)
        )
        is False
    )


def test_should_retry_network_error() -> None:
    assert should_retry_error(httpx.NetworkError("n"), RetryPolicy()) is True


def test_should_retry_other_exception() -> None:
    assert should_retry_error(RuntimeError("other"), RetryPolicy()) is False


def test_exponential_backoff_values() -> None:
    policy = RetryPolicy(base_delay=1.0, backoff_factor=2.0, max_delay=32.0)
    assert delay_for_attempt(0, policy) == 1.0
    assert delay_for_attempt(1, policy) == 2.0
    assert delay_for_attempt(2, policy) == 4.0
    assert delay_for_attempt(3, policy) == 8.0
    assert delay_for_attempt(4, policy) == 16.0
    assert delay_for_attempt(5, policy) == 32.0


def test_backoff_capped_at_max_delay() -> None:
    policy = RetryPolicy(base_delay=1.0, backoff_factor=10.0, max_delay=5.0)
    assert delay_for_attempt(3, policy) == 5.0
    assert delay_for_attempt(100, policy) == 5.0


def test_deterministic_schedule_no_jitter() -> None:
    policy = RetryPolicy(base_delay=0.5, backoff_factor=3.0, max_delay=10.0)
    attempts = list(range(6))
    first = [delay_for_attempt(a, policy) for a in attempts]
    second = [delay_for_attempt(a, policy) for a in attempts]
    assert first == second


def test_zero_retries_policy() -> None:
    policy = RetryPolicy(max_retries=0)
    assert policy.max_retries == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_retries": -1},
        {"base_delay": -0.5},
        {"max_delay": 5.0, "base_delay": 10.0},
        {"backoff_factor": 0.5},
    ],
)
def test_invalid_policy_values_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


def test_max_delay_equal_base_accepted() -> None:
    policy = RetryPolicy(base_delay=2.0, max_delay=2.0)
    assert delay_for_attempt(1, policy) == 2.0


def test_retry_after_seconds_digits() -> None:
    assert retry_after_seconds({"Retry-After": "5"}) == 5.0


def test_retry_after_seconds_none() -> None:
    assert retry_after_seconds(None) is None
    assert retry_after_seconds({}) is None
    assert retry_after_seconds({"Other": "1"}) is None


def test_retry_after_seconds_invalid_returns_none() -> None:
    assert retry_after_seconds({"Retry-After": "garbage"}) is None


def test_retry_after_http_date_parses() -> None:
    value = retry_after_seconds(
        {"Retry-After": "2026-10-21T07:28:00+00:00"}
    )
    assert value is not None
    assert value >= 0


def test_retry_after_white_space() -> None:
    assert retry_after_seconds({"Retry-After": " 7 "}) == 7.0


def test_policy_frozen() -> None:
    policy = RetryPolicy()
    with pytest.raises(Exception):
        policy.max_retries = 2  # type: ignore[misc]


def test_policy_equality() -> None:
    assert RetryPolicy(max_retries=2) == RetryPolicy(max_retries=2)
    assert RetryPolicy(max_retries=2) != RetryPolicy(max_retries=3)
