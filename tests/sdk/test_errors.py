"""Tests for SDK typed exceptions."""

from __future__ import annotations

import pytest

from predictron_sdk.errors import (
    APIError,
    AuthenticationError,
    ConflictError,
    NotFoundError,
    RateLimitError,
    SDKError,
    SerializationError,
    ServerError,
    TimeoutError,
    ValidationError,
    error_for_status,
    message_from_payload,
)


def test_sdk_error_message_and_code() -> None:
    err = SDKError("boom", code="E123")
    assert err.message == "boom"
    assert err.code == "E123"
    assert str(err) == "boom"


def test_sdk_error_defaults() -> None:
    err = SDKError("boom")
    assert err.code is None


def test_api_error_attributes() -> None:
    err = APIError(
        "nope",
        status_code=400,
        code="BAD",
        method="POST",
        url="https://x/v",
        headers={"X-Test": "1"},
        response_body={"detail": "nope"},
    )
    assert err.status_code == 400
    assert err.method == "POST"
    assert err.url == "https://x/v"
    assert err.headers == {"X-Test": "1"}
    assert err.response_body == {"detail": "nope"}


def test_api_error_str_includes_status() -> None:
    assert "HTTP 400" in str(APIError("x", status_code=400))


def test_api_error_str_without_status() -> None:
    assert str(APIError("x")) == "x"


def test_exception_hierarchy() -> None:
    assert issubclass(AuthenticationError, APIError)
    assert issubclass(RateLimitError, APIError)
    assert issubclass(NotFoundError, APIError)
    assert issubclass(ConflictError, APIError)
    assert issubclass(ValidationError, APIError)
    assert issubclass(ServerError, APIError)
    assert issubclass(APIError, SDKError)
    assert issubclass(TimeoutError, SDKError)
    assert issubclass(SerializationError, SDKError)


def test_rate_limit_retry_after() -> None:
    err = RateLimitError("slow down", status_code=429, retry_after=2.5)
    assert err.retry_after == 2.5


def test_rate_limit_retry_after_defaults_none() -> None:
    assert RateLimitError("x").retry_after is None


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, AuthenticationError),
        (404, NotFoundError),
        (409, ConflictError),
        (422, ValidationError),
        (429, RateLimitError),
        (500, ServerError),
        (502, ServerError),
        (503, ServerError),
        (504, ServerError),
        (399, APIError),
        (400, APIError),
        (403, APIError),
    ],
)
def test_error_for_status_maps_correct_type(status, expected) -> None:
    err = error_for_status(status)
    assert isinstance(err, expected)
    assert err.status_code == status


def test_error_for_status_message_from_detail() -> None:
    err = error_for_status(400, response_body={"detail": "bad payload"})
    assert err.message == "bad payload"
    assert err.code is None


def test_error_for_status_message_from_code() -> None:
    err = error_for_status(
        400, response_body={"detail": "x", "code": "INVALID_INPUT"}
    )
    assert err.code == "INVALID_INPUT"


def test_error_for_status_retry_after_mapped() -> None:
    err = error_for_status(
        429, headers={"Retry-After": "3"}
    )
    assert isinstance(err, RateLimitError)
    assert err.retry_after == 3.0


def test_error_for_status_no_retry_after() -> None:
    err = error_for_status(429)
    assert err.retry_after is None


@pytest.mark.parametrize(
    "payload,substr",
    [
        ({"detail": "d"}, "d"),
        ({"detail": "d", "code": "c"}, "d"),
        ({"code": "C"}, "C"),
        ({"detail": 5}, "5"),
        (None, "HTTP"),
        ({"detail": ""}, "HTTP"),
    ],
)
def test_message_from_payload(payload, substr) -> None:
    assert substr in message_from_payload(payload, 400)


def test_message_from_payload_list_of_dicts() -> None:
    payload = {
        "detail": [
            {"msg": "field required", "loc": ["query", "q"]},
            {"msg": "too long"},
        ]
    }
    message = message_from_payload(payload, 422)
    assert "field required" in message
    assert "too long" in message


def test_message_from_payload_list_of_strings() -> None:
    payload = {"detail": ["a", "b"]}
    assert message_from_payload(payload, 422) == "a; b"


def test_message_from_payload_empty_list() -> None:
    message = message_from_payload({"detail": []}, 422)
    assert "HTTP 422" in message


def test_api_error_headers_are_copy() -> None:
    source = {"A": "1"}
    err = APIError("x", headers=source)
    source["A"] = "2"
    assert err.headers["A"] == "1"


def test_sdk_error_is_an_exception() -> None:
    assert issubclass(SDKError, Exception)


def test_raising_and_catching_typed() -> None:
    with pytest.raises(AuthenticationError) as excinfo:
        raise AuthenticationError("no token", status_code=401)
    assert excinfo.value.status_code == 401


@pytest.mark.parametrize(
    "cls",
    [
        AuthenticationError,
        NotFoundError,
        ConflictError,
        ValidationError,
        ServerError,
    ],
)
def test_typed_errors_carry_api_metadata(cls) -> None:
    err = cls("m", status_code=400, method="GET", url="u", response_body={"x": 1})
    assert isinstance(err, APIError)
    assert err.status_code == 400
    assert err.response_body == {"x": 1}


def test_retry_after_http_date() -> None:
    err = error_for_status(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
    assert isinstance(err, RateLimitError)
    assert err.retry_after is not None


def test_serialization_error_message() -> None:
    err = SerializationError("bad model")
    assert "bad model" in str(err)
