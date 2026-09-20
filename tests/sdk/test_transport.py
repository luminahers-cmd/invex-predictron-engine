"""Tests for the HTTP transport layer."""

from __future__ import annotations

import json

import httpx
import pytest

from predictron_sdk.auth import BearerTokenAuth
from predictron_sdk.config import SDKConfig
from predictron_sdk.errors import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    SDKError,
    SerializationError,
    ServerError,
    ValidationError,
)
from predictron_sdk.errors import (
    TimeoutError as SDKTimeoutError,
)
from predictron_sdk.models import Health
from predictron_sdk.retry import RetryPolicy
from predictron_sdk.transport import APIResponse, Transport, parse_model
from tests.sdk.conftest import json_response


def build_transport(
    handler, *, policy: RetryPolicy | None = None, sleep=None, **config_kwargs
) -> Transport:
    auth = BearerTokenAuth("tok")
    http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://unused",
    )
    return Transport(
        config=SDKConfig(base_url="https://api.test", api_key="k", **config_kwargs),
        auth=auth,
        retry_policy=policy or RetryPolicy(max_retries=0),
        http_client=http_client,
        sleep_fn=sleep if sleep is not None else (lambda _: None),
    )


def test_get_basic() -> None:
    calls: list[httpx.Request] = []

    def handler(request):
        calls.append(request)
        assert request.url.path == "/api/v1/health"
        assert request.headers["Authorization"] == "Bearer tok"
        return json_response(200, {"status": "ok", "version": "1.0.0"})

    transport = build_transport(handler)
    response = transport.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"
    assert response.method == "GET"
    assert response.url == "https://api.test/api/v1/health"
    assert "Bearer tok" in calls[0].headers["Authorization"]


def test_post_json() -> None:
    def handler(request):
        payload = json.loads(request.content)
        assert payload == {"a": 1}
        return json_response(200, {"ok": True})

    transport = build_transport(handler)
    response = transport.post("/v", json={"a": 1})
    assert response.json == {"ok": True}


def test_response_request_id_header() -> None:
    def handler(request):
        return json_response(200, {}, **{"X-Request-ID": "rid-1"})

    transport = build_transport(handler)
    assert transport.get("/v").request_id == "rid-1"


def test_response_no_request_id() -> None:
    def handler(request):
        return json_response(200, {})

    transport = build_transport(handler)
    assert transport.get("/v").request_id is None


def test_transport_applies_api_prefix() -> None:
    def handler(request):
        assert request.url.path == "/api/v1/venture"
        return json_response(200, {})

    transport = build_transport(handler)
    transport.get("/venture")


def test_custom_api_prefix() -> None:
    def handler(request):
        assert request.url.path == "/custom/venture"
        return json_response(200, {})

    transport = build_transport(handler, api_prefix="/custom")
    transport.get("/venture")


def test_error_mapping_401() -> None:
    transport = build_transport(
        lambda r: json_response(401, {"detail": "unauthorized", "code": "NO_AUTH"})
    )
    with pytest.raises(AuthenticationError) as excinfo:
        transport.get("/analyze")
    err = excinfo.value
    assert err.code == "NO_AUTH"
    assert err.status_code == 401


def test_error_mapping_404() -> None:
    transport = build_transport(lambda r: json_response(404, {"detail": "nope"}))
    with pytest.raises(NotFoundError):
        transport.get("/analyze/x")


def test_error_mapping_422() -> None:
    transport = build_transport(lambda r: json_response(422, {"detail": []}))
    with pytest.raises(ValidationError):
        transport.get("/analyze")


def test_error_mapping_429() -> None:
    transport = build_transport(
        lambda r: json_response(429, {"detail": "slow"}, **{"Retry-After": "10"})
    )
    with pytest.raises(RateLimitError) as excinfo:
        transport.get("/analyze")
    assert excinfo.value.retry_after == 10.0


def test_error_mapping_500() -> None:
    transport = build_transport(lambda r: json_response(500, {"detail": "oops"}))
    with pytest.raises(ServerError) as excinfo:
        transport.get("/analyze")
    assert excinfo.value.status_code == 500


def test_error_mapping_non_standard_4xx() -> None:
    transport = build_transport(lambda r: json_response(418, {"detail": "teapot"}))
    with pytest.raises(SDKError) as excinfo:
        transport.get("/analyze")
    assert isinstance(excinfo.value, SDKError)


def test_empty_body_parses_none() -> None:
    transport = build_transport(lambda r: httpx.Response(204))
    response = transport.get("/v")
    assert response.json is None


def test_non_json_body_parses_none() -> None:
    transport = build_transport(lambda r: httpx.Response(200, text="not json"))
    assert transport.get("/v").json is None


def test_retry_503_then_success() -> None:
    sleeps: list[float] = []
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return json_response(503, {"detail": "busy"})
        return json_response(200, {"ok": True})

    transport = build_transport(
        handler,
        policy=RetryPolicy(
            max_retries=2, base_delay=0.1, backoff_factor=2.0
        ),
        sleep=sleeps.append,
    )
    response = transport.get("/v")
    assert response.status_code == 200
    assert attempts["n"] == 2
    assert sleeps == [0.1]


def test_retry_gives_up_after_max() -> None:
    sleeps: list[float] = []

    def handler(request):
        return json_response(503, {"detail": "busy"})

    transport = build_transport(
        handler,
        policy=RetryPolicy(max_retries=2, base_delay=0.05, backoff_factor=2.0),
        sleep=sleeps.append,
    )
    with pytest.raises(ServerError):
        transport.get("/v")
    assert len(sleeps) == 2


def test_retry_429_respects_retry_after() -> None:
    sleeps: list[float] = []

    def handler(request):
        return json_response(429, {"detail": "slow"}, **{"Retry-After": "1"})

    transport = build_transport(
        handler,
        policy=RetryPolicy(max_retries=1, base_delay=0.01, backoff_factor=2.0),
        sleep=sleeps.append,
    )
    with pytest.raises(RateLimitError):
        transport.get("/v")
    assert sleeps[0] == 1.0


def test_retry_disabled_when_use_retry_after_false() -> None:
    sleeps: list[float] = []

    def handler(request):
        return json_response(429, {"detail": "slow"}, **{"Retry-After": "1"})

    transport = build_transport(
        handler,
        policy=RetryPolicy(
            max_retries=1,
            base_delay=0.01,
            backoff_factor=2.0,
            use_retry_after=False,
        ),
        sleep=sleeps.append,
    )
    with pytest.raises(RateLimitError):
        transport.get("/v")
    assert sleeps[0] == 0.01


def test_no_retries_when_max_is_zero() -> None:
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return json_response(503, {})

    transport = build_transport(handler, policy=RetryPolicy(max_retries=0))
    with pytest.raises(ServerError):
        transport.get("/v")
    assert attempts["n"] == 1


def test_no_retry_on_non_retryable_status() -> None:
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return json_response(500, {})

    transport = build_transport(
        handler, policy=RetryPolicy(max_retries=3, base_delay=0.01)
    )
    with pytest.raises(ServerError):
        transport.get("/v")
    assert attempts["n"] == 1


def test_timeout_raises_typed_and_retries() -> None:
    sleeps: list[float] = []
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ReadTimeout("slow")
        return json_response(200, {"ok": True})

    transport = build_transport(
        handler,
        policy=RetryPolicy(max_retries=2, base_delay=0.1, backoff_factor=2.0),
        sleep=sleeps.append,
    )
    response = transport.get("/v")
    assert response.status_code == 200
    assert attempts["n"] == 2


def test_timeout_exhausted_raises() -> None:
    def handler(request):
        raise httpx.ReadTimeout("slow")

    transport = build_transport(
        handler,
        policy=RetryPolicy(
            max_retries=1, base_delay=0.01, retry_on_timeout=True
        ),
    )
    with pytest.raises(SDKTimeoutError):
        transport.get("/v")


def test_timeout_not_retried_when_disabled() -> None:
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        raise httpx.ReadTimeout("slow")

    transport = build_transport(
        handler,
        policy=RetryPolicy(
            max_retries=3, retry_on_timeout=False
        ),
    )
    with pytest.raises(SDKTimeoutError):
        transport.get("/v")
    assert attempts["n"] == 1


def test_connect_error_raises_sdk_error() -> None:
    def handler(request):
        raise httpx.ConnectError("refused")

    transport = build_transport(handler, policy=RetryPolicy(max_retries=0))
    with pytest.raises(SDKError):
        transport.get("/v")


def test_connect_error_retries_then_raises() -> None:
    sleeps: list[float] = []
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("refused")
        return json_response(200, {"ok": True})

    transport = build_transport(
        handler,
        policy=RetryPolicy(max_retries=2, base_delay=0.1),
        sleep=sleeps.append,
    )
    assert transport.get("/v").status_code == 200
    assert attempts["n"] == 2


def test_closed_transport_raises() -> None:
    transport = build_transport(lambda r: json_response(200, {}))
    transport.close()
    assert transport.is_closed
    with pytest.raises(RuntimeError):
        transport.get("/v")


def test_transport_context_manager() -> None:
    with build_transport(lambda r: json_response(200, {})) as transport:
        assert transport.get("/v").status_code == 200
    assert transport.is_closed


def test_auth_property() -> None:
    transport = build_transport(lambda r: json_response(200, {}))
    assert isinstance(transport.auth, BearerTokenAuth)


def test_custom_headers_merged() -> None:
    def handler(request):
        assert request.headers["X-Extra"] == "1"
        return json_response(200, {})

    transport = build_transport(handler)
    transport.get("/v", headers={"X-Extra": "1"})


def test_api_response_repr() -> None:
    response = APIResponse(
        status_code=200,
        headers={},
        text="{}",
        json={},
        method="GET",
        url="https://x",
    )
    assert "GET" in repr(response)
    assert "200" in repr(response)


def test_parse_model_success() -> None:
    model = parse_model(Health, {"status": "ok", "version": "1.0"})
    assert isinstance(model, Health)
    assert model.version == "1.0"


def test_parse_model_invalid_raises_serialization_error() -> None:
    with pytest.raises(SerializationError):
        parse_model(Health, {"status": 123})


def test_parse_model_partial_data_uses_defaults() -> None:
    model = parse_model(Health, {"status": "degraded"})
    assert model.version == ""
    assert model.db_healthy is False
