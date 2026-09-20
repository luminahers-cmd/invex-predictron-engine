"""Tests for the top-level client (config, auth, lifecycle, aliases)."""

from __future__ import annotations

import httpx

from predictron_sdk import (
    ApiKeyAuth,
    BearerTokenAuth,
    NullAuth,
    PredictronClient,
    RetryPolicy,
    StaticHeaderAuth,
    make_bearer_client,
)
from predictron_sdk.auth import CompositeAuth
from predictron_sdk.config import SDKConfig, normalize_base_url
from tests.sdk.conftest import json_response, record_calls


def test_client_requires_no_args_uses_defaults(monkeypatch) -> None:
    monkeypatch.delenv("PREDICTRON_API_KEY", raising=False)
    monkeypatch.delenv("PREDICTRON_BASE_URL", raising=False)
    client = PredictronClient()
    assert client.base_url == "http://localhost:8000"
    assert isinstance(client.auth, NullAuth)


def test_client_uses_env_api_key(monkeypatch) -> None:
    monkeypatch.setenv("PREDICTRON_API_KEY", "env-key")
    monkeypatch.delenv("PREDICTRON_BASE_URL", raising=False)
    client = PredictronClient()
    assert isinstance(client.auth, BearerTokenAuth)


def test_client_uses_env_base_url(monkeypatch) -> None:
    monkeypatch.delenv("PREDICTRON_API_KEY", raising=False)
    monkeypatch.setenv("PREDICTRON_BASE_URL", "https://env.example.com")
    client = PredictronClient()
    assert client.base_url == "https://env.example.com"


def test_client_timeout_enforced(monkeypatch) -> None:
    monkeypatch.setenv("PREDICTRON_TIMEOUT", "7.5")
    client = PredictronClient()
    assert client._config.timeout == 7.5


def test_from_env(monkeypatch) -> None:
    monkeypatch.setenv("PREDICTRON_API_KEY", "k")
    monkeypatch.setenv("PREDICTRON_BASE_URL", "https://e.example.com")
    client = PredictronClient.from_env()
    assert isinstance(client.auth, BearerTokenAuth)


def test_make_bearer_client() -> None:
    client = make_bearer_client("secret")
    assert isinstance(client.auth, BearerTokenAuth)


def test_client_api_key_parameter() -> None:
    client = PredictronClient(
        api_key="abc", transport=httpx.MockTransport(lambda r: json_response(200, {}))
    )
    assert isinstance(client.auth, BearerTokenAuth)


def test_client_custom_auth_provider_wins_over_api_key() -> None:
    auth = ApiKeyAuth("abc")
    client = PredictronClient(
        api_key="def",
        auth=auth,
        transport=httpx.MockTransport(lambda r: json_response(200, {})),
    )
    assert client.auth is auth


def test_client_composite_auth() -> None:
    calls: list[httpx.Request] = []
    auth = CompositeAuth(
        StaticHeaderAuth("X-Key", "k1"), StaticHeaderAuth("X-Tok", "t1")
    )
    client = PredictronClient(
        auth=auth,
        transport=httpx.MockTransport(record_calls(calls, lambda r: json_response(200, {}))),
    )
    client.health.check()
    assert calls[0].headers["X-Key"] == "k1"
    assert calls[0].headers["X-Tok"] == "t1"


def test_client_custom_headers() -> None:
    calls: list[httpx.Request] = []
    client = PredictronClient(
        api_key="k",
        headers={"X-Custom": "yes"},
        transport=httpx.MockTransport(record_calls(calls, lambda r: json_response(200, {}))),
    )
    client.health.check()
    assert calls[0].headers["X-Custom"] == "yes"


def test_client_custom_timeout_yields_float() -> None:
    client = PredictronClient(timeout=12.5)
    assert client._config.timeout == 12.5


def test_client_custom_retry_policy() -> None:
    policy = RetryPolicy(max_retries=3, base_delay=0.01)
    client = PredictronClient(retry_policy=policy)
    assert client._transport.retry_policy is policy


def test_client_max_retries_kwarg() -> None:
    client = PredictronClient(max_retries=2)
    assert client._config.retry_policy.max_retries == 2


def test_client_transport_injection(no_retries) -> None:
    calls: list[httpx.Request] = []
    mock = httpx.MockTransport(record_calls(calls, lambda r: json_response(200, {})))
    client = PredictronClient(transport=mock)
    client.health.check()
    assert len(calls) == 1


def test_client_context_manager() -> None:
    with PredictronClient(
        transport=httpx.MockTransport(lambda r: json_response(200, {}))
    ) as client:
        assert client.health.check().status == "ok"


def test_client_close() -> None:
    client = PredictronClient(
        transport=httpx.MockTransport(lambda r: json_response(200, {}))
    )
    client.close()
    assert client._transport.is_closed


def test_client_custom_http_client_not_closed() -> None:
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda r: json_response(200, {}))
    )
    client = PredictronClient(http_client=http_client)
    client.close()
    assert http_client.is_closed is False
    http_client.close()


def test_client_repr() -> None:
    client = PredictronClient(api_key="k")
    assert "PredictronClient" in repr(client)
    assert "BearerTokenAuth" in repr(client)


def test_client_resources_exposed() -> None:
    client = PredictronClient()
    assert client.venture is not None
    assert client.portfolio is not None
    assert client.batch is not None
    assert client.analyze is not None
    assert client.health is not None
    assert client.search_resource is not None
    assert client._compare is not None


def test_client_sleep_fn_used_for_retries() -> None:
    sleeps: list[float] = []
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return json_response(503, {})
        return json_response(200, {"status": "ok"})

    client = PredictronClient(
        transport=httpx.MockTransport(handler),
        retry_policy=RetryPolicy(max_retries=2, base_delay=0.1),
        sleep_fn=sleeps.append,
    )
    assert client.health.check().status == "ok"
    assert attempts["n"] == 2
    assert sleeps == [0.1]


def test_client_aliases_delegate() -> None:
    def handler(request):
        if request.url.path == "/api/v1/venture":
            return json_response(
                200,
                {
                    "startup_name": "A",
                    "overall_score": 1.0,
                    "overall_confidence": 0.5,
                },
            )
        if request.url.path == "/api/v1/health":
            return json_response(200, {"status": "ok"})
        return json_response(200, {})

    client = PredictronClient(transport=httpx.MockTransport(handler))
    assert client.evaluate(
        startup_name="A", description="A sufficiently long description."
    ).startup_name == "A"
    assert client.health_status().status == "ok"


def test_normalize_base_url() -> None:
    assert normalize_base_url("https://api.x") == "https://api.x"
    assert normalize_base_url("https://api.x/") == "https://api.x"


def test_sdkconfig_defaults() -> None:
    config = SDKConfig(base_url="https://api.x", api_key="k")
    assert config.timeout == 30.0
    assert config.api_prefix == "/api/v1"


def test_client_config_object() -> None:
    config = SDKConfig(
        base_url="https://api.x",
        api_key="k",
        timeout=3.0,
        additional_headers={"X-A": "b"},
    )
    client = PredictronClient(config=config)
    assert client._config is config


def test_client_auth_named_api_key_alias() -> None:
    client = PredictronClient(
        api_key="secret", transport=httpx.MockTransport(lambda r: json_response(200, {}))
    )
    assert client.auth is not None
