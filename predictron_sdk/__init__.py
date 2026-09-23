"""Predictron SDK — the official Python client for the Predictron API.

Everything deterministic. No ML, no LLM, no embeddings. The SDK wraps the
public API surface with strongly-typed models and typed exceptions.
"""

from __future__ import annotations

from predictron_sdk.auth import (
    ApiKeyAuth,
    AuthProvider,
    BearerTokenAuth,
    CompositeAuth,
    NullAuth,
    StaticHeaderAuth,
    auth_from_api_key,
)
from predictron_sdk.batch import (
    TERMINAL_STATUSES,
    BatchTimeoutError,
    batch_items_from_names,
    build_batch_items,
    items_from_requests,
    wait_for_completion,
)
from predictron_sdk.client import (
    AnalyzeResource,
    BatchResource,
    CompaniesResource,
    CompareResource,
    DueDiligenceResource,
    HealthResource,
    LearningResource,
    PortfolioResource,
    PredictronClient,
    SearchResource,
    VentureResource,
    make_bearer_client,
)
from predictron_sdk.config import (
    SDKConfig,
    build_config,
    config_from_env,
)
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
)
from predictron_sdk.models import JobStatus
from predictron_sdk.pagination import Page, Paginator
from predictron_sdk.retry import RetryPolicy
from predictron_sdk.stream import BatchJobStream, progress_events
from predictron_sdk.version import __version__

__all__ = [
    "__version__",
    # Client
    "PredictronClient",
    "make_bearer_client",
    # Resources
    "AnalyzeResource",
    "BatchResource",
    "CompaniesResource",
    "CompareResource",
    "DueDiligenceResource",
    "HealthResource",
    "LearningResource",
    "PortfolioResource",
    "SearchResource",
    "VentureResource",
    # Auth
    "AuthProvider",
    "NullAuth",
    "BearerTokenAuth",
    "ApiKeyAuth",
    "StaticHeaderAuth",
    "CompositeAuth",
    "auth_from_api_key",
    # Errors
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
    # Config / retry / pagination
    "SDKConfig",
    "build_config",
    "config_from_env",
    "RetryPolicy",
    "Paginator",
    "Page",
    # Batch / stream
    "JobStatus",
    "TERMINAL_STATUSES",
    "batch_items_from_names",
    "build_batch_items",
    "items_from_requests",
    "wait_for_completion",
    "BatchTimeoutError",
    "BatchJobStream",
    "progress_events",
]
