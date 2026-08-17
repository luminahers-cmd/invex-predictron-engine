"""Search backend abstraction for the Internet Search Discovery Layer.

Defines the protocol that any search provider (Tavily, Brave, Bing, Google
CSE, SerpAPI, DuckDuckGo, internal index, etc.) must implement to plug into
the :class:`~predictron_engine.evidence.search_provider.SearchEvidenceProvider`.

The layer is deliberately thin: a backend returns raw
:class:`SearchResult` items; all scoring, deduplication, and filtering
lives in the ranking module so backends remain stateless and testable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """One item returned by a search backend.

    Backends populate whatever fields they can provide.  Fields left at
    their defaults are treated as *unknown* by downstream consumers — no
    backend is required to fill every column.
    """

    url: str = Field(..., description="Discovered URL (raw, as returned by the backend)")
    title: str = Field(default="", description="Page title from search results")
    snippet: str = Field(default="", description="Brief text excerpt from the result")
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Backend-native relevance score in [0, 1], if available",
    )


class SearchSettings(BaseModel):
    """Configuration for a search discovery run.

    ``backend`` is a human-readable label for diagnostics; the actual
    backend instance is injected at construction time so there is never
    any string-based dispatch.
    """

    max_results: int = Field(
        default=20,
        ge=1,
        description="Maximum candidate URLs to request from the backend",
    )
    timeout: float = Field(
        default=10.0,
        gt=0,
        description="Per-query timeout in seconds",
    )
    backend: str = Field(
        default="noop",
        description="Display name of the enabled backend (for diagnostics only)",
    )
    retry_count: int = Field(
        default=1,
        ge=0,
        description="Number of retries on backend failure",
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0,
        description="Seconds to wait between retries",
    )


@runtime_checkable
class SearchBackend(Protocol):
    """Protocol every search backend must satisfy.

    Implementations are expected to be *stateless* — all mutable state
    (HTTP clients, API keys, connection pools) lives inside the
    implementation and is managed through ``close`` / async-context-manager
    protocols as needed.
    """

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Execute *query* and return up to *max_results* results.

        Must never raise for transient network or API errors; return an
        empty list instead.  Backend-specific exceptions may propagate —
        the provider layer catches them.
        """
        ...

    async def close(self) -> None:
        """Release any resources held by the backend."""
        ...
