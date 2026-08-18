"""Provider contracts for the Evidence Collection Layer.

Providers are pluggable, self-contained sources of evidence (website
content today; social profiles, app-store listings, etc. in future
sprints). Each provider decides whether it applies to a given collection
context, collects evidence, and reports its own documents, provenance,
and timing. The orchestrator merges provider results into a single
:class:`EvidenceBundle` and isolates failures so one provider can never
abort a run.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, HttpUrl

from predictron_engine.evidence.models import EvidenceDocument, EvidenceSource


class CollectContext(BaseModel):
    """Inputs an evidence provider may use to decide and collect.

    ``website`` is the raw website string supplied to the collection
    request (possibly missing). Providers are free to ignore fields they
    do not need; the model will grow as future providers require richer
    inputs.
    """

    startup_name: str = Field(..., description="Company name")
    website: str | None = Field(default=None, description="Raw website string, if any")


class PrioritizationSummary(BaseModel):
    """Diagnostics for evidence prioritization and page selection.

    Included in :class:`ProviderResult` when the search provider
    performs prioritization.  Always optional and additive — never
    breaks existing consumers.
    """

    detected_official_url: str | None = Field(
        default=None,
        description="URL of the identified official website",
    )
    official_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for official website detection",
    )
    selected_count: int = Field(
        default=0,
        ge=0,
        description="Number of pages selected for fetching",
    )
    skipped_count: int = Field(
        default=0,
        ge=0,
        description="Number of pages skipped (not selected)",
    )
    evidence_types: dict[str, int] = Field(
        default_factory=dict,
        description="Count of pages by evidence type",
    )


class ProviderResult(BaseModel):
    """Output produced by one evidence provider for a collection run."""

    provider: str = Field(..., description="Name of the provider that produced this result")
    website: HttpUrl | None = Field(
        default=None, description="Canonical source URL collected, when applicable"
    )
    documents: list[EvidenceDocument] = Field(
        default_factory=list, description="Collected documents"
    )
    sources: list[EvidenceSource] = Field(
        default_factory=list, description="Provenance for every attempted source"
    )
    attempted_pages: int = Field(default=0, description="Number of sources attempted")
    duration_ms: int = Field(default=0, description="Wall-clock time of the provider run")
    success: bool = Field(default=True, description="Whether collection completed successfully")
    failure_reason: str | None = Field(
        default=None, description="Why the provider failed, if it did"
    )
    prioritization: PrioritizationSummary | None = Field(
        default=None,
        description="Prioritization diagnostics when applicable (search provider)",
    )


@runtime_checkable
class EvidenceProvider(Protocol):
    """A pluggable source of evidence for a startup analysis."""

    name: str

    def can_collect(self, context: CollectContext) -> bool:
        """Return True when this provider can collect for the given context."""
        ...

    async def collect(self, context: CollectContext) -> ProviderResult:
        """Collect evidence, never raising for page-level failures."""
        ...
