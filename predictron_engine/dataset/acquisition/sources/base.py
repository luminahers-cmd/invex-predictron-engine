"""Base source connector protocol for the acquisition framework.

Every source connector implements this protocol.  Connectors are
responsible for discovering, fetching, normalizing, and checkpointing
data from a specific public source.  They never know about
DatasetStore internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from predictron_engine.dataset.imports import RawImportRecord


@dataclass
class SourceDescriptor:
    """Declarative metadata about a source connector."""

    name: str
    description: str = ""
    source_url: str = ""
    reliability: str = "medium"
    update_frequency: str = "static"
    requires_download: bool = False
    default_rate_limit: float = 0.0
    tags: list[str] = field(default_factory=list)


@dataclass
class FetchResult:
    """Result of a fetch operation."""

    file_path: str
    file_hash: str = ""
    file_size: int = 0
    fetched_at: str = ""
    source_version: str = ""

    def __post_init__(self) -> None:
        if not self.fetched_at:
            self.fetched_at = datetime.now(UTC).isoformat()


@runtime_checkable
class BaseSource(Protocol):
    """Protocol for acquisition source connectors.

    Each connector discovers, fetches, normalizes, and checkpoints
    data from a specific public source.  Connectors operate on local
    files and return RawImportRecords for the existing import pipeline.

    Implementations must NOT:
    - Scrape websites that prohibit automated access
    - Fabricate any data
    - Know about DatasetStore internals
    """

    @property
    def descriptor(self) -> SourceDescriptor:
        """Declarative metadata about this source."""
        ...

    def discover(self, config: dict[str, object] | None = None) -> list[str]:
        """Discover available data files or endpoints for this source.

        Returns a list of file paths or URLs that can be fetched.
        """
        ...

    def fetch(
        self,
        target: str,
        dest_dir: str,
        *,
        config: dict[str, object] | None = None,
    ) -> FetchResult:
        """Fetch a specific data file to a local destination.

        Parameters
        ----------
        target :
            File path or URL returned by discover().
        dest_dir :
            Local directory to write the fetched file to.
        config :
            Optional configuration (API keys, rate limits, etc.).

        Returns
        -------
        FetchResult with the local file path and metadata.
        """
        ...

    def normalize(self, file_path: str) -> list[RawImportRecord]:
        """Normalize a fetched file into RawImportRecords.

        This delegates to the existing import adapter for parsing.
        """
        ...

    def checkpoint(self, file_path: str) -> str:
        """Compute a deterministic checkpoint key for a file.

        Default implementation returns the file's content hash.
        """
        ...
