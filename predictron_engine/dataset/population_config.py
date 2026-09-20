"""Population source configuration (Project V4).

Defines how sources are configured for the dataset population
orchestrator.  A ``PopulationConfig`` describes an ordered set of
sources, each with a connector name and the files/directories it
should acquire from.

Configuration is additive: it does not alter the acquisition
framework.  It simply tells :class:`PopulationOrchestrator` which
sources to iterate over and in what order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SourceConfig:
    """Configuration for a single population source.

    Attributes
    ----------
    name :
        Connector name (e.g. ``yc_oss``, ``csv_export``,
        ``sec_edgar``, ``companies_house``).
    file_paths :
        Explicit file paths to acquire from.
    directories :
        Directories to scan for matching source files.
    enable_resume :
        Whether this source may resume incomplete acquisitions.
    """

    name: str
    file_paths: list[str] = field(default_factory=list)
    directories: list[str] = field(default_factory=list)
    enable_resume: bool = True

    def to_dict(self) -> dict[str, object]:
        """Serializable dict view."""
        return {
            "name": self.name,
            "file_paths": list(self.file_paths),
            "directories": list(self.directories),
            "enable_resume": self.enable_resume,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SourceConfig:
        """Construct from a dict (as produced by :meth:`to_dict`)."""
        raw_paths = data.get("file_paths", [])
        raw_dirs = data.get("directories", [])
        file_paths: list[str] = []
        directories: list[str] = []
        if isinstance(raw_paths, list):
            file_paths = [str(p) for p in raw_paths if isinstance(p, str)]
        if isinstance(raw_dirs, list):
            directories = [str(d) for d in raw_dirs if isinstance(d, str)]
        return cls(
            name=str(data.get("name", "")),
            file_paths=file_paths,
            directories=directories,
            enable_resume=bool(data.get("enable_resume", True)),
        )


@dataclass
class PopulationConfig:
    """Ordered configuration of sources for a population run.

    Attributes
    ----------
    sources :
        Ordered list of source configurations to iterate over.
    idempotent :
        Whether to skip sources/files that have already been imported.
    batch_size :
        Records per batch passed to the acquisition pipeline.
    checkpoint_every :
        Persist a checkpoint every N batches.
    """

    sources: list[SourceConfig] = field(default_factory=list)
    idempotent: bool = True
    batch_size: int = 1000
    checkpoint_every: int = 1

    def source_names(self) -> list[str]:
        """Return the ordered source names."""
        return [s.name for s in self.sources]

    def get(self, name: str) -> SourceConfig | None:
        """Retrieve a source configuration by name, or None."""
        for src in self.sources:
            if src.name == name:
                return src
        return None

    def to_dict(self) -> dict[str, object]:
        """Serializable dict view."""
        return {
            "sources": [s.to_dict() for s in self.sources],
            "idempotent": self.idempotent,
            "batch_size": self.batch_size,
            "checkpoint_every": self.checkpoint_every,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> PopulationConfig:
        """Construct from a dict (as produced by :meth:`to_dict`)."""
        raw_sources = data.get("sources", [])
        sources: list[SourceConfig] = []
        if isinstance(raw_sources, list):
            for d in raw_sources:
                if isinstance(d, dict):
                    sources.append(SourceConfig.from_dict(d))
        batch_size = data.get("batch_size", 1000)
        checkpoint_every = data.get("checkpoint_every", 1)
        return cls(
            sources=sources,
            idempotent=bool(data.get("idempotent", True)),
            batch_size=int(batch_size) if isinstance(batch_size, int) else 1000,
            checkpoint_every=(
                int(checkpoint_every)
                if isinstance(checkpoint_every, int)
                else 1
            ),
        )


def load_config(path: str | Path) -> PopulationConfig:
    """Load a population configuration from a JSON file.

    The file is expected to contain a mapping as produced by
    :meth:`PopulationConfig.to_dict`.
    """
    file_path = Path(path)
    data: dict[str, Any] = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"population config '{path}' must contain a JSON object"
        raise ValueError(msg)
    return PopulationConfig.from_dict(data)


def build_default_config() -> PopulationConfig:
    """Build a default configuration from the registered connectors.

    Every registered acquisition connector is included as a source that
    can be populated from configured directories.  File paths are left
    empty so operators can supply them via CLI flags or a config file.
    """
    from predictron_engine.dataset.acquisition.sources import (
        SourceConnectorRegistry,
    )

    registry = SourceConnectorRegistry.default()
    sources = [
        SourceConfig(name=name)
        for name in registry.list_sources()
    ]
    return PopulationConfig(sources=sources)
