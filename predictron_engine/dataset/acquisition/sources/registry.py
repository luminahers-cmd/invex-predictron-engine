"""Acquisition source connector registry.

Central registry for all available source connectors.
"""

from __future__ import annotations

from predictron_engine.dataset.acquisition.sources.base import BaseSource
from predictron_engine.dataset.acquisition.sources.companies_house_connector import (
    CompaniesHouseConnector,
)
from predictron_engine.dataset.acquisition.sources.csv_export_connector import (
    CsvExportConnector,
)
from predictron_engine.dataset.acquisition.sources.sec_edgar_connector import (
    SecEdgarConnector,
)
from predictron_engine.dataset.acquisition.sources.yc_oss_connector import (
    YcOssConnector,
)


class SourceConnectorRegistry:
    """Registry of available acquisition source connectors."""

    def __init__(self) -> None:
        self._connectors: dict[str, BaseSource] = {}

    def register(self, connector: BaseSource) -> None:
        """Register a source connector."""
        self._connectors[connector.descriptor.name] = connector

    def get(self, name: str) -> BaseSource | None:
        """Retrieve a registered connector by name."""
        return self._connectors.get(name)

    def list_sources(self) -> list[str]:
        """List all registered source names."""
        return list(self._connectors.keys())

    def list_descriptors(self) -> list[dict[str, object]]:
        """List descriptors for all registered sources."""
        result: list[dict[str, object]] = []
        for conn in self._connectors.values():
            d = conn.descriptor
            result.append({
                "name": d.name,
                "description": d.description,
                "reliability": d.reliability,
                "update_frequency": d.update_frequency,
                "requires_download": d.requires_download,
                "tags": d.tags,
            })
        return result

    @classmethod
    def default(cls) -> SourceConnectorRegistry:
        """Create a registry with all built-in connectors."""
        registry = cls()
        registry.register(SecEdgarConnector())
        registry.register(CompaniesHouseConnector())
        registry.register(YcOssConnector())
        registry.register(CsvExportConnector())
        return registry
