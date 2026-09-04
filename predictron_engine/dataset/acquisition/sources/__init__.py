"""Acquisition source connectors.

Independent connectors for acquiring data from public sources.
Each connector implements the BaseSource protocol and exposes
discover(), fetch(), normalize(), and checkpoint() methods.
"""

from predictron_engine.dataset.acquisition.sources.base import (
    BaseSource,
    FetchResult,
    SourceDescriptor,
)
from predictron_engine.dataset.acquisition.sources.companies_house_connector import (
    CompaniesHouseConnector,
)
from predictron_engine.dataset.acquisition.sources.csv_export_connector import (
    CsvExportConnector,
)
from predictron_engine.dataset.acquisition.sources.registry import (
    SourceConnectorRegistry,
)
from predictron_engine.dataset.acquisition.sources.sec_edgar_connector import (
    SecEdgarConnector,
)
from predictron_engine.dataset.acquisition.sources.yc_oss_connector import (
    YcOssConnector,
)

__all__ = [
    "BaseSource",
    "CompaniesHouseConnector",
    "CsvExportConnector",
    "FetchResult",
    "SecEdgarConnector",
    "SourceConnectorRegistry",
    "SourceDescriptor",
    "YcOssConnector",
]
