"""Public dataset source adapters (Part C).

Independent adapters for parsing publicly available startup datasets.
Each adapter implements the ImportSource protocol and yields
RawImportRecords without scraping, downloading, or fabricating any data.

Supported sources:
  - SEC EDGAR: Startup company filings
  - YC OSS: Y Combinator open-source dataset
  - Government registries: Company registration records

All adapters are stateless and handle their specific source format.
They do not make network requests — they parse local files.
"""

from predictron_engine.dataset.sources.gov_registries import GovRegistrySource
from predictron_engine.dataset.sources.sec_edgar import SecEdgarSource
from predictron_engine.dataset.sources.yc_oss import YcOssSource

__all__ = ["GovRegistrySource", "SecEdgarSource", "YcOssSource"]
