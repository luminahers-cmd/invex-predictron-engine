"""Data Collector — gathers and enriches startup data.

The collector sits as the second stage in the pipeline, between the
normalizer and the feature extractor. Its role is to take a clean
Startup object and prepare a richer CollectedData object that the
extractor can analyze.

Responsibilities:
  - Parse and decompose structured fields (URLs, text)
  - Compute basic metadata (word counts, domain extraction)
  - Gather enrichment signals from available sources
  - Preserve information lineage for downstream auditability

The collector does NOT extract domain features (that is the extractor's
job). It prepares the data surface that extraction operates on.

Extensibility:
  - Inject external data sources via the constructor
  - Add new collection strategies as independent components
  - The collector protocol accepts any implementation that returns CollectedData
"""

import logging
from urllib.parse import urlparse

from predictron_engine.models.collected_data import CollectedData
from predictron_engine.models.startup import Startup

logger = logging.getLogger(__name__)


class DataSource:
    """Protocol-compatible interface for external data sources.

    Future implementations might query Crunchbase, PitchBook, or
    internal databases. The collector delegates enrichment to these
    sources when they are available.
    """

    def fetch(self, startup: Startup) -> dict[str, str | int | float | bool]:
        """Fetch enrichment data for a startup. Returns empty dict by default."""
        return {}


class DefaultDataCollector:
    """Standard implementation of the DataCollector protocol.

    Performs baseline collection using only the data present in the
    Startup object. External data sources can be injected via the
    constructor for enrichment.
    """

    def __init__(self, sources: list[DataSource] | None = None) -> None:
        self._sources = sources or []

    def collect(self, startup: Startup) -> CollectedData:
        """Collect and enrich data from the normalized startup."""
        logger.info("Collecting data for: %s", startup.name)

        domain = self._extract_domain(startup.website)
        tokens = self._tokenize_description(startup.description)
        enrichment = self._gather_enrichment(startup)

        return CollectedData(
            startup_name=startup.name,
            website_domain=domain,
            description_tokens=tokens,
            description_word_count=len(tokens),
            has_website=startup.website is not None,
            has_pitch_deck=startup.pitch_deck_url is not None,
            founder_count=len(startup.founder_linkedin_urls),
            url_metadata=self._collect_url_metadata(startup),
            enrichment_signals=enrichment,
        )

    def _extract_domain(self, url: str) -> str | None:
        """Extract the root domain from a URL string."""
        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            parts = host.split(".")
            if len(parts) >= 2:
                return ".".join(parts[-2:])
            return host or None
        except Exception:
            return None

    def _tokenize_description(self, description: str) -> list[str]:
        """Split description text into lowercase word tokens."""
        return [w.lower() for w in description.split() if len(w) > 2]

    def _collect_url_metadata(self, startup: Startup) -> dict[str, str]:
        """Extract metadata from provided URLs."""
        metadata: dict[str, str] = {}
        if startup.website:
            metadata["website_url"] = startup.website
        if startup.pitch_deck_url:
            metadata["pitch_deck_url"] = startup.pitch_deck_url
        for i, url in enumerate(startup.founder_linkedin_urls):
            metadata[f"founder_{i}_linkedin"] = url
        return metadata

    def _gather_enrichment(
        self, startup: Startup
    ) -> dict[str, str | int | float | bool]:
        """Gather enrichment signals from injected data sources."""
        signals: dict[str, str | int | float | bool] = {}
        for source in self._sources:
            try:
                signals.update(source.fetch(startup))
            except Exception:
                logger.warning(
                    "Data source %s failed, continuing",
                    type(source).__name__,
                )
        return signals
