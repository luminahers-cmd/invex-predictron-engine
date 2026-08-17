"""Normalizer — converts API requests into internal Startup models.

The normalizer is the first stage of the pipeline. It receives a validated
StartupAnalysisRequest (already validated by FastAPI/Pydantic at the route
layer) and transforms it into the internal Startup representation.

Responsibilities:
  - Strip and normalize string fields
  - Convert typed URL objects to plain strings
  - Preserve raw data for downstream access
  - Attach normalization timestamp

This stage performs no enrichment, feature extraction, or judgment.
"""

import logging
from datetime import UTC, datetime

from app.schemas.analysis import StartupAnalysisRequest
from predictron_engine.models.startup import Startup

logger = logging.getLogger(__name__)


class DefaultNormalizer:
    """Standard implementation of the Normalizer protocol."""

    def normalize(self, request: StartupAnalysisRequest) -> Startup:
        """Convert an API request into an internal Startup model."""
        logger.info("Normalizing startup: %s", request.startup_name)

        return Startup(
            name=request.startup_name.strip(),
            website=str(request.website) if request.website else "",
            description=request.description.strip(),
            pitch_deck_url=(
                str(request.pitch_deck_url) if request.pitch_deck_url else None
            ),
            founder_linkedin_urls=[str(u) for u in request.founder_linkedin_urls],
            raw_data=request.model_dump(mode="json"),
            normalized_at=datetime.now(UTC),
        )
