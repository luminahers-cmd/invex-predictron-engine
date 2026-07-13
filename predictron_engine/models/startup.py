"""Internal canonical representation of a startup."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class Startup(BaseModel):
    """Canonical startup representation after normalization.

    This is the internal data object that flows through the entire pipeline.
    It carries both the normalized structured fields and any raw data that
    arrived in the original request for downstream access.
    """

    name: str = Field(..., min_length=1, description="Startup company name")
    website: str = Field(..., description="Normalized website URL string")
    description: str = Field(..., min_length=1, description="Startup description text")
    pitch_deck_url: str | None = Field(
        default=None, description="Pitch deck URL if available"
    )
    founder_linkedin_urls: list[str] = Field(
        default_factory=list, description="Founder LinkedIn profile URLs"
    )
    raw_data: dict[str, Any] = Field(
        default_factory=dict, description="Preserved original request data"
    )
    normalized_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of normalization",
    )
