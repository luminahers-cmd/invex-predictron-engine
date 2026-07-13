"""Data collected and enriched by the collection module."""

from pydantic import BaseModel, Field


class CollectedData(BaseModel):
    """Enriched data gathered during the collection phase.

    The collector sits between normalization and extraction. It transforms
    a clean Startup object into a richer dataset that the extractor can
    analyze. This may include aggregated metadata, external signals,
    or computed derivatives of the raw input.

    All fields are optional because collection is best-effort — the
    pipeline must remain functional even when enrichment sources are
    unavailable.
    """

    startup_name: str = Field(
        ..., description="Startup name carried from normalization"
    )
    website_domain: str | None = Field(
        default=None, description="Extracted root domain"
    )
    description_tokens: list[str] = Field(
        default_factory=list, description="Tokenized description words"
    )
    description_word_count: int = Field(
        default=0, description="Word count of description"
    )
    has_website: bool = Field(default=True, description="Whether a website was provided")
    has_pitch_deck: bool = Field(
        default=False, description="Whether a pitch deck URL exists"
    )
    founder_count: int = Field(
        default=0, description="Number of founder profiles provided"
    )
    url_metadata: dict[str, str] = Field(
        default_factory=dict, description="Metadata derived from URLs"
    )
    enrichment_signals: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="External enrichment signals when available",
    )
