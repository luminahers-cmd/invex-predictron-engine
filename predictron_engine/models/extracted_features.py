"""Structured factual attributes extracted from startup data."""

from pydantic import BaseModel, Field


class ExtractedFeatures(BaseModel):
    """Pure factual attributes derived from startup data.

    This model carries only objective, verifiable facts. No scoring,
    no subjective judgments — just structured data that downstream
    reasoning and scoring modules consume.

    Every field is nullable or defaulted. Extraction is best-effort:
    if the extractor cannot determine an attribute from available data,
    the field remains None. Downstream modules must handle missing
    features gracefully.
    """

    industry: str | None = Field(
        default=None, description="Primary industry classification"
    )
    sub_industry: str | None = Field(
        default=None, description="Secondary industry classification"
    )
    business_model: str | None = Field(
        default=None, description="Business model type (e.g. SaaS, marketplace)"
    )
    funding_stage: str | None = Field(
        default=None, description="Current or most recent funding stage"
    )
    geography: str | None = Field(
        default=None, description="Primary geographic market"
    )
    headquarters_region: str | None = Field(
        default=None, description="HQ region if determinable"
    )
    technology_stack: list[str] = Field(
        default_factory=list, description="Identified technologies in use"
    )
    customer_type: str | None = Field(
        default=None, description="Target customer segment (B2B, B2C, B2B2C)"
    )
    team_size_indicator: str | None = Field(
        default=None, description="Approximate team size range"
    )
    founded_year: int | None = Field(
        default=None, description="Year the company was founded"
    )
    has_revenue: bool | None = Field(
        default=None, description="Whether revenue generation is evident"
    )
    key_keywords: list[str] = Field(
        default_factory=list, description="Key terms extracted from description"
    )
    description_length: int = Field(
        default=0, description="Character length of description"
    )
    has_pitch_deck: bool = Field(
        default=False, description="Whether a pitch deck was provided"
    )
    founder_profile_count: int = Field(
        default=0, description="Number of founder profiles"
    )
    data_completeness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of feature fields that are non-null",
    )

    # --- Market Intelligence fields (Sprint 1) ---

    target_market: str | None = Field(
        default=None,
        description="Primary target market description",
    )
    market_maturity: str | None = Field(
        default=None,
        description="Market maturity stage (emerging, growth, mature, saturated)",
    )
    market_keywords: list[str] = Field(
        default_factory=list,
        description="Domain-specific market terms extracted from description",
    )
    market_signals: list[str] = Field(
        default_factory=list,
        description="Detected market signals",
    )
    market_characteristics: list[str] = Field(
        default_factory=list,
        description="Detected market characteristics",
    )
    enterprise_orientation: str | None = Field(
        default=None,
        description="Enterprise vs consumer orientation (enterprise, consumer, hybrid)",
    )
    industry_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the industry classification (0.0-1.0)",
    )
    customer_segment: str | None = Field(
        default=None,
        description="Specific customer segment",
    )
