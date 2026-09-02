"""Historical dataset record schema (Part A).

Stores immutable snapshots of engine predictions at analysis time.
No outcome data is fabricated — all outcome fields default to UNKNOWN
and are populated only through verified imports or manual verification.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class FundingStage(str, Enum):
    """Canonical funding stage labels aligned with knowledge layer."""

    PRE_SEED = "pre_seed"
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b"
    SERIES_C = "series_c"
    SERIES_D = "series_d"
    SERIES_E = "series_e"
    GROWTH = "growth"
    IPO = "ipo"
    UNKNOWN = "unknown"


class DecisionLabel(str, Enum):
    """Engine decision labels at analysis time."""

    STRONG_INVEST = "strong_invest"
    INVEST = "invest"
    WATCH = "watch"
    INVESTIGATE_FURTHER = "investigate_further"
    PASS = "pass"


class PredictionSummary(BaseModel):
    """Compact prediction snapshot captured at analysis time.

    Stores the engine's output fields that will later be compared
    against actual outcomes.  Every field is immutable once recorded.
    """

    decision: DecisionLabel = Field(
        ..., description="Engine investment decision at analysis time"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence (0-1)"
    )
    composite_score: float = Field(
        ..., ge=0.0, le=100.0, description="Composite venture score (0-100)"
    )
    dimension_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-dimension scores (0-100)",
    )
    investment_readiness_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Investment readiness score if available",
    )
    recommendation_count: int = Field(
        default=0, ge=0, description="Number of recommendations generated"
    )


class DatasetRecord(BaseModel):
    """Immutable historical record of a single engine prediction.

    Captures the full context of an analysis run without modifying
    any engine internals.  The record is the atomic unit of the
    dataset — one record per analysis, linked to a future outcome
    through the OutcomeRecord.
    """

    record_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique dataset record identifier (UUIDv4)",
    )
    startup_name: str = Field(..., description="Startup name as submitted")
    website: str = Field(..., description="Startup website URL")
    analysis_date: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of the analysis",
    )
    engine_version: str = Field(
        ..., description="Predictron Engine version used"
    )
    benchmark_version: str | None = Field(
        default=None,
        description="Benchmark version used, if any",
    )
    evidence_bundle_reference: str | None = Field(
        default=None,
        description=(
            "Reference to the evidence bundle used "
            "(file path or identifier)"
        ),
    )
    prediction: PredictionSummary = Field(
        ..., description="Engine prediction snapshot at analysis time"
    )
    funding_stage_at_analysis: FundingStage = Field(
        default=FundingStage.UNKNOWN,
        description="Funding stage known at analysis time",
    )
    analysis_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary metadata preserved from the analysis run "
            "(engine metadata, processing time, etc.)"
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description="User-defined tags for filtering and grouping",
    )
    source: str = Field(
        default="direct",
        description=(
            "Origin of this record: 'direct' for live analyses, "
            "'import' for imported data, 'manual' for hand-entered"
        ),
    )

    def prediction_hash_fields(self) -> tuple[str, float, float]:
        """Return the fields used for prediction identity verification."""
        return (
            self.prediction.decision.value,
            self.prediction.confidence,
            self.prediction.composite_score,
        )
