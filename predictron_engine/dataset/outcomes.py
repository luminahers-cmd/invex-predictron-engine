"""Outcome tracking models (Part B).

Designs support for future startup outcomes.  All outcome fields
allow UNKNOWN and must never be fabricated.  Outcomes are linked to
DatasetRecords through record_id and are populated only through
verified imports or manual verification.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class OutcomeStatus(str, Enum):
    """Current verification status of an outcome record."""

    UNKNOWN = "unknown"
    PARTIALLY_VERIFIED = "partially_verified"
    FULLY_VERIFIED = "fully_verified"
    OUTDATED = "outdated"


class OutcomeVerdict(str, Enum):
    """Categorical outcome label for evaluation.

    Derived from observed outcomes, not predictions.
    """

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    INCONCLUSIVE = "inconclusive"
    UNKNOWN = "unknown"


class FundingEvent(BaseModel):
    """A single funding round or financial event."""

    round_type: str = Field(
        ..., description="Round type (seed, series_a, etc.)"
    )
    amount_usd: float | None = Field(
        default=None, ge=0.0, description="Funding amount in USD"
    )
    date: datetime | None = Field(
        default=None, description="Date of the funding round"
    )
    investors: list[str] = Field(
        default_factory=list, description="Participating investors"
    )
    valuation_usd: float | None = Field(
        default=None, ge=0.0, description="Post-money valuation if known"
    )
    source: str = Field(
        default="unknown", description="Source of this funding data"
    )


class StartupOutcome(BaseModel):
    """Observable outcome data for a startup.

    All fields allow None or UNKNOWN.  No field is ever fabricated.
    Updated over time as new information becomes available.
    """

    funding_rounds: list[FundingEvent] = Field(
        default_factory=list,
        description="Chronological list of funding rounds",
    )
    total_funding_usd: float | None = Field(
        default=None, ge=0.0, description="Total funding raised to date"
    )
    acquisition: str | None = Field(
        default=None,
        description="Acquirer name if acquired, None if not acquired",
    )
    acquisition_price_usd: float | None = Field(
        default=None, ge=0.0, description="Acquisition price if known"
    )
    shutdown: bool | None = Field(
        default=None, description="True if the startup has shut down"
    )
    shutdown_date: datetime | None = Field(
        default=None, description="Date of shutdown if applicable"
    )
    bankruptcy: bool | None = Field(
        default=None, description="True if bankruptcy was filed"
    )
    bankruptcy_date: datetime | None = Field(
        default=None, description="Date of bankruptcy filing if applicable"
    )
    arr_milestones: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "ARR snapshots: [{date, arr_usd, source}]"
        ),
    )
    employee_count_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Employee count snapshots: [{date, count, source}]"
        ),
    )
    valuation_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Valuation snapshots: [{date, valuation_usd, source}]"
        ),
    )
    investors: list[str] = Field(
        default_factory=list,
        description="Known investors across all rounds",
    )
    exit_date: datetime | None = Field(
        default=None,
        description="Date of exit (IPO or acquisition) if applicable",
    )
    exit_type: str | None = Field(
        default=None,
        description="Exit type: 'ipo', 'acquisition', 'shutdown', None",
    )
    latest_verification_date: datetime | None = Field(
        default=None,
        description="When this outcome was last verified or updated",
    )
    status: OutcomeStatus = Field(
        default=OutcomeStatus.UNKNOWN,
        description="Verification status of this outcome record",
    )

    def is_terminal(self) -> bool:
        """Return True if the outcome represents a terminal state."""
        if self.acquisition is not None:
            return True
        if self.shutdown is True:
            return True
        if self.bankruptcy is True:
            return True
        if self.exit_type in ("ipo", "acquisition", "shutdown"):
            return True
        return False


class OutcomeRecord(BaseModel):
    """Links a DatasetRecord to its observed outcome.

    This is the bridge between prediction and reality.  The record_id
    links to a DatasetRecord; the outcome contains what actually
    happened.  The verdict is derived from the outcome, never
    predicted.
    """

    outcome_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique outcome record identifier (UUIDv4)",
    )
    record_id: str = Field(
        ..., description="Dataset record this outcome belongs to"
    )
    outcome: StartupOutcome = Field(
        default_factory=StartupOutcome,
        description="Observed outcome data",
    )
    verdict: OutcomeVerdict = Field(
        default=OutcomeVerdict.UNKNOWN,
        description=(
            "Categorical verdict derived from observed outcomes"
        ),
    )
    verdict_reasoning: str = Field(
        default="",
        description="Explanation of how the verdict was determined",
    )
    time_horizon_days: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Days between analysis_date and outcome observation"
        ),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this outcome record was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this outcome was last updated",
    )
    notes: str = Field(
        default="", description="Free-text notes about this outcome"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata about the outcome",
    )

    def derive_verdict(self) -> OutcomeVerdict:
        """Derive a verdict from the outcome without prediction context.

        Returns UNKNOWN if insufficient data is available.
        """
        o = self.outcome
        if o.status == OutcomeStatus.UNKNOWN:
            return OutcomeVerdict.UNKNOWN
        if o.shutdown is True or o.bankruptcy is True:
            return OutcomeVerdict.FAILURE
        if o.acquisition is not None:
            return OutcomeVerdict.SUCCESS
        if o.exit_type == "ipo":
            return OutcomeVerdict.SUCCESS
        if o.exit_type == "acquisition":
            return OutcomeVerdict.SUCCESS
        if o.total_funding_usd is not None and o.total_funding_usd > 0:
            return OutcomeVerdict.PARTIAL_SUCCESS
        return OutcomeVerdict.UNKNOWN
