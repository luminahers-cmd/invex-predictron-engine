"""Phase V1 — Ground-Truth Dataset Schema (design only, not populated).

This module defines the *shape* of the longitudinal outcome dataset used
to evaluate whether the Predictron Engine makes good investment decisions
on real startups.  It contains only data models, enums, and constants.

Design intent
-------------
* **Design-only.** No population logic, no loader, no metrics, no engine
  coupling.  This file *documents the contract* that downstream
  evaluation tooling (V1B metrics, V1C pipeline) will consume.
* **Longitudinal.** A startup is analysed at a point in time
  (``analysis_timestamp``, ``funding_stage_at_analysis``) and the dataset
  records ``actual_outcome`` observed later, plus a sequence of verified
  outcome events.  Nothing here fabricates outcomes; every field is
  sourced from the ground-truth sources catalogued in V1D.
* **Temporal integrity.** A "prediction" and its "ground truth" must be
  unambiguous about the valuation/evaluation horizon:
  ``outcome_observation_date`` is when the outcome state was last verified,
  and ``evaluation_horizon_days`` is how many days past the analysis the
  evaluation window was closed.

A future V1C implementation will persist records conforming to this
schema (JSON lines, one record per analysed startup) and evaluate engine
predictions against them.  No such records exist yet.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# Versioned so the on-disk format can evolve while remaining auditable.
GROUND_TRUTH_SCHEMA_VERSION = 1

# Canonical identifier for this dataset family.
DATASET_NAME = "predictron_ground_truth"


class StartupStatus(str, Enum):
    """Actual future outcome of the startup, observed after analysis."""

    OPERATING = "operating"  # still active, no terminal event
    ACQUIRED = "acquired"
    SHUTDOWN = "shutdown"
    UNKNOWN = "unknown"  # status not verifiable from available sources


class FundingStage(str, Enum):
    """Funding stage at or after the analysis timestamp.

    Mirrors the stages the engine already reasons about so predictions
    and ground truth speak the same vocabulary.
    """

    IDEA = "idea"
    PRE_SEED = "pre_seed"
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b"
    SERIES_C = "series_c"
    GROWTH = "growth"


class OutcomeEventKind(str, Enum):
    """The type of a verified outcome event."""

    FUNDING = "funding"
    FOLLOW_ON_ROUND = "follow_on_round"
    ACQUISITION = "acquisition"
    SHUTDOWN = "shutdown"
    ARR_MILESTONE = "arr_milestone"
    EMPLOYEE_GROWTH = "employee_growth"
    INVESTOR_PARTICIPATION = "investor_participation"


class OutcomeEvent(BaseModel):
    """A single verified, timestamped outcome event.

    Every event must be attributable to at least one source so the
    dataset remains auditable and free of fabricated labels.
    """

    kind: OutcomeEventKind
    occurred_at: date = Field(..., description="Date the event was verified")
    value_currency_usd: float | None = Field(
        default=None,
        description=(
            "Nominal amount in USD for funding/ARR events "
            "(e.g. round amount, ARR). None when not applicable."
        ),
    )
    amount_nominal_units: float | None = Field(
        default=None,
        description="Scalar for non-currency events, e.g. headcount or ARR in USD/y.",
    )
    description: str = Field(default="", description="Human-readable event description")
    sources: list[str] = Field(
        default_factory=list,
        description="Source identifiers from the V1D source catalogue",
    )


class EngineSnapshot(BaseModel):
    """The engine prediction being evaluated (captured at analysis time).

    Stores only what is needed to score a prediction against outcome:
    the decision, its confidence, and the composite score.  Full engine
    outputs (scores per dimension, observations, evidence) are optional
    and stored separately for deeper analysis; they do not gate metrics.
    """

    investment_decision: str | None = Field(
        default=None,
        description="Decision category value (e.g. strong_invest/invest/pass)",
    )
    conviction: str | None = Field(
        default=None, description="Conviction level value (e.g. high/low)"
    )
    composite_score: float | None = Field(
        default=None, ge=0.0, le=100.0, description="Composite investment score"
    )
    overall_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Overall confidence in [0,1]"
    )
    decision_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Calibrated decision confidence"
    )
    engine_version: str = Field(
        default="", description="Engine version that produced the prediction"
    )
    engine_output_path: str | None = Field(
        default=None,
        description="Optional path to the full serialized Report for deep analysis",
    )


class GroundTruthRecord(BaseModel):
    """One startup's analysis-time prediction paired with its later outcome.

    This is the canonical longitudinal record.  A dataset is a sequence
    of these records.  The schema is forward-compatible: unknown fields
    are preserved by the loader but not required for scoring.
    """

    # --- Startup identity ---------------------------------------------------
    startup_id: str = Field(..., description="Stable unique identifier for the startup")
    startup_name: str = Field(..., description="Display name")
    startup_website: str | None = Field(default=None)
    industry_category: str | None = Field(default=None)
    geography: str | None = Field(default=None)

    # --- Analysis-time context ---------------------------------------------
    analysis_timestamp: datetime = Field(
        ..., description="UTC timestamp when the engine analysed the startup"
    )
    funding_stage_at_analysis: FundingStage = Field(
        ..., description="Funding stage used as analysis-time input"
    )

    # --- Engine prediction --------------------------------------------------
    engine_snapshot: EngineSnapshot = Field(
        ..., description="The engine prediction being evaluated"
    )

    # --- Actual future outcome ---------------------------------------------
    actual_status: StartupStatus = Field(
        ..., description="Observed terminal/current status after the horizon"
    )
    outcome_observation_date: date = Field(
        ..., description="Date the outcome state was last verified"
    )
    evaluation_horizon_days: int = Field(
        ge=0,
        description="Days from analysis to outcome-observation (horizon closed)",
    )
    outcome_events: list[OutcomeEvent] = Field(
        default_factory=list,
        description="Verified funding/ARR/acquisition/shutdown/growth events",
    )
    exit_value_usd: float | None = Field(
        default=None,
        description="Reported acquisition/exit value, if any",
    )

    # --- Dataset provenance -------------------------------------------------
    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this record was created in the dataset",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Ground-truth data sources backing this record (V1D)",
    )
    schema_version: int = Field(
        default=GROUND_TRUTH_SCHEMA_VERSION,
        description="Ground-truth dataset schema version",
    )


# Outcome labels derived from ``actual_status`` for metric scoring.  A
# startup is a "success" when it was acquired at a positive value or is
# still operating with an ARR milestone at/after the horizon; otherwise
# it is a "failure".  This mapping is deliberately small and explicit:
# the V1B metrics consume it and it must be agreed upon before scoring.
SUCCESS_STATUSES: frozenset[str] = frozenset({StartupStatus.ACQUIRED.value})
PENDING_SUCCESS_STATUSES: frozenset[str] = frozenset({StartupStatus.OPERATING.value})


def derive_binary_outcome(record: GroundTruthRecord) -> Literal[0, 1] | None:
    """Return a binary success label for a record, or None when undefined.

    ``None`` encodes "insufficient ground truth to label" (e.g. status is
    ``UNKNOWN`` or is ``OPERATING`` with no ARR milestone and no positive
    exit).  Metrics must treat ``None`` as un-scoreable and exclude it,
    never as a fabricated outcome.

    This pure function is provided because the acquisition status alone is
    an insufficiently rich signal: a shutdown with a positive acquisition
    history, or an operating company that merely survived, must be handled
    deliberately rather than auto-labelled here.
    """
    status = record.actual_status
    if status == StartupStatus.ACQUIRED:
        return 1 if (record.exit_value_usd or 0) > 0 else 0
    if status == StartupStatus.SHUTDOWN:
        return 0
    if status == StartupStatus.OPERATING:
        has_arr_milestone = any(
            e.kind == OutcomeEventKind.ARR_MILESTONE
            and (e.amount_nominal_units or 0) > 0
            for e in record.outcome_events
        )
        return 1 if has_arr_milestone else None
    return None
