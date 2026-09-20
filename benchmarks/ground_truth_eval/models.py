"""Golden dataset data contracts for the Ground Truth Evaluation Platform.

A *golden dataset* is a curated, versioned collection of benchmark
companies.  Each company bundles the inputs the engine was run against
(``request`` + ``historical_evidence``), the prediction the engine
produced at analysis time (``historical_prediction``), the later
verified outcome (``verified_outcome``), and the provenance of every
label.  Nothing in this module invents outcomes: an entry whose outcome
cannot be verified stays un-scoreable (``binary_outcome()`` returns
``None``) and is excluded from every binary metric, never coerced.

The model reuses the canned contract enums/derivations from ``...ground_truth.schema``
(``StartupStatus``, ``OutcomeEvent``, ``derive_binary_outcome``) so the whole
repository speaks one outcome vocabulary.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from benchmarks.ground_truth.schema import (
    GROUND_TRUTH_SCHEMA_VERSION,
    EngineSnapshot,
    FundingStage,
    GroundTruthRecord,
    OutcomeEvent,
    StartupStatus,
    derive_binary_outcome,
)

# Version of the GoldenDataset on-disk format itself.
GOLDEN_DATASET_SCHEMA_VERSION = 1

# Decision values the engine can emit.  Kept as plain strings so the model
# stays decoupled from the report enums (which live in the engine layer).
POSITIVE_DECISIONS: frozenset[str] = frozenset({"strong_invest", "invest"})
NEGATIVE_DECISIONS: frozenset[str] = frozenset({"pass"})
NEUTRAL_DECISIONS: frozenset[str] = frozenset({"watch", "investigate_further"})


class HistoricalPrediction(BaseModel):
    """The prediction the engine produced at analysis time.

    This is a frozen historical artifact: the runner must never write into
    it.  ``overall_score`` and ``overall_confidence`` are the canonical
    scalar predictions; ``decision`` is the decision-category value.
    """

    overall_score: float = Field(..., ge=0.0, le=100.0)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    decision: str = Field(..., description="Decision category value")
    composite_score: float | None = Field(default=None, ge=0.0, le=100.0)
    recommendation_categories: list[str] = Field(default_factory=list)
    dimension_scores: dict[str, float] = Field(default_factory=dict)

    def predicted_positive(self) -> bool | None:
        """Binary classifiable view of the historical decision.

        Returns ``None`` for neutral decisions (watch / investigate
        further), which cannot be coerced into a positive/negative call.
        """
        return predicted_positive(self.decision)


class HistoricalEvidence(BaseModel):
    """Reference to the evidence the analysis was based on.

    ``corpus_name`` names a committed offline evidence corpus resolved via
    :func:`predictron_engine.evidence.replay.dataset.resolve_dataset_path`.
    Supplying a corpus lets the runner replay the analysis deterministically
    and network-free.  ``sources`` records the provenance of the corpus.
    """

    corpus_name: str | None = Field(default=None)
    corpus_path: str | None = Field(default=None)
    sources: list[str] = Field(default_factory=list)

    @property
    def reference(self) -> str | None:
        return self.corpus_name or self.corpus_path or None


class VerifiedOutcome(BaseModel):
    """The later, verified outcome for a benchmark company.

    ``status`` and the optional exit value / milestone events are the raw
    verified facts.  ``binary_outcome()`` derives a 0/1 success label using
    the same rules as the engine-side ground-truth contract
    (:func:`~benchmarks.ground_truth.schema.derive_binary_outcome`) and
    returns ``None`` when the outcome is genuinely un-scoreable.
    """

    status: StartupStatus = Field(..., description="Verified startup status")
    verification_date: date = Field(..., description="Date the outcome was last verified")
    outcome_events: list[OutcomeEvent] = Field(default_factory=list)
    exit_value_usd: float | None = Field(default=None)
    sources: list[str] = Field(
        default_factory=list,
        description="Ground-truth sources backing this outcome",
    )

    def binary_outcome(self) -> Literal[0, 1] | None:
        """Binary 0/1 success label, or ``None`` when un-scoreable."""
        record = GroundTruthRecord(
            startup_id="",
            startup_name="",
            analysis_timestamp=datetime(1970, 1, 1, tzinfo=UTC),
            funding_stage_at_analysis=FundingStage.SEED,
            engine_snapshot=EngineSnapshot(engine_version="historical"),
            actual_status=self.status,
            outcome_observation_date=self.verification_date,
            evaluation_horizon_days=0,
            outcome_events=self.outcome_events,
            exit_value_usd=self.exit_value_usd,
        )
        return derive_binary_outcome(record)

    @property
    def is_scoreable(self) -> bool:
        return self.binary_outcome() is not None


class Provenance(BaseModel):
    """Provenance of one golden entry (labels, evidence, outcome)."""

    sources: list[str] = Field(default_factory=list)
    collector: str | None = Field(default=None)
    verified_by: str | None = Field(default=None)
    is_example: bool = Field(default=False)
    notes: str = Field(default="")


class GoldenEntry(BaseModel):
    """One benchmark company in a golden dataset.

    Contains every element required by the platform spec: company id,
    historical prediction, historical evidence, verified outcome,
    verification date, and provenance.  ``request`` carries the same fields
    as :class:`app.schemas.analysis.StartupAnalysisRequest` so the runner
    can rebuild a request 1:1.
    """

    company_id: str = Field(..., description="Stable unique company identifier")
    company_name: str = Field(..., description="Display name")
    request: dict[str, Any] = Field(
        ...,
        description="StartupAnalysisRequest-compatible request fields",
    )
    sector: str | None = Field(default=None, description="Sector/category label")
    country: str | None = Field(default=None, description="Country/population label")
    stage: str | None = Field(default=None, description="Funding stage label")
    historical_prediction: HistoricalPrediction = Field(
        ..., description="Frozen prediction produced at analysis time"
    )
    historical_evidence: HistoricalEvidence = Field(
        ..., description="Evidence corpus the analysis was based on"
    )
    verified_outcome: VerifiedOutcome = Field(
        ..., description="Later verified outcome for the company"
    )
    provenance: Provenance = Field(default_factory=Provenance)
    analysis_timestamp: datetime | None = Field(
        default=None,
        description=(
            "UTC timestamp the evidence scope is pinned to.  When set, the "
            "runner enforces a look-ahead guard: corpus evidence fetched after "
            "this instant is blocked from the engine run."
        ),
    )
    evaluation_horizon_days: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Minimum days allowed between analysis and outcome observation; "
            "used to validate temporal integrity of the entry."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def verification_date(self) -> date:
        return self.verified_outcome.verification_date

    @property
    def is_time_pinned(self) -> bool:
        """True when the entry carries an analysis-time anchor."""
        return self.analysis_timestamp is not None


class GoldenDataset(BaseModel):
    """A curated, versioned benchmark dataset.

    ``benchmark_version`` is the version string of this dataset (e.g.
    ``"1.0"``, ``"1.1"``, ``"2026-01"``).  ``dataset_hash`` is computed over
    the canonical serialization and used to fingerprint runs.
    """

    dataset_name: str
    benchmark_version: str
    schema_version: int = Field(default=GOLDEN_DATASET_SCHEMA_VERSION)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    entries: list[GoldenEntry] = Field(default_factory=list)

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    def entry_by_id(self, company_id: str) -> GoldenEntry | None:
        for entry in self.entries:
            if entry.company_id == company_id:
                return entry
        return None

    def scoreable_entry_count(self) -> int:
        return sum(1 for e in self.entries if e.verified_outcome.is_scoreable)


def predicted_positive(decision: str) -> bool | None:
    """Return the binary classifiable view of a decision value.

    ``True`` for strong_invest/invest, ``False`` for pass, ``None`` for
    neutral decisions that cannot be classified.
    """
    if decision in POSITIVE_DECISIONS:
        return True
    if decision in NEGATIVE_DECISIONS:
        return False
    return None


__all__ = [
    "GOLDEN_DATASET_SCHEMA_VERSION",
    "POSITIVE_DECISIONS",
    "NEGATIVE_DECISIONS",
    "NEUTRAL_DECISIONS",
    "HistoricalPrediction",
    "HistoricalEvidence",
    "VerifiedOutcome",
    "Provenance",
    "GoldenEntry",
    "GoldenDataset",
    "predicted_positive",
    "GROUND_TRUTH_SCHEMA_VERSION",
]
