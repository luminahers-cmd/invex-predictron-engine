"""Forecasting groundwork for the prediction validation loop (Phase 4).

Bridges a frozen :class:`TimeScopedPrediction` to the *evaluation side* of
the pipeline:

* :class:`ForecastHorizon` — the temporal contract (window length plus a
  deterministic tolerance) that links an analysis to its due outcome;
* :class:`PredictionForecast` — a prediction awaited against that horizon;
* :func:`dataset_record_from_prediction` — maps a frozen prediction onto the
  engine's :class:`DatasetRecord` shape so the existing evaluation machinery
  (:meth:`PredictionEvaluation.from_records`) can consume frozen cohorts.

Pure, deterministic, and additive: nothing here fabricates outcomes or
touches storage.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from predictron_engine.dataset.models import DatasetRecord, PredictionSummary
from predictron_engine.dataset.outcomes import OutcomeRecord
from predictron_engine.dataset.prediction import TimeScopedPrediction


class ForecastHorizon(BaseModel):
    """Temporal contract between an analysis and its due outcome."""

    days: int = Field(..., gt=0, description="Evaluation horizon length in days")
    tolerance_days: int = Field(
        default=1,
        ge=0,
        description="Grace applied when deciding whether a forecast is due",
    )

    def due_at(self, analysis_timestamp: datetime) -> datetime:
        """Earliest moment a matching outcome observation is due."""
        if analysis_timestamp.tzinfo is None:
            return analysis_timestamp.replace(tzinfo=UTC) + timedelta(
                days=self.days
            )
        return analysis_timestamp + timedelta(days=self.days)


class PredictionForecast(BaseModel):
    """A frozen prediction awaited against an evaluation horizon."""

    forecast_id: str = Field(default_factory=lambda: str(uuid4()))
    company_id: str = Field(..., description="Stable company identifier")
    company_name: str = Field(..., description="Company name as submitted")
    analysis_timestamp: datetime = Field(
        ..., description="UTC timestamp the prediction is pinned to"
    )
    horizon: ForecastHorizon = Field(..., description="Evaluation horizon contract")
    prediction: PredictionSummary = Field(
        ..., description="Frozen prediction summary (immutable)"
    )
    outcome_record: OutcomeRecord | None = Field(
        default=None, description="Observed outcome, once recorded"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def due_at(self) -> datetime:
        """Earliest moment the outcome is due for this forecast."""
        return self.horizon.due_at(self.analysis_timestamp)

    @property
    def is_satisfied(self) -> bool:
        """True once an outcome observation has been attached."""
        return self.outcome_record is not None

    def is_due(self, *, as_of: datetime | None = None) -> bool:
        """True when the evaluation window has (or is within tolerance of) elapsed."""
        as_of_utc = as_of or datetime.now(UTC)
        earliest = self.due_at - timedelta(days=self.horizon.tolerance_days)
        return as_of_utc >= earliest

    def days_until_due(self, *, as_of: datetime | None = None) -> float:
        """Days remaining until the forecast is due (negative when overdue)."""
        as_of_utc = as_of or datetime.now(UTC)
        return (self.due_at - as_of_utc).total_seconds() / 86400.0

    def with_outcome(self, outcome_record: OutcomeRecord) -> PredictionForecast:
        """Return a copy with the observed outcome attached (immutable)."""
        return self.model_copy(update={"outcome_record": outcome_record})


def forecast_from_prediction(
    prediction: TimeScopedPrediction,
    *,
    horizon_days: int | None = None,
) -> PredictionForecast:
    """Wrap a frozen time-scoped prediction as a forecast.

    ``horizon_days`` overrides the prediction's own
    ``evaluation_horizon_days``; at least one of the two must resolve.
    """
    days = horizon_days if horizon_days is not None else prediction.evaluation_horizon_days
    if days is None:
        raise ValueError(
            "a forecast horizon is required: set evaluation_horizon_days on the "
            "prediction or pass horizon_days"
        )
    return PredictionForecast(
        company_id=prediction.company_id,
        company_name=prediction.company_name,
        analysis_timestamp=prediction.analysis_timestamp,
        horizon=ForecastHorizon(days=days),
        prediction=prediction.prediction,
        metadata={
            "prediction_id": prediction.prediction_id,
            "engine_version": prediction.engine_version,
            "evidence_reference": prediction.evidence_reference,
        },
    )


def dataset_record_from_prediction(
    prediction: TimeScopedPrediction,
    *,
    website: str | None = None,
) -> DatasetRecord:
    """Map a frozen prediction onto the engine's DatasetRecord shape.

    ``record_id`` is the company id so evaluations derived from frozen
    cohorts stay deterministic and company-anchored, mirroring the cohort
    build step (:mod:`benchmarks.cohort.build`).
    """
    analysis_metadata: dict[str, Any] = dict(prediction.metadata)
    if prediction.scope_stats is not None:
        analysis_metadata["prediction_scope"] = prediction.scope_stats.model_dump(
            mode="json"
        )
    if prediction.evaluation_horizon_days is not None:
        analysis_metadata["evaluation_horizon_days"] = prediction.evaluation_horizon_days
    return DatasetRecord(
        record_id=prediction.company_id,
        startup_name=prediction.company_name,
        website=website or "",
        analysis_date=prediction.analysis_timestamp,
        engine_version=prediction.engine_version,
        evidence_bundle_reference=prediction.evidence_reference,
        prediction=prediction.prediction,
        analysis_metadata=analysis_metadata,
        source="frozen_prediction",
    )


__all__ = [
    "ForecastHorizon",
    "PredictionForecast",
    "dataset_record_from_prediction",
    "forecast_from_prediction",
]
