"""Tests for the Phase 4 forecasting groundwork (predictron_engine.dataset.forecast).

Covers the temporal horizon contract, forecast lifecycle helpers, and the
DatasetRecord bridge used to feed frozen predictions into the evaluation
machinery.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from predictron_engine.dataset.forecast import (
    ForecastHorizon,
    dataset_record_from_prediction,
    forecast_from_prediction,
)
from predictron_engine.dataset.models import DecisionLabel, PredictionSummary
from predictron_engine.dataset.prediction import TimeScopedPrediction

ANALYSIS = datetime(2026, 1, 1, tzinfo=UTC)


def _prediction() -> TimeScopedPrediction:
    return TimeScopedPrediction(
        prediction_id="pred-1",
        company_id="cmp-1",
        company_name="Example",
        analysis_timestamp=ANALYSIS,
        evaluation_horizon_days=180,
        engine_version="9.9.9",
        evidence_reference="b2b_saas",
        prediction=PredictionSummary(
            decision=DecisionLabel.INVEST,
            confidence=0.75,
            composite_score=70.0,
            dimension_scores={"market": 80.0},
            investment_readiness_score=60.0,
            recommendation_count=3,
        ),
    )


class TestForecastHorizon:
    def test_due_at_adds_horizon(self) -> None:
        horizon = ForecastHorizon(days=30)
        assert horizon.due_at(ANALYSIS) == datetime(2026, 1, 31, tzinfo=UTC)

    def test_due_at_normalizes_naive(self) -> None:
        horizon = ForecastHorizon(days=30)
        naive = datetime(2026, 1, 1)
        assert horizon.due_at(naive) == datetime(2026, 1, 31, tzinfo=UTC)

    def test_rejects_zero_days(self) -> None:
        with pytest.raises(ValueError):
            ForecastHorizon(days=0)


class TestPredictionForecast:
    def test_forecast_from_prediction_uses_own_horizon(self) -> None:
        forecast = forecast_from_prediction(_prediction())
        assert forecast.horizon.days == 180
        assert forecast.due_at == datetime(2026, 6, 30, tzinfo=UTC)
        assert forecast.is_satisfied is False

    def test_forecast_from_prediction_override_horizon(self) -> None:
        forecast = forecast_from_prediction(_prediction(), horizon_days=90)
        assert forecast.horizon.days == 90
        assert forecast.metadata["prediction_id"] == "pred-1"

    def test_forecast_requires_horizon(self) -> None:
        prediction = _prediction().model_copy(update={"evaluation_horizon_days": None})
        with pytest.raises(ValueError, match="horizon"):
            forecast_from_prediction(prediction)

    def test_is_due_window_and_tolerance(self) -> None:
        forecast = forecast_from_prediction(_prediction())
        due = datetime(2026, 7, 1, tzinfo=UTC)
        assert forecast.is_due(as_of=due) is True
        # Within tolerance (1 day grace) it is still considered due.
        assert forecast.is_due(as_of=datetime(2026, 6, 29, tzinfo=UTC)) is True
        assert forecast.is_due(as_of=datetime(2026, 6, 1, tzinfo=UTC)) is False

    def test_days_until_due(self) -> None:
        forecast = forecast_from_prediction(_prediction())
        assert forecast.days_until_due(as_of=ANALYSIS) == pytest.approx(180.0)

    def test_with_outcome_attaches_observation(self) -> None:
        from predictron_engine.dataset.outcomes import (
            OutcomeRecord,
        )

        forecast = forecast_from_prediction(_prediction())
        updated = forecast.with_outcome(OutcomeRecord(record_id="cmp-1"))
        assert updated.is_satisfied is True
        assert forecast.is_satisfied is False  # original stays immutable


class TestDatasetRecordBridge:
    def test_maps_prediction_fields(self) -> None:
        record = dataset_record_from_prediction(_prediction())
        assert record.record_id == "cmp-1"
        assert record.startup_name == "Example"
        assert record.source == "frozen_prediction"
        assert record.prediction.decision == DecisionLabel.INVEST
        assert record.evidence_bundle_reference == "b2b_saas"

    def test_metadata_carries_scope_and_horizon(self) -> None:
        record = dataset_record_from_prediction(_prediction())
        assert record.analysis_metadata["evaluation_horizon_days"] == 180

    def test_website_defaults_to_empty(self) -> None:
        record = dataset_record_from_prediction(_prediction())
        assert record.website == ""

    def test_website_override(self) -> None:
        record = dataset_record_from_prediction(
            _prediction(), website="https://example.com"
        )
        assert record.website == "https://example.com"
