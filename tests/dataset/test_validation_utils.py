"""Tests for field-level validation utilities (Part B)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from predictron_engine.dataset.models import PredictionSummary
from predictron_engine.dataset.outcomes import OutcomeRecord, StartupOutcome
from predictron_engine.dataset.validation_utils import (
    validate_outcome_fields,
    validate_record_completeness,
    validate_record_fields,
)
from tests.dataset.conftest import make_outcome, make_record

FIXED_NOW = datetime(2026, 6, 1, tzinfo=UTC)
PAST_DATE = datetime(2024, 6, 1, tzinfo=UTC)


def _record_with_past_date():
    return make_record().model_copy(
        update={"analysis_date": PAST_DATE}
    )


def test_valid_record_passes() -> None:
    record = _record_with_past_date()
    issues = validate_record_fields(record, now=FIXED_NOW)
    assert issues == []


def test_missing_startup_name() -> None:
    record = make_record().model_copy(update={"startup_name": "   "})
    kinds = {i.kind for i in validate_record_fields(record, now=FIXED_NOW)}
    assert "missing_startup_name" in kinds


def test_missing_website() -> None:
    record = make_record().model_copy(update={"website": ""})
    kinds = {i.kind for i in validate_record_fields(record, now=FIXED_NOW)}
    assert "missing_website" in kinds


def test_invalid_website_scheme() -> None:
    record = make_record().model_copy(update={"website": "example.com"})
    kinds = {i.kind for i in validate_record_fields(record, now=FIXED_NOW)}
    assert "invalid_website" in kinds


def test_future_analysis_date() -> None:
    record = make_record().model_copy(
        update={"analysis_date": FIXED_NOW + timedelta(days=30)}
    )
    kinds = {i.kind for i in validate_record_fields(record, now=FIXED_NOW)}
    assert "future_analysis_date" in kinds


def test_implausible_analysis_date() -> None:
    record = make_record().model_copy(
        update={"analysis_date": datetime(1995, 1, 1, tzinfo=UTC)}
    )
    kinds = {i.kind for i in validate_record_fields(record, now=FIXED_NOW)}
    assert "implausible_analysis_date" in kinds


def test_invalid_prediction_values() -> None:
    pred = PredictionSummary.model_construct(
        decision="invest",
        confidence=1.5,
        composite_score=120.0,
        dimension_scores={"market": 150.0},
    )
    record = _record_with_past_date().model_copy(update={"prediction": pred})
    kinds = {i.kind for i in validate_record_fields(record, now=FIXED_NOW)}
    assert "invalid_confidence" in kinds
    assert "invalid_composite_score" in kinds
    assert "invalid_dimension_score" in kinds


def test_outcome_future_verification() -> None:
    outcome = make_outcome("r1")
    outcome.outcome.latest_verification_date = FIXED_NOW + timedelta(days=10)
    kinds = {i.kind for i in validate_outcome_fields(outcome, now=FIXED_NOW)}
    assert "future_verification_date" in kinds


def test_outcome_bankruptcy_before_shutdown() -> None:
    outcome = OutcomeRecord(
        record_id="r1",
        outcome=StartupOutcome(
            shutdown=True,
            shutdown_date=datetime(2024, 5, 1, tzinfo=UTC),
            bankruptcy=True,
            bankruptcy_date=datetime(2024, 4, 1, tzinfo=UTC),
        ),
    )
    kinds = {i.kind for i in validate_outcome_fields(outcome, now=FIXED_NOW)}
    assert "invalid_chronology" in kinds


def test_outcome_negative_funding() -> None:
    outcome = make_outcome("r1")
    outcome.outcome.total_funding_usd = -5.0
    kinds = {i.kind for i in validate_outcome_fields(outcome, now=FIXED_NOW)}
    assert "negative_total_funding" in kinds


def test_outcome_missing_record_reference() -> None:
    outcome = make_outcome("r1").model_copy(update={"record_id": ""})
    kinds = {i.kind for i in validate_outcome_fields(outcome, now=FIXED_NOW)}
    assert "missing_record_reference" in kinds


def test_completeness_report() -> None:
    record = make_record()
    missing, present = validate_record_completeness(record)
    assert "startup_name" in present
    assert "website" in present
    # make_record has no investment_readiness_score -> missing
    assert "prediction.investment_readiness_score" in missing
