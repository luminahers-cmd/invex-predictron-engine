"""Shared fixtures for dataset pipeline tests."""

from __future__ import annotations

import pytest

from predictron_engine.dataset.models import (
    DatasetRecord,
    DecisionLabel,
    PredictionSummary,
)
from predictron_engine.dataset.outcomes import (
    OutcomeRecord,
    OutcomeStatus,
    StartupOutcome,
)


def make_record(
    startup_name: str = "Acme Corp",
    website: str = "https://acme.example.com",
    decision: DecisionLabel = DecisionLabel.INVEST,
    confidence: float = 0.75,
    composite_score: float = 68.5,
    record_id: str | None = None,
) -> DatasetRecord:
    """Create a DatasetRecord for testing."""
    kwargs: dict[str, object] = {
        "startup_name": startup_name,
        "website": website,
        "engine_version": "0.12.1",
        "prediction": PredictionSummary(
            decision=decision,
            confidence=confidence,
            composite_score=composite_score,
            dimension_scores={"market": 70.0, "team": 75.0},
        ),
    }
    if record_id is not None:
        kwargs["record_id"] = record_id
    return DatasetRecord(**kwargs)


def make_outcome(
    record_id: str,
    shutdown: bool | None = None,
    acquisition: str | None = None,
    status: OutcomeStatus = OutcomeStatus.FULLY_VERIFIED,
) -> OutcomeRecord:
    """Create an OutcomeRecord for testing."""
    return OutcomeRecord(
        record_id=record_id,
        outcome=StartupOutcome(
            shutdown=shutdown,
            acquisition=acquisition,
            status=status,
        ),
    )


@pytest.fixture
def dataset_record_factory():
    return make_record


@pytest.fixture
def outcome_record_factory():
    return make_outcome
