"""Shared fixtures for dataset pipeline tests."""

from __future__ import annotations

import csv
from pathlib import Path

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
from predictron_engine.dataset.population_config import (
    PopulationConfig,
    SourceConfig,
)
from predictron_engine.dataset.store import DatasetStore


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


def make_csv(tmp_path: Path, name: str, rows: list[tuple[str, str]]) -> Path:
    """Write a CSV file with the given (Name, Website) rows."""
    p = tmp_path / name
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Name", "Website"])
        writer.writeheader()
        for company, website in rows:
            writer.writerow({"Name": company, "Website": website})
    return p


@pytest.fixture()
def dataset_store(tmp_path: Path) -> DatasetStore:
    store = DatasetStore(tmp_path / "dataset")
    store.initialize()
    return store


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    """A CSV with a few companies."""
    p = tmp_path / "startups.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Name", "Website"])
        writer.writeheader()
        writer.writerow({"Name": "Acme Corp", "Website": "https://acme.com"})
        writer.writerow({"Name": "Beta Inc", "Website": "https://beta.com"})
        writer.writerow({"Name": "Gamma Ltd", "Website": "https://gamma.com"})
    return p


@pytest.fixture()
def csv_export_config(csv_file: Path) -> PopulationConfig:
    return PopulationConfig(
        sources=[SourceConfig(name="csv_export", file_paths=[str(csv_file)])]
    )
