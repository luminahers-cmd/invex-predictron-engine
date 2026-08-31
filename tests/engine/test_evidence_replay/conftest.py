"""Shared fixtures for the evidence replay regression suite."""

from __future__ import annotations

import pytest

from predictron_engine.evidence.replay.dataset import dataset_root

COMMITTED_DATASET = dataset_root() / "sample_evidence.json"


@pytest.fixture
def committed_dataset() -> str:
    """Path (as a name string) to the committed sample evidence corpus."""
    assert COMMITTED_DATASET.exists(), f"committed dataset missing: {COMMITTED_DATASET}"
    return str(COMMITTED_DATASET)
