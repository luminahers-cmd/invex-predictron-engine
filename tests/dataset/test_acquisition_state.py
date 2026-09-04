"""Tests for acquisition state management and checkpoint recovery."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from predictron_engine.dataset.acquisition.state import (
    AcquisitionBatch,
    AcquisitionRecord,
    AcquisitionStateManager,
    CheckpointData,
    compute_file_hash,
    generate_batch_id,
)


@pytest.fixture()
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def state_mgr(tmp_root: Path) -> AcquisitionStateManager:
    mgr = AcquisitionStateManager(tmp_root)
    mgr.initialize()
    return mgr


class TestCheckpointData:
    def test_default_fields(self) -> None:
        cp = CheckpointData()
        assert cp.batch_id == ""
        assert cp.processed_records == 0
        assert cp.is_complete is False

    def test_advance(self) -> None:
        cp = CheckpointData(total_records=100)
        cp.advance(imported=50, failed=5, skipped=3)
        assert cp.processed_records == 58
        assert cp.imported_records == 50
        assert cp.failed_records == 5
        assert cp.skipped_records == 3

    def test_is_complete(self) -> None:
        cp = CheckpointData(total_records=100)
        cp.advance(imported=100)
        assert cp.is_complete is True

    def test_is_complete_empty(self) -> None:
        cp = CheckpointData(total_records=0)
        assert cp.is_complete is False


class TestAcquisitionBatch:
    def test_auto_batch_id(self) -> None:
        batch = AcquisitionBatch()
        assert len(batch.batch_id) == 36  # UUID format
        assert batch.batch_id != ""
        assert len(batch.batch_id) > 10

    def test_custom_batch_id(self) -> None:
        batch = AcquisitionBatch(batch_id="custom-id")
        assert batch.batch_id == "custom-id"

    def test_records_default_empty(self) -> None:
        batch = AcquisitionBatch()
        assert batch.records == []


class TestAcquisitionStateManager:
    def test_initialize_creates_dirs(self, tmp_root: Path) -> None:
        mgr = AcquisitionStateManager(tmp_root)
        mgr.initialize()
        assert (tmp_root / "acquisition" / "history").is_dir()
        assert (tmp_root / "acquisition" / "checkpoints").is_dir()

    def test_record_and_list_batches(self, state_mgr: AcquisitionStateManager) -> None:
        batch = AcquisitionBatch(
            batch_id="test-001",
            source_name="yc_oss",
            file_path="/data/yc.csv",
            file_hash="abc123",
            record_count=100,
        )
        state_mgr.record_batch(batch)
        batches = state_mgr.list_batches()
        assert len(batches) == 1
        assert batches[0].source_name == "yc_oss"
        assert batches[0].record_count == 100

    def test_get_last_batch_for_source(self, state_mgr: AcquisitionStateManager) -> None:
        batch1 = AcquisitionBatch(
            batch_id="b1", source_name="yc_oss", record_count=50
        )
        batch2 = AcquisitionBatch(
            batch_id="b2", source_name="yc_oss", record_count=75
        )
        state_mgr.record_batch(batch1)
        state_mgr.record_batch(batch2)

        last = state_mgr.get_last_batch_for_source("yc_oss")
        assert last is not None
        assert last.batch_id == "b2"

    def test_get_last_batch_missing(self, state_mgr: AcquisitionStateManager) -> None:
        assert state_mgr.get_last_batch_for_source("nonexistent") is None

    def test_has_file_been_imported(self, state_mgr: AcquisitionStateManager) -> None:
        assert state_mgr.has_file_been_imported("hash123") is False
        batch = AcquisitionBatch(
            batch_id="b1", source_name="yc_oss", file_hash="hash123"
        )
        state_mgr.record_batch(batch)
        assert state_mgr.has_file_been_imported("hash123") is True

    def test_save_and_load_checkpoint(self, state_mgr: AcquisitionStateManager) -> None:
        cp = CheckpointData(
            batch_id="cp-001",
            source_name="yc_oss",
            total_records=1000,
            processed_records=500,
        )
        state_mgr.save_checkpoint(cp)
        loaded = state_mgr.load_checkpoint("cp-001")
        assert loaded is not None
        assert loaded.batch_id == "cp-001"
        assert loaded.processed_records == 500

    def test_load_missing_checkpoint(self, state_mgr: AcquisitionStateManager) -> None:
        assert state_mgr.load_checkpoint("nonexistent") is None

    def test_list_checkpoints(self, state_mgr: AcquisitionStateManager) -> None:
        cp1 = CheckpointData(batch_id="cp1", total_records=100, processed_records=50)
        cp2 = CheckpointData(batch_id="cp2", total_records=100, processed_records=100)
        state_mgr.save_checkpoint(cp1)
        state_mgr.save_checkpoint(cp2)

        incomplete = state_mgr.list_checkpoints()
        assert len(incomplete) == 1
        assert incomplete[0].batch_id == "cp1"

    def test_clear_checkpoint(self, state_mgr: AcquisitionStateManager) -> None:
        cp = CheckpointData(batch_id="cp-clear")
        state_mgr.save_checkpoint(cp)
        assert state_mgr.load_checkpoint("cp-clear") is not None
        state_mgr.clear_checkpoint("cp-clear")
        assert state_mgr.load_checkpoint("cp-clear") is None

    def test_find_resume_checkpoint(self, state_mgr: AcquisitionStateManager) -> None:
        cp = CheckpointData(
            batch_id="resume-001",
            source_name="yc_oss",
            file_hash="hashabc",
            total_records=100,
            processed_records=30,
        )
        state_mgr.save_checkpoint(cp)
        found = state_mgr.find_resume_checkpoint("yc_oss", "hashabc")
        assert found is not None
        assert found.batch_id == "resume-001"


class TestHelpers:
    def test_compute_file_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h = compute_file_hash(f)
        assert len(h) == 64
        assert isinstance(h, str)

    def test_compute_file_hash_deterministic(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        assert compute_file_hash(f) == compute_file_hash(f)

    def test_generate_batch_id(self) -> None:
        id1 = generate_batch_id()
        id2 = generate_batch_id()
        assert id1 != id2
        assert id1.startswith("batch-")
