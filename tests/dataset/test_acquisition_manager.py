"""Tests for the acquisition manager."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from predictron_engine.dataset.acquisition.manager import (
    AcquireOptions,
    AcquisitionManager,
)
from predictron_engine.dataset.store import DatasetStore


@pytest.fixture()
def store(tmp_path: Path) -> DatasetStore:
    ds = DatasetStore(tmp_path / "dataset")
    ds.initialize()
    return ds


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "startups.csv"
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Name", "Website"])
        writer.writeheader()
        writer.writerow({"Name": "Acme", "Website": "https://acme.com"})
    return p


class TestAcquireOptions:
    def test_defaults(self) -> None:
        opts = AcquireOptions()
        assert opts.source_name is None
        assert opts.dry_run is False
        assert opts.limit is None
        assert opts.batch_size == 1000


class TestAcquisitionManager:
    def test_list_sources(self, store: DatasetStore) -> None:
        mgr = AcquisitionManager(store)
        sources = mgr.list_sources()
        names = {s["name"] for s in sources}
        assert "sec_edgar" in names
        assert "yc_oss" in names
        assert "csv_export" in names

    def test_acquire_dry_run(self, store: DatasetStore, csv_file: Path) -> None:
        mgr = AcquisitionManager(store)
        result = mgr.acquire(AcquireOptions(
            source_name="csv_export",
            file_path=str(csv_file),
            dry_run=True,
        ))
        assert result.dry_run is True
        assert result.metrics.total_raw_records == 1
        assert store.count_records() == 0

    def test_acquire_import(self, store: DatasetStore, csv_file: Path) -> None:
        mgr = AcquisitionManager(store)
        result = mgr.acquire(AcquireOptions(
            source_name="csv_export",
            file_path=str(csv_file),
        ))
        assert result.metrics.imported == 1
        assert store.count_records() == 1

    def test_acquire_idempotent(self, store: DatasetStore, csv_file: Path) -> None:
        mgr = AcquisitionManager(store)
        opts = AcquireOptions(
            source_name="csv_export",
            file_path=str(csv_file),
        )
        r1 = mgr.acquire(opts)
        assert r1.metrics.imported == 1

        r2 = mgr.acquire(opts)
        assert r2.metrics.skipped_idempotent == 1
        assert store.count_records() == 1

    def test_acquire_source_convenience(
        self, store: DatasetStore, csv_file: Path
    ) -> None:
        mgr = AcquisitionManager(store)
        result = mgr.acquire_source(
            "csv_export",
            {"file_path": str(csv_file)},
        )
        assert result.metrics.imported == 1

    def test_acquire_no_source_found(self, store: DatasetStore) -> None:
        mgr = AcquisitionManager(store)
        result = mgr.acquire(AcquireOptions())
        assert len(result.metrics.errors) > 0

    def test_acquire_no_files_found(
        self, store: DatasetStore, tmp_path: Path
    ) -> None:
        mgr = AcquisitionManager(store)
        result = mgr.acquire(AcquireOptions(
            source_name="csv_export",
            file_path=str(tmp_path / "nonexistent.csv"),
        ))
        assert len(result.metrics.errors) > 0

    def test_status(self, store: DatasetStore, csv_file: Path) -> None:
        mgr = AcquisitionManager(store)
        mgr.acquire(AcquireOptions(
            source_name="csv_export",
            file_path=str(csv_file),
        ))
        status = mgr.status()
        assert status["record_count"] == 1
        assert status["outcome_count"] == 1
        assert status["total_batches"] == 1
