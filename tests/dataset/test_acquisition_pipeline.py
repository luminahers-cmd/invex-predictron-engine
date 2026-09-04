"""Tests for the acquisition pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from predictron_engine.dataset.acquisition.pipeline import (
    AcquisitionMetrics,
    AcquisitionPipeline,
    AcquisitionResult,
    RetryPolicy,
)
from predictron_engine.dataset.acquisition.sources.csv_export_connector import (
    CsvExportConnector,
)
from predictron_engine.dataset.store import DatasetStore


@pytest.fixture()
def store(tmp_path: Path) -> DatasetStore:
    ds = DatasetStore(tmp_path / "dataset")
    ds.initialize()
    return ds


@pytest.fixture()
def csv_source(tmp_path: Path) -> Path:
    import csv

    p = tmp_path / "startups.csv"
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Name", "Website"])
        writer.writeheader()
        writer.writerow({"Name": "Acme Corp", "Website": "https://acme.com"})
        writer.writerow({"Name": "Beta Inc", "Website": "https://beta.com"})
    return p


class TestAcquisitionMetrics:
    def test_to_dict(self) -> None:
        m = AcquisitionMetrics(imported=10, failed=2)
        d = m.to_dict()
        assert d["imported"] == 10
        assert d["failed"] == 2
        assert d["elapsed_seconds"] == 0.0

    def test_defaults(self) -> None:
        m = AcquisitionMetrics()
        assert m.total_files == 0
        assert m.imported == 0
        assert m.errors == []


class TestAcquisitionResult:
    def test_to_dict(self) -> None:
        r = AcquisitionResult(
            metrics=AcquisitionMetrics(imported=5),
            source_name="test",
        )
        d = r.to_dict()
        assert d["source_name"] == "test"
        assert d["imported"] == 5
        assert d["dry_run"] is False


class TestRetryPolicy:
    def test_defaults(self) -> None:
        p = RetryPolicy()
        assert p.max_retries == 3
        assert p.backoff_base == 1.0
        assert OSError in p.retry_on


class TestAcquisitionPipeline:
    def test_dry_run(self, store: DatasetStore, csv_source: Path) -> None:
        pipeline = AcquisitionPipeline(store, dry_run=True)
        conn = CsvExportConnector()
        result = pipeline.run_source(conn, str(csv_source), idempotent=False)

        assert result.dry_run is True
        assert result.metrics.total_raw_records == 2
        assert result.metrics.imported == 0
        assert store.count_records() == 0

    def test_import(self, store: DatasetStore, csv_source: Path) -> None:
        pipeline = AcquisitionPipeline(store, batch_size=10)
        conn = CsvExportConnector()
        result = pipeline.run_source(conn, str(csv_source), idempotent=False)

        assert result.dry_run is False
        assert result.metrics.imported == 2
        assert result.metrics.failed == 0
        assert store.count_records() == 2

    def test_idempotent_skips(self, store: DatasetStore, csv_source: Path) -> None:
        pipeline = AcquisitionPipeline(store)
        conn = CsvExportConnector()

        result1 = pipeline.run_source(conn, str(csv_source), idempotent=True)
        assert result1.metrics.imported == 2

        result2 = pipeline.run_source(conn, str(csv_source), idempotent=True)
        assert result2.metrics.skipped_idempotent == 1
        assert result2.metrics.imported == 0

    def test_limit(self, store: DatasetStore, csv_source: Path) -> None:
        pipeline = AcquisitionPipeline(store, limit=1)
        conn = CsvExportConnector()
        result = pipeline.run_source(conn, str(csv_source), idempotent=False)

        assert result.metrics.imported == 1
        assert store.count_records() == 1

    def test_batch_multiple_files(
        self, store: DatasetStore, tmp_path: Path
    ) -> None:
        import csv

        files: list[Path] = []
        for i in range(3):
            p = tmp_path / f"batch_{i}.csv"
            with p.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["Name", "Website"])
                writer.writeheader()
                writer.writerow({
                    "Name": f"Company {i}",
                    "Website": f"https://co{i}.com",
                })
            files.append(p)

        pipeline = AcquisitionPipeline(store)
        conn = CsvExportConnector()
        result = pipeline.run_source_batch(
            conn, [str(f) for f in files], idempotent=False
        )

        assert result.metrics.imported == 3
        assert result.metrics.total_files == 3
        assert store.count_records() == 3
