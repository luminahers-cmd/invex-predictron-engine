"""Tests for the acquisition CLI commands."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from predictron_engine.dataset.cli import build_parser, main


@pytest.fixture()
def dataset_dir(tmp_path: Path) -> Path:
    d = tmp_path / "dataset"
    d.mkdir()
    return d


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "startups.csv"
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Name", "Website"])
        writer.writeheader()
        writer.writerow({"Name": "Acme", "Website": "https://acme.com"})
    return p


class TestAcquireCLI:
    def test_acquire_parses(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "acquire", "--dataset", "/tmp/ds",
            "--source", "csv_export",
        ])
        assert args.command == "acquire"
        assert args.source == "csv_export"

    def test_acquire_dry_run(
        self, dataset_dir: Path, csv_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main([
            "acquire",
            "--dataset", str(dataset_dir),
            "--source", "csv_export",
            "--file", str(csv_file),
            "--dry-run",
        ])
        assert rc == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["dry_run"] is True
        assert data["total_raw_records"] == 1

    def test_acquire_import(
        self, dataset_dir: Path, csv_file: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main([
            "acquire",
            "--dataset", str(dataset_dir),
            "--source", "csv_export",
            "--file", str(csv_file),
        ])
        assert rc == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["imported"] == 1

    def test_acquire_no_source(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main([
            "acquire",
            "--dataset", str(dataset_dir),
        ])
        assert rc == 1


class TestAcquireStatusCLI:
    def test_status(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(["acquire-status", "--dataset", str(dataset_dir)])
        assert rc == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "record_count" in data


class TestAcquireSourcesCLI:
    def test_sources(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["acquire-sources"])
        assert rc == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        names = {s["name"] for s in data}
        assert "sec_edgar" in names
        assert "yc_oss" in names


class TestAcquireScheduleCLI:
    def test_schedule_list(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main([
            "acquire-schedule", "--dataset", str(dataset_dir), "list"
        ])
        assert rc == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert isinstance(data, list)

    def test_schedule_add(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main([
            "acquire-schedule",
            "--dataset", str(dataset_dir),
            "add",
            "--schedule-source", "yc_oss",
            "--frequency", "daily",
        ])
        assert rc == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["added"] == "yc_oss"

    def test_schedule_remove(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main([
            "acquire-schedule",
            "--dataset", str(dataset_dir),
            "add",
            "--schedule-source", "yc_oss",
        ])
        capsys.readouterr()
        rc = main([
            "acquire-schedule",
            "--dataset", str(dataset_dir),
            "remove",
            "--schedule-source", "yc_oss",
        ])
        assert rc == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["removed"] == "yc_oss"
