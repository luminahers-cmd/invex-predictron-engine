"""Tests for the populate CLI command (Project V4)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from predictron_engine.dataset.cli import main
from predictron_engine.dataset.population_config import (
    PopulationConfig,
    SourceConfig,
)


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "startups.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Name", "Website"])
        writer.writeheader()
        writer.writerow({"Name": "Acme", "Website": "https://acme.com"})
        writer.writerow({"Name": "Beta", "Website": "https://beta.com"})
    return p


@pytest.fixture()
def config_file(tmp_path: Path, csv_file: Path) -> Path:
    cfg = PopulationConfig(
        sources=[SourceConfig(name="csv_export", file_paths=[str(csv_file)])]
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    return path


class TestPopulateCLI:
    def test_populate_parses(self) -> None:
        from predictron_engine.dataset.cli import build_parser

        args = build_parser().parse_args([
            "populate", "--dataset", "/tmp/ds", "--all", "--resume", "--dry-run",
        ])
        assert args.command == "populate"
        assert args.all is True
        assert args.resume is True
        assert args.dry_run is True

    def test_populate_imports(
        self, tmp_path: Path, config_file: Path, csv_file: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main([
            "populate",
            "--dataset", str(tmp_path / "ds"),
            "--config", str(config_file),
        ])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["counts"]["imported"] == 2
        assert data["counts"]["failed"] == 0

    def test_populate_dry_run_imports_nothing(
        self, tmp_path: Path, config_file: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main([
            "populate",
            "--dataset", str(tmp_path / "ds"),
            "--config", str(config_file),
            "--dry-run",
        ])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["dry_run"] is True
        assert data["counts"]["imported"] == 0

    def test_populate_writes_report(
        self, tmp_path: Path, config_file: Path,
    ) -> None:
        out = tmp_path / "reports" / "out.json"
        rc = main([
            "populate",
            "--dataset", str(tmp_path / "ds"),
            "--config", str(config_file),
            "--report", str(out),
        ])
        assert rc == 0
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["report_type"] == "population_report"
        assert data["counts"]["imported"] == 2
