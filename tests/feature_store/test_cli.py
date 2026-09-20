"""CLI interface tests for feature-store commands."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from predictron_engine.feature_store.cli import main as cli_main


class TestCLIBuild:
    def test_feature_build_no_records(self, tmp_path):
        with patch(
            "predictron_engine.feature_store.cli._load_dataset_store"
        ) as mock_ds:
            mock_ds.return_value = MagicMock(
                list_records=MagicMock(return_value=[]),
            )
            cli_main([
                "--data-dir", str(tmp_path),
                "--feature-dir", str(tmp_path / "features"),
                "feature-build",
            ])

    def test_feature_build_with_records(self, tmp_path):
        record = MagicMock()
        record.record_id = "rec-1"
        record.founded_year = 2022
        record.funding_stage = None
        record.profile.founded_year = 2022
        record.profile.employee_range = None
        record.profile.employee_count = 10
        record.profile.industries = []

        mock_ds = MagicMock()
        mock_ds.list_records.return_value = ["rec-1"]
        mock_ds.load_record.return_value = record
        mock_ds.load_timeline.return_value = None

        features_dir = tmp_path / "features"
        with patch(
            "predictron_engine.feature_store.cli._load_dataset_store",
            return_value=mock_ds,
        ):
            cli_main([
                "--data-dir", str(tmp_path),
                "--feature-dir", str(features_dir),
                "feature-build",
                "-v",
            ])

        company_file = features_dir / "companies" / "rec-1.json"
        assert company_file.exists()
        data = json.loads(company_file.read_text(encoding="utf-8"))
        assert data["company_id"] == "rec-1"
        assert len(data["features"]) > 0


class TestCLIReport:
    def test_feature_report_empty_store(self, tmp_path):
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        cli_main([
            "--feature-dir", str(features_dir),
            "feature-report",
        ])

    def test_feature_report_with_company(self, tmp_path):
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        company_file = features_dir / "rec-1.json"
        fs_data = {
            "company_id": "rec-1",
            "built_at": "2024-01-01T00:00:00+00:00",
            "as_of": None,
            "features": {
                "company_age": {
                    "snapshot_id": "s1",
                    "company_id": "rec-1",
                    "feature_id": "company_age",
                    "feature_name": "Company Age",
                    "category": "company",
                    "value": 4.0,
                    "value_type": "float",
                    "status": "computed",
                    "computation_version": "1.0.0",
                    "computed_at": "2024-01-01T00:00:00+00:00",
                    "evidence_references": [],
                },
            },
        }
        company_file.write_text(
            json.dumps(fs_data), encoding="utf-8",
        )
        cli_main([
            "--feature-dir", str(features_dir),
            "feature-report",
        ])

    def test_feature_report_to_file(self, tmp_path):
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        output_file = tmp_path / "report.json"
        cli_main([
            "--feature-dir", str(features_dir),
            "feature-report",
            "-o", str(output_file),
        ])
        assert output_file.exists()


class TestCLIValidate:
    def test_feature_validate_empty(self, tmp_path):
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        cli_main([
            "--feature-dir", str(features_dir),
            "feature-validate",
        ])

    def test_feature_validate_single_company(self, tmp_path):
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        cli_main([
            "--feature-dir", str(features_dir),
            "feature-validate",
            "--company-id", "rec-1",
        ])


class TestCLIHistory:
    def test_feature_history_no_company(self, tmp_path):
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        with pytest.raises(SystemExit):
            cli_main([
                "--feature-dir", str(features_dir),
                "feature-history",
            ])

    def test_feature_history_empty(self, tmp_path):
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        cli_main([
            "--feature-dir", str(features_dir),
            "feature-history",
            "--company-id", "rec-1",
        ])


class TestCLIExport:
    def test_feature_export_default(self, tmp_path):
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        export_file = tmp_path / "features" / "export.json"
        cli_main([
            "--feature-dir", str(features_dir),
            "feature-export",
            "-o", str(export_file),
        ])
        assert export_file.exists()
        data = json.loads(export_file.read_text(encoding="utf-8"))
        assert "company_count" in data


class TestCLIRegistry:
    def test_feature_registry_default(self, tmp_path):
        cli_main([
            "--feature-dir", str(tmp_path),
            "feature-registry",
        ])

    def test_feature_registry_by_category(self, tmp_path):
        cli_main([
            "--feature-dir", str(tmp_path),
            "feature-registry",
            "--category", "company",
        ])

    def test_feature_registry_invalid_category(self, tmp_path):
        with pytest.raises(SystemExit):
            cli_main([
                "--feature-dir", str(tmp_path),
                "feature-registry",
                "--category", "nonexistent",
            ])


class TestCLINoCommand:
    def test_no_command_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            cli_main([])
