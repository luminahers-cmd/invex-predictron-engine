"""Tests for the cohort manifest CSV bridge (Phase 4).

Covers the CSV dump/load round trip, the extension-based dispatch of
:func:`load_manifest_any`, deterministic cell encoding, and the strict
parse errors the loader raises for malformed rows.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmarks.cohort.manifest import (
    CohortManifest,
    ManifestError,
    ManifestOutcome,
    ManifestSourceRecord,
)
from benchmarks.cohort.manifest_csv import (
    CSV_COLUMNS,
    ManifestFormatError,
    dump_manifest_csv,
    load_manifest_any,
    load_manifest_csv,
    manifest_from_rows,
    manifest_to_rows,
)
from benchmarks.ground_truth.schema import OutcomeEvent, OutcomeEventKind, StartupStatus

ANALYSIS = datetime(2026, 1, 1, tzinfo=UTC)
VERIFICATION = datetime(2026, 6, 30, tzinfo=UTC).date()
CREATED_AT = datetime(2026, 9, 20, tzinfo=UTC)


def _manifest() -> CohortManifest:
    outcome = ManifestOutcome(
        status=StartupStatus.ACQUIRED,
        verification_date=VERIFICATION,
        verified=True,
        verified_by="qa",
        notes="acquired",
        sources=["official filing"],
        outcome_events=[
            OutcomeEvent(
                kind=OutcomeEventKind.ACQUISITION,
                occurred_at=datetime(2026, 4, 15, tzinfo=UTC).date(),
                value_currency_usd=120_000_000.0,
                amount_nominal_units=None,
                description="Team acquired",
                sources=["filing"],
            ),
            OutcomeEvent(
                kind=OutcomeEventKind.FUNDING,
                occurred_at=datetime(2025, 5, 1, tzinfo=UTC).date(),
                value_currency_usd=5_000_000.0,
                amount_nominal_units=None,
                description="Series A",
                sources=["crunchbase"],
            ),
        ],
    )
    records = [
        ManifestSourceRecord(
            company_id="cmp-1",
            company_name="Example One",
            description="First test company with a documented outcome.",
            website="https://one.example.com",
            pitch_deck_url="https://one.example.com/deck.pdf",
            founder_linkedin_urls=["https://linkedin.com/in/a"],
            sector="saas",
            country="US",
            stage="series_a",
            evidence_corpus="b2b_saas",
            analysis_timestamp=ANALYSIS,
            evaluation_horizon_days=180,
            outcome=outcome,
            metadata={"sources": ["tests"]},
        ),
        ManifestSourceRecord(
            company_id="cmp-2",
            company_name="Example Two",
            description="Second test company awaiting ground truth.",
            website=None,
            evidence_corpus="ai_infrastructure",
            analysis_timestamp=ANALYSIS,
            outcome=None,
        ),
    ]
    return CohortManifest(
        dataset_name="test_cohort_csv",
        benchmark_version="1.0",
        created_at=CREATED_AT,
        evaluation_horizon_days=90,
        source_records=records,
    )


class TestRows:
    def test_rows_cover_expected_columns(self) -> None:
        rows = manifest_to_rows(_manifest())
        # Manifest-origin row comes first, then records sorted by company_id.
        assert len(rows) == 3
        assert rows[0]["section"] == "manifest"
        assert rows[0]["dataset_name"] == "test_cohort_csv"

        record_keys = set(rows[1].keys())
        assert record_keys <= set(CSV_COLUMNS)
        assert rows[1]["company_id"] == "cmp-1"
        assert rows[1]["company_name"] == "Example One"
        assert rows[1]["outcome_status"] == "acquired"
        assert rows[1]["outcome_verification_date"] == "2026-06-30"
        assert rows[1]["outcome_verified"] == "1"
        assert rows[1]["record_evaluation_horizon_days"] == "180"
        assert rows[1]["founder_linkedin_urls_json"] == '["https://linkedin.com/in/a"]'
        assert rows[1]["record_metadata_json"] == '{"sources":["tests"]}'

    def test_rows_are_deterministic(self) -> None:
        assert manifest_to_rows(_manifest()) == manifest_to_rows(_manifest())

    def test_row_without_outcome_uses_empty_cells(self) -> None:
        row = manifest_to_rows(_manifest())[2]
        assert row["company_id"] == "cmp-2"
        assert row["outcome_status"] == ""
        assert row["outcome_verification_date"] == ""
        assert row["outcome_verified"] == "0"
        assert row["website"] == ""

    def test_manifest_from_rows_builds_manifest(self) -> None:
        manifest = manifest_from_rows(manifest_to_rows(_manifest()))
        assert manifest == _manifest()
        assert len(manifest.source_records) == 2
        assert manifest.source_records[0].outcome is not None
        assert len(manifest.source_records[0].outcome.outcome_events) == 2

    def test_outcome_events_survive_roundtrip(self) -> None:
        manifest = manifest_from_rows(manifest_to_rows(_manifest()))
        events = manifest.source_records[0].outcome.outcome_events
        assert events[0].kind == OutcomeEventKind.ACQUISITION
        assert events[0].value_currency_usd == 120_000_000.0
        assert events[1].kind == OutcomeEventKind.FUNDING
        assert events[1].sources == ["crunchbase"]

    def test_metadata_survives_roundtrip(self) -> None:
        manifest = manifest_from_rows(manifest_to_rows(_manifest()))
        assert manifest.source_records[0].metadata == {"sources": ["tests"]}


class TestCsvFiles:
    def test_dump_load_roundtrip(self, tmp_path: Path) -> None:
        path = dump_manifest_csv(_manifest(), tmp_path / "cohort.csv")
        loaded = load_manifest_csv(path)
        assert loaded == _manifest()
        assert loaded.dataset_name == "test_cohort_csv"

    def test_dump_is_atomic_and_readable(self, tmp_path: Path) -> None:
        path = dump_manifest_csv(_manifest(), tmp_path / "cohort.csv")
        text = path.read_text(encoding="utf-8")
        assert text.splitlines()[0].startswith("section,schema_version")
        assert "company_id" in text.splitlines()[0]
        assert not list(tmp_path.glob("*.tmp"))

    def test_refuses_clobber(self, tmp_path: Path) -> None:
        path = dump_manifest_csv(_manifest(), tmp_path / "cohort.csv")
        with pytest.raises(ManifestError, match="already exists"):
            dump_manifest_csv(_manifest(), path)

    def test_load_manifest_any_dispatches_on_suffix(self, tmp_path: Path) -> None:
        from benchmarks.cohort.manifest import dump_manifest

        csv_path = dump_manifest_csv(_manifest(), tmp_path / "cohort.csv")
        json_path = tmp_path / "cohort.json"
        dump_manifest(_manifest(), json_path)
        assert load_manifest_any(csv_path) == _manifest()
        assert load_manifest_any(json_path) == _manifest()

        # No suffix (model_dump-style reference lists) defaults to JSON parsing.
        bare = tmp_path / "manifest"
        bare.write_text(
            _manifest().model_dump_json(indent=2), encoding="utf-8"
        )
        assert load_manifest_any(bare) == _manifest()

    def test_load_manifest_any_rejects_unknown_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "cohort.xlsx"
        path.write_text("not a manifest", encoding="utf-8")
        with pytest.raises(ManifestError, match="unsupported manifest format"):
            load_manifest_any(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="not found"):
            load_manifest_csv(tmp_path / "missing.csv")


class TestCsvValidation:
    @staticmethod
    def _write_record_csv(tmp_path: Path, **cell_overrides: str) -> Path:
        """Write a CSV with the full column header and a single record row.

        ``cell_overrides`` sets the value written for a column (all other
        non-required columns stay empty), so the strict loader reaches the
        per-field validation logic.
        """
        defaults: dict[str, str] = {
            "section": "record",
            "company_id": "c1",
            "company_name": "Example",
            "description": "desc",
            "evidence_corpus": "b2b_saas",
            "analysis_timestamp": "2026-01-01T00:00:00+00:00",
        }
        row = {
            column: cell_overrides.get(column, defaults.get(column, ""))
            for column in CSV_COLUMNS
        }
        path = tmp_path / "bad.csv"
        path.write_text(
            ",".join(CSV_COLUMNS) + "\n" + ",".join(row[column] for column in CSV_COLUMNS) + "\n",
            encoding="utf-8",
        )
        return path

    def test_missing_required_column_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text(
            "company_name,description\nexample,desc\n", encoding="utf-8"
        )
        with pytest.raises(ManifestFormatError, match="company_id"):
            load_manifest_csv(path)

    def test_outcome_without_verification_date_rejected(self, tmp_path: Path) -> None:
        path = self._write_record_csv(
            tmp_path, outcome_status="acquired", outcome_verification_date=""
        )
        with pytest.raises(ManifestFormatError, match="outcome_verification_date"):
            load_manifest_csv(path)

    def test_blank_analysis_timestamp_rejected(self, tmp_path: Path) -> None:
        path = self._write_record_csv(tmp_path, analysis_timestamp="")
        with pytest.raises(ManifestFormatError, match="analysis_timestamp"):
            load_manifest_csv(path)

    def test_negative_horizon_rejected(self, tmp_path: Path) -> None:
        path = self._write_record_csv(tmp_path, record_evaluation_horizon_days="-5")
        with pytest.raises(ManifestFormatError, match="record_evaluation_horizon_days"):
            load_manifest_csv(path)

    def test_bad_float_cell_rejected(self, tmp_path: Path) -> None:
        path = self._write_record_csv(
            tmp_path,
            outcome_status="acquired",
            outcome_verification_date="2026-06-30",
            outcome_exit_value_usd="abc",
        )
        with pytest.raises(ManifestFormatError, match="exit_value_usd"):
            load_manifest_csv(path)
