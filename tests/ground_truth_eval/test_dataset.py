"""Tests for golden dataset loading, validation, and hashing."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmarks.ground_truth.schema import StartupStatus
from benchmarks.ground_truth_eval.dataset import (
    DEFAULT_GOLDEN_DIR,
    DatasetIssue,
    GoldenDatasetError,
    IssueSeverity,
    canonical_bytes,
    canonical_json,
    dataset_hash,
    discover_golden_datasets,
    dump_golden_dataset,
    load_golden_dataset,
    load_golden_dataset_bytes,
    validate_golden_dataset,
)
from benchmarks.ground_truth_eval.models import (
    GOLDEN_DATASET_SCHEMA_VERSION,
    GoldenDataset,
    GoldenEntry,
    HistoricalEvidence,
    HistoricalPrediction,
    Provenance,
    VerifiedOutcome,
)


def _minimal_entry(
    *,
    company_id: str = "acme_1",
    startup_name: str = "Acme Corp",
    description: str = "A sufficiently long startup description.",
    corpus_name: str | None = "b2b_saas",
    status: StartupStatus = StartupStatus.OPERATING,
    arr_usd: float | None = 4_000_000,
) -> GoldenEntry:
    return GoldenEntry(
        company_id=company_id,
        company_name=startup_name,
        request={"startup_name": startup_name, "description": description},
        historical_prediction=HistoricalPrediction(
            overall_score=70.0, overall_confidence=0.7, decision="invest"
        ),
        historical_evidence=HistoricalEvidence(
            corpus_name=corpus_name,
            sources=["test"],
        ),
        verified_outcome=VerifiedOutcome(
            status=status,
            verification_date=datetime(2026, 1, 1, tzinfo=UTC).date(),
            sources=["test"],
            exit_value_usd=None,
            outcome_events=[],
        ),
        provenance=Provenance(is_example=True, sources=["test"]),
    )


def _minimal_dataset(
    entries: list[GoldenEntry] | None = None,
    *,
    name: str = "test",
) -> GoldenDataset:
    return GoldenDataset(
        dataset_name=name,
        benchmark_version="1.0",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        entries=entries or [_minimal_entry()],
    )


class TestDatasetHash:
    def test_determinism(self) -> None:
        d = _minimal_dataset()
        h1 = dataset_hash(d)
        h2 = dataset_hash(d)
        assert h1 == h2

    def test_different_dataset_different_hash(self) -> None:
        d1 = _minimal_dataset()
        d2 = _minimal_dataset(entries=[_minimal_entry(company_id="x")])
        assert dataset_hash(d1) != dataset_hash(d2)

    def test_hash_is_hex64(self) -> None:
        h = dataset_hash(_minimal_dataset())
        assert len(h) == 64
        int(h, 16)


class TestCanonicalJson:
    def test_deterministic_output(self) -> None:
        d = _minimal_dataset()
        j1 = canonical_json(d)
        j2 = canonical_json(d)
        assert j1 == j2

    def test_keys_are_sorted(self) -> None:
        d = _minimal_dataset()
        j = canonical_json(d)
        keys = list(j.keys())
        assert keys == sorted(keys)


class TestCanonicalBytes:
    def test_is_bytes(self) -> None:
        b = canonical_bytes(_minimal_dataset())
        assert isinstance(b, bytes)

    def test_stable_determinism(self) -> None:
        d = _minimal_dataset()
        assert canonical_bytes(d) == canonical_bytes(d)


class TestLoadGoldenDataset:
    def test_load_valid_file(self, tmp_path: Path) -> None:
        d = _minimal_dataset()
        path = tmp_path / "d.json"
        dump_golden_dataset(d, path)
        loaded = load_golden_dataset(path)
        assert loaded.dataset_name == "test"
        assert len(loaded.entries) == 1

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(GoldenDatasetError, match="not found"):
            load_golden_dataset(tmp_path / "nope.json")

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{", encoding="utf-8")
        with pytest.raises(GoldenDatasetError, match="invalid JSON"):
            load_golden_dataset(path)

    def test_load_non_object(self, tmp_path: Path) -> None:
        path = tmp_path / "arr.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(GoldenDatasetError, match="not an object"):
            load_golden_dataset(path)

    def test_load_bad_schema(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 9999,
                    "dataset_name": "bad",
                    "benchmark_version": "1.0",
                    "entries": [],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(GoldenDatasetError):
            load_golden_dataset(path)


class TestLoadGoldenDatasetBytes:
    def test_valid(self) -> None:
        d = _minimal_dataset()
        payload = canonical_bytes(d)
        loaded = load_golden_dataset_bytes(payload)
        assert loaded.dataset_name == "test"

    def test_invalid_json_bytes(self) -> None:
        with pytest.raises(GoldenDatasetError, match="invalid"):
            load_golden_dataset_bytes(b"{bad")

    def test_non_object(self) -> None:
        with pytest.raises(GoldenDatasetError, match="not an object"):
            load_golden_dataset_bytes(b"[]")


class TestDumpGoldenDataset:
    def test_creates_file(self, tmp_path: Path) -> None:
        d = _minimal_dataset()
        path = tmp_path / "out.json"
        result = dump_golden_dataset(d, path)
        assert result == path
        assert path.exists()

    def test_deterministic_content(self, tmp_path: Path) -> None:
        d = _minimal_dataset()
        p1 = tmp_path / "a.json"
        p2 = tmp_path / "b.json"
        dump_golden_dataset(d, p1)
        dump_golden_dataset(d, p2)
        assert p1.read_bytes() == p2.read_bytes()

    def test_refuses_overwrite(self, tmp_path: Path) -> None:
        d = _minimal_dataset()
        path = tmp_path / "d.json"
        dump_golden_dataset(d, path)
        with pytest.raises(GoldenDatasetError, match="already exists"):
            dump_golden_dataset(d, path)

    def test_overwrite_flag(self, tmp_path: Path) -> None:
        d = _minimal_dataset()
        path = tmp_path / "d.json"
        dump_golden_dataset(d, path)
        dump_golden_dataset(d, path, overwrite=True)
        assert path.exists()


class TestDiscoverGoldenDatasets:
    def test_empty_dir(self, tmp_path: Path) -> None:
        assert discover_golden_datasets(tmp_path) == []

    def test_finds_json(self, tmp_path: Path) -> None:
        (tmp_path / "a.json").write_text("{}")
        (tmp_path / "b.txt").write_text("")
        found = discover_golden_datasets(tmp_path)
        assert len(found) == 1
        assert found[0].name == "a.json"

    def test_sorted(self, tmp_path: Path) -> None:
        for name in ("c.json", "a.json", "b.json"):
            (tmp_path / name).write_text("{}")
        names = [p.name for p in discover_golden_datasets(tmp_path)]
        assert names == ["a.json", "b.json", "c.json"]

    def test_default_root(self) -> None:
        found = discover_golden_datasets()
        assert len(found) >= 1
        assert any("example" in p.name for p in found)


class TestValidation:
    def test_empty_dataset_error(self) -> None:
        d = GoldenDataset(dataset_name="x", benchmark_version="1.0", entries=[])
        report = validate_golden_dataset(d)
        assert not report.is_valid
        assert any("no entries" in i.message for i in report.issues)

    def test_duplicate_id_error(self) -> None:
        e = _minimal_entry(company_id="dup")
        d = _minimal_dataset(entries=[e, _minimal_entry(company_id="dup")])
        report = validate_golden_dataset(d)
        assert not report.is_valid
        assert any("duplicate" in i.message for i in report.issues)

    def test_short_description_error(self) -> None:
        e = _minimal_entry(description="short")
        report = validate_golden_dataset(_minimal_dataset(entries=[e]))
        assert not report.is_valid
        assert any("short" in i.message for i in report.issues)

    def test_missing_startup_name_error(self) -> None:
        e = GoldenEntry(
            company_id="x",
            company_name="X",
            request={},
            historical_prediction=HistoricalPrediction(
                overall_score=50.0, overall_confidence=0.5, decision="invest"
            ),
            historical_evidence=HistoricalEvidence(),
            verified_outcome=VerifiedOutcome(
                status=StartupStatus.OPERATING,
                verification_date=datetime(2026, 1, 1, tzinfo=UTC).date(),
            ),
            provenance=Provenance(is_example=True),
        )
        report = validate_golden_dataset(_minimal_dataset(entries=[e]))
        assert not report.is_valid

    def test_no_evidence_reference_error(self) -> None:
        e = _minimal_entry(corpus_name=None)
        report = validate_golden_dataset(_minimal_dataset(entries=[e]))
        assert not report.is_valid
        assert any("evidence" in i.message.lower() for i in report.issues)

    def test_unscoreable_outcome_warning(self) -> None:
        e = _minimal_entry(status=StartupStatus.OPERATING, arr_usd=None)
        report = validate_golden_dataset(_minimal_dataset(entries=[e]))
        assert report.is_valid
        assert any("not binary-scoreable" in i.message for i in report.issues)
        assert report.warning_count >= 1

    def test_warning_severity(self, entry_builder) -> None:
        assert IssueSeverity.WARNING == "warning"
        assert IssueSeverity.ERROR == "error"
        di = DatasetIssue(severity="warning", message="x")
        assert di.severity == "warning"

    def test_validation_report_to_dict(self, entry_builder) -> None:
        e = entry_builder()
        d = _minimal_dataset(entries=[e])
        report = validate_golden_dataset(d)
        as_dict = report.to_dict()
        assert as_dict["is_valid"] is True
        assert isinstance(as_dict["issues"], list)


class TestSchemaConstants:
    def test_version_matches(self) -> None:
        assert GOLDEN_DATASET_SCHEMA_VERSION == 1

    def test_default_golden_dir_exists(self) -> None:
        assert DEFAULT_GOLDEN_DIR.exists()
        assert DEFAULT_GOLDEN_DIR.is_dir()
