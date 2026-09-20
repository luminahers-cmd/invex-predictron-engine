"""Golden dataset loading, validation, and hashing.

A golden dataset is a single JSON document (indented, sorted keys) that
validates against :class:`~benchmarks.ground_truth_eval.models.GoldenDataset`.
Loading is strict: malformed or schema-incompatible documents raise
:class:`GoldenDatasetError` instead of silently degrading.

Hashing is canonical and deterministic: the dataset is serialized to JSON
with sorted keys and stable separators, then sha256 over the UTF-8 bytes.
Reordering entries, changing field order, or reformatting the file does
NOT change the hash; changing any *value* does.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.ground_truth_eval.models import (
    GOLDEN_DATASET_SCHEMA_VERSION,
    GoldenDataset,
)


class GoldenDatasetError(RuntimeError):
    """Raised when a golden dataset cannot be loaded or has an invalid schema."""


# Committed golden dataset root.
DEFAULT_GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden_datasets"


def canonical_json(dataset: GoldenDataset) -> dict[str, Any]:
    """Return the canonical JSON-safe dict view of a dataset.

    Uses ``mode="json"`` so dates/uuids serialize to strings and the dict is
    JSON-round-trip stable, then orders keys recursively so two datasets
    with equal values always produce equal bytes.
    """
    return _sort_keys(dataset.model_dump(mode="json"))  # type: ignore[no-any-return]


def _sort_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sort_keys(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sort_keys(v) for v in value]
    return value


def canonical_bytes(dataset: GoldenDataset) -> bytes:
    """Return deterministic UTF-8 bytes for a dataset (sorted keys)."""
    return json.dumps(
        canonical_json(dataset), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def dataset_hash(dataset: GoldenDataset) -> str:
    """Return the canonical sha256 (hex) fingerprint of a dataset."""
    return hashlib.sha256(canonical_bytes(dataset)).hexdigest()


def load_golden_dataset(path: Path | str) -> GoldenDataset:
    """Load and validate a golden dataset from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise GoldenDatasetError(f"golden dataset not found: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GoldenDatasetError(f"invalid JSON in golden dataset {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise GoldenDatasetError(f"golden dataset {path} is not an object")
    try:
        dataset = GoldenDataset.model_validate(document)
    except Exception as exc:  # noqa: BLE001 - surface schema violations
        raise GoldenDatasetError(f"golden dataset {path} failed validation: {exc}") from exc
    if dataset.schema_version != GOLDEN_DATASET_SCHEMA_VERSION:
        raise GoldenDatasetError(
            f"golden dataset {path} has unsupported schema_version="
            f"{dataset.schema_version} (expected {GOLDEN_DATASET_SCHEMA_VERSION})"
        )
    return dataset


def load_golden_dataset_bytes(payload: bytes) -> GoldenDataset:
    """Load and validate a golden dataset from raw JSON bytes."""
    try:
        document = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise GoldenDatasetError(f"invalid golden dataset JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise GoldenDatasetError("golden dataset JSON is not an object")
    try:
        dataset = GoldenDataset.model_validate(document)
    except Exception as exc:  # noqa: BLE001 - surface schema violations
        raise GoldenDatasetError(f"golden dataset failed validation: {exc}") from exc
    if dataset.schema_version != GOLDEN_DATASET_SCHEMA_VERSION:
        raise GoldenDatasetError(
            f"golden dataset has unsupported schema_version="
            f"{dataset.schema_version} (expected {GOLDEN_DATASET_SCHEMA_VERSION})"
        )
    return dataset


def dump_golden_dataset(
    dataset: GoldenDataset,
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a golden dataset to disk (deterministic, atomic).

    Refuses to overwrite an existing file unless ``overwrite=True`` so an
    accidentally re-generated dataset can never clobber a curated one.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise GoldenDatasetError(
            f"golden dataset already exists: {path} (pass overwrite=True to replace)"
        )
    payload = json.dumps(canonical_json(dataset), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return path


def discover_golden_datasets(root: Path | str | None = None) -> list[Path]:
    """Return all ``*.json`` golden dataset files under a root (sorted)."""
    root = Path(root) if root is not None else DEFAULT_GOLDEN_DIR
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix == ".json")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class IssueSeverity(str):
    ERROR = "error"
    WARNING = "warning"


@dataclass
class DatasetIssue:
    """One validation finding about a golden dataset."""

    severity: str
    message: str
    company_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "message": self.message,
            "company_id": self.company_id,
        }


@dataclass
class ValidationReport:
    """Aggregate validation result for a golden dataset."""

    issues: list[DatasetIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(i.severity == IssueSeverity.ERROR for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)

    def to_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
        }


def validate_golden_dataset(dataset: GoldenDataset) -> ValidationReport:
    """Validate a golden dataset for curation errors.

    Checks that never fabricate or repair data: entries must have unique
    ids, valid requests, resolvable evidence references, and consistent
    outcome vocabulary.  Un-scoreable outcomes are warnings, not errors.
    """
    report = ValidationReport()

    if dataset.schema_version != GOLDEN_DATASET_SCHEMA_VERSION:
        report.issues.append(
            DatasetIssue(
                severity=IssueSeverity.ERROR,
                message=(
                    f"unsupported schema_version={dataset.schema_version} "
                    f"(expected {GOLDEN_DATASET_SCHEMA_VERSION})"
                ),
            )
        )

    if not dataset.entries:
        report.issues.append(
            DatasetIssue(severity=IssueSeverity.ERROR, message="dataset has no entries")
        )

    seen: set[str] = set()
    for entry in dataset.entries:
        if entry.company_id in seen:
            report.issues.append(
                DatasetIssue(
                    severity=IssueSeverity.ERROR,
                    message=f"duplicate company_id {entry.company_id!r}",
                    company_id=entry.company_id,
                )
            )
        seen.add(entry.company_id)

        if not entry.request.get("startup_name"):
            report.issues.append(
                DatasetIssue(
                    severity=IssueSeverity.ERROR,
                    message="missing startup_name in request",
                    company_id=entry.company_id,
                )
            )
        description = entry.request.get("description") or ""
        if len(description) < 10:
            report.issues.append(
                DatasetIssue(
                    severity=IssueSeverity.ERROR,
                    message="description too short (< 10 chars)",
                    company_id=entry.company_id,
                )
            )

        if entry.historical_evidence.reference is None:
            report.issues.append(
                DatasetIssue(
                    severity=IssueSeverity.ERROR,
                    message="no historical evidence reference (corpus_name/corpus_path)",
                    company_id=entry.company_id,
                )
            )

        timestamp = entry.analysis_timestamp
        if timestamp is not None:
            if timestamp.tzinfo is None:
                report.issues.append(
                    DatasetIssue(
                        severity=IssueSeverity.ERROR,
                        message=(
                            "analysis_timestamp must be timezone-aware (UTC) for a "
                            "deterministic look-ahead guard"
                        ),
                        company_id=entry.company_id,
                    )
                )
            else:
                observed = entry.verification_date
                if observed < timestamp.date():
                    report.issues.append(
                        DatasetIssue(
                            severity=IssueSeverity.ERROR,
                            message=(
                                "verified outcome observed before analysis_timestamp "
                                "(impossible provenance)"
                            ),
                            company_id=entry.company_id,
                        )
                    )
                horizon = entry.evaluation_horizon_days
                if horizon is not None and observed >= timestamp.date():
                    elapsed = (observed - timestamp.date()).days
                    if elapsed < horizon:
                        report.issues.append(
                            DatasetIssue(
                                severity=IssueSeverity.WARNING,
                                message=(
                                    f"outcome observed {elapsed}d after analysis; "
                                    f"below evaluation_horizon_days={horizon}"
                                ),
                                company_id=entry.company_id,
                            )
                        )

        if not entry.verified_outcome.sources and not entry.provenance.sources:
            report.issues.append(
                DatasetIssue(
                    severity=IssueSeverity.WARNING,
                    message="outcome has no provenance sources",
                    company_id=entry.company_id,
                )
            )

        if not entry.verified_outcome.is_scoreable:
            report.issues.append(
                DatasetIssue(
                    severity=IssueSeverity.WARNING,
                    message=(
                        "outcome is not binary-scoreable "
                        "(un-verified or survival-only); excluded from binary metrics"
                    ),
                    company_id=entry.company_id,
                )
            )

    return report


__all__ = [
    "DEFAULT_GOLDEN_DIR",
    "GoldenDatasetError",
    "DatasetIssue",
    "ValidationReport",
    "IssueSeverity",
    "canonical_json",
    "canonical_bytes",
    "dataset_hash",
    "load_golden_dataset",
    "load_golden_dataset_bytes",
    "dump_golden_dataset",
    "discover_golden_datasets",
    "validate_golden_dataset",
]
