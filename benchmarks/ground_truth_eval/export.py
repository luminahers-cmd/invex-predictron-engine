"""Deterministic export/import of benchmark artifacts (Project E5).

Provides canonical JSON payloads and atomic, deterministic writers for
golden datasets, benchmark runs, metric blocks, and reports.  Re-exporting
the same artifact always produces byte-identical JSON, so exports are
suitable for diffing, checksumming, and long-term auditability.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.ground_truth_eval.metrics import GroundTruthMetrics
from benchmarks.ground_truth_eval.models import GoldenDataset, GoldenEntry
from benchmarks.ground_truth_eval.reports import REPORT_KINDS
from benchmarks.ground_truth_eval.runner import BenchmarkRun

EXPORT_SCHEMA_VERSION = 1


class ExportError(RuntimeError):
    """Raised when an export/import payload is invalid or IO fails."""


def canonical_json(obj: Any) -> str:
    """Serialize ``obj`` to deterministic, sorted, indented JSON."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"


def write_json(
    payload: Any,
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a canonical JSON payload, refusing to overwrite."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise ExportError(f"refusing to overwrite existing file: {path}")
    if not path.name.endswith(".json"):
        raise ExportError(f"export path must end with .json: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(canonical_json(payload), encoding="utf-8")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def export_dataset_payload(dataset: GoldenDataset) -> dict[str, Any]:
    """Canonical JSON document for a golden dataset."""
    return {
        "schema_version": 1,
        "dataset_name": dataset.dataset_name,
        "benchmark_version": dataset.benchmark_version,
        "created_at": dataset.created_at.isoformat(),
        "entries": [e.model_dump(mode="json") for e in dataset.entries],
    }


def export_run_payload(run: BenchmarkRun) -> dict[str, Any]:
    """Canonical JSON document for a benchmark run (entries ordered)."""
    ordered = sorted(run.entries, key=lambda e: e.company_id)
    return {
        "kind": "benchmark_run",
        "run_id": run.run_id,
        "created_at": run.created_at.isoformat(),
        "engine_version": run.engine_version,
        "benchmark_version": run.benchmark_version,
        "dataset_name": run.dataset_name,
        "dataset_hash": run.dataset_hash,
        "result_hash": run.result_hash,
        "entries": [e.sorted().model_dump(mode="json") for e in ordered],
    }


def export_metrics_payload(metrics: GroundTruthMetrics) -> dict[str, Any]:
    """Canonical JSON document for a metric block."""
    return metrics.to_dict()


def export_bundle_payload(
    *,
    dataset: GoldenDataset,
    run: BenchmarkRun,
    metrics: GroundTruthMetrics,
    reports: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A single self-contained export document for a complete evaluation."""
    payload: dict[str, Any] = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "dataset": export_dataset_payload(dataset),
        "run": export_run_payload(run),
        "metrics": export_metrics_payload(metrics),
    }
    if reports is not None:
        payload["reports"] = {kind: reports[kind] for kind in REPORT_KINDS if kind in reports}
    return payload


# ---------------------------------------------------------------------------
# High-level export helpers
# ---------------------------------------------------------------------------


def export_dataset(
    dataset: GoldenDataset,
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    return write_json(export_dataset_payload(dataset), path, overwrite=overwrite)


def export_run(
    run: BenchmarkRun,
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    return write_json(export_run_payload(run), path, overwrite=overwrite)


def export_metrics(
    metrics: GroundTruthMetrics,
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    return write_json(export_metrics_payload(metrics), path, overwrite=overwrite)


def export_report(
    report: dict[str, Any],
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    return write_json(report, path, overwrite=overwrite)


def export_bundle(
    *,
    dataset: GoldenDataset,
    run: BenchmarkRun,
    metrics: GroundTruthMetrics,
    reports: dict[str, Any] | None = None,
    path: Path | str,
    overwrite: bool = False,
) -> Path:
    return write_json(
        export_bundle_payload(dataset=dataset, run=run, metrics=metrics, reports=reports),
        path,
        overwrite=overwrite,
    )


# ---------------------------------------------------------------------------
# Import / round-trip
# ---------------------------------------------------------------------------


def read_json(path: Path | str) -> Any:
    path = Path(path)
    if not path.exists():
        raise ExportError(f"file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExportError(f"invalid JSON in {path}: {exc}") from exc


def import_dataset_payload(payload: dict[str, Any]) -> GoldenDataset:
    """Reconstruct a dataset from an exported dataset document.

    Goes through the canonical model so validation rules are shared with
    the ``DEFAULT_GOLDEN_DIR`` loader and schema checks are enforced.
    """
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int) or schema_version != 1:
        raise ExportError(f"unsupported dataset schema_version={schema_version!r}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ExportError("dataset export has no entries")
    try:
        return GoldenDataset(
            dataset_name=str(payload["dataset_name"]),
            benchmark_version=str(payload["benchmark_version"]),
            created_at=datetime.fromisoformat(str(payload["created_at"])),
            entries=[GoldenEntry.model_validate(e) for e in entries],
        )
    except Exception as exc:  # noqa: BLE001 - surface malformed exports
        raise ExportError(f"dataset export failed validation: {exc}") from exc


def import_run_payload(payload: dict[str, Any]) -> BenchmarkRun:
    if payload.get("kind") != "benchmark_run":
        raise ExportError("run export missing 'benchmark_run' kind")
    try:
        run = BenchmarkRun.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - surface malformed exports
        raise ExportError(f"run export failed validation: {exc}") from exc
    return run


def load_exported(path: Path | str) -> dict[str, Any]:
    """Load a raw exported document (dataset/run/metrics/reports)."""
    return read_json(path)  # type: ignore[no-any-return]


def make_bundle_paths(
    directory: Path | str,
    *,
    run_id: str,
) -> dict[str, Path]:
    """Stable filenames for a full bundle export of one run."""
    base = Path(directory)
    return {
        "dataset": base / f"{run_id}.dataset.json",
        "run": base / f"{run_id}.run.json",
        "metrics": base / f"{run_id}.metrics.json",
        "bundle": base / f"{run_id}.bundle.json",
        "reports": base / f"{run_id}.reports.json",
    }


__all__ = [
    "EXPORT_SCHEMA_VERSION",
    "ExportError",
    "canonical_json",
    "write_json",
    "export_dataset_payload",
    "export_run_payload",
    "export_metrics_payload",
    "export_bundle_payload",
    "export_dataset",
    "export_run",
    "export_metrics",
    "export_report",
    "export_bundle",
    "read_json",
    "import_dataset_payload",
    "import_run_payload",
    "load_exported",
    "make_bundle_paths",
]
