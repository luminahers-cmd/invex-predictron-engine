"""Append-only benchmark history store (Project E5).

Every benchmark execution is persisted as its own JSON file under a
history directory.  The store is strictly **append-only**:

* a run may never be overwritten — appending an existing ``run_id``
  raises, and writes go through an atomic temp-file replacement;
* each stored record re-verifies its ``result_hash`` on both write and
  read, so a corrupted/unmodified run is detected, not silently loaded;
* the on-disk entries are canonical (sorted by company id with ordered
  nested containers), independently of the run's in-memory ordering.

Wall-clock timestamps are informational only; hashes and metrics never
depend on them.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.ground_truth_eval.runner import BenchmarkRun, _result_hash

# A run id is used directly as a filename, so keep it strictly safe.
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

DEFAULT_HISTORY_DIR = Path(__file__).resolve().parents[1] / "benchmark_history"


class HistoryError(RuntimeError):
    """Raised for any history-store violation (append, integrity, IO)."""


@dataclass(frozen=True)
class RunSummary:
    """Lightweight header for a stored run (index/listing purposes)."""

    run_id: str
    created_at: datetime
    engine_version: str
    benchmark_version: str
    dataset_name: str
    dataset_hash: str
    result_hash: str
    entry_count: int
    successful_count: int
    failed_count: int
    has_metrics: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data


class BenchmarkHistory:
    """Append-only, hash-verified store of benchmark runs."""

    def __init__(self, root: Path | str = DEFAULT_HISTORY_DIR) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def run_path(self, run_id: str) -> Path:
        if not _RUN_ID_PATTERN.match(run_id):
            raise HistoryError(f"run_id {run_id!r} is not filename-safe")
        return self.root / f"{run_id}.json"

    def index_path(self) -> Path:
        return self.root / "index.json"

    # ------------------------------------------------------------------
    # Write path (append-only)
    # ------------------------------------------------------------------

    def append(
        self,
        run: BenchmarkRun,
        *,
        metrics: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Persist a run. Refuses to overwrite an existing run by default.

        ``metrics`` is an *optional* precomputed metric block (the output
        of :func:`~benchmarks.ground_truth_eval.metrics.compute_metrics`)
        used by trend reports.  If supplied, it is stored verbatim in a
        ``metrics`` section.  It defaults to ``None`` and never causes the
        record to fabricate numbers.
        """
        path = self.run_path(run.run_id)
        if path.exists() and not overwrite:
            raise HistoryError(f"run {run.run_id!r} already exists; refusing to overwrite")

        # Verify integrity before persisting: the run's declared result
        # hash must match its entries (catches accidental mutations).
        computed = _result_hash(run.entries)
        if computed != run.result_hash:
            raise HistoryError(
                f"run {run.run_id!r} result_hash mismatch: "
                f"declared {run.result_hash}, computed {computed}"
            )

        document: dict[str, Any] = {
            "kind": "benchmark_run",
            "run": _run_to_document(run),
        }
        if metrics is not None:
            document["metrics"] = metrics

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload + "\n", encoding="utf-8")
        tmp.replace(path)
        self._update_index(run, has_metrics=metrics is not None)
        return path

    def _update_index(self, run: BenchmarkRun, *, has_metrics: bool) -> None:
        summaries = self.list_summaries()
        existing = {s.run_id for s in summaries}
        if run.run_id in existing:
            return
        summary = RunSummary(
            run_id=run.run_id,
            created_at=run.created_at,
            engine_version=run.engine_version,
            benchmark_version=run.benchmark_version,
            dataset_name=run.dataset_name,
            dataset_hash=run.dataset_hash,
            result_hash=run.result_hash,
            entry_count=len(run.entries),
            successful_count=len(run.successful_entries),
            failed_count=len(run.entries) - len(run.successful_entries),
            has_metrics=has_metrics,
        )
        summaries.append(summary)
        summaries.sort(key=lambda s: (s.created_at, s.run_id))
        payload = json.dumps(
            [s.to_dict() for s in summaries],
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        tmp = self.index_path().with_suffix(".tmp")
        tmp.write_text(payload + "\n", encoding="utf-8")
        tmp.replace(self.index_path())

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def load(self, run_id: str) -> BenchmarkRun:
        """Load and integrity-verify a stored run."""
        path = self.run_path(run_id)
        if not path.exists():
            raise HistoryError(f"run {run_id!r} not found in {self.root}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HistoryError(f"corrupt run file {path}: {exc}") from exc
        if not isinstance(document, dict) or document.get("kind") != "benchmark_run":
            raise HistoryError(f"invalid run file {path}: missing 'benchmark_run' kind")
        run_data = document.get("run")
        if not isinstance(run_data, dict):
            raise HistoryError(f"invalid run file {path}: missing 'run' section")
        try:
            run = BenchmarkRun.model_validate(run_data)
        except Exception as exc:  # noqa: BLE001 - surface malformed records
            raise HistoryError(f"run file {path} failed validation: {exc}") from exc
        computed = _result_hash(run.entries)
        if computed != run.result_hash:
            raise HistoryError(
                f"run {run_id!r} integrity check failed: "
                f"stored result_hash {run.result_hash}, computed {computed}"
            )
        return run

    def load_metrics(self, run_id: str) -> dict[str, Any] | None:
        """Return the optional stored metric block for a run (or None)."""
        path = self.run_path(run_id)
        if not path.exists():
            raise HistoryError(f"run {run_id!r} not found in {self.root}")
        document = json.loads(path.read_text(encoding="utf-8"))
        block = document.get("metrics")
        return block if isinstance(block, dict) else None

    def has_run(self, run_id: str) -> bool:
        return self.run_path(run_id).exists()

    # ------------------------------------------------------------------
    # Listing / index
    # ------------------------------------------------------------------

    def list_summaries(self) -> list[RunSummary]:
        """Return stored runs sorted by (created_at, run_id)."""
        index = self.index_path()
        if not index.exists():
            return []
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        summaries: list[RunSummary] = []
        for entry in data:
            try:
                d = dict(entry)
                d["created_at"] = datetime.fromisoformat(str(d["created_at"]))
                summaries.append(RunSummary(**d))
            except (KeyError, TypeError, ValueError):
                continue
        summaries.sort(key=lambda s: (s.created_at, s.run_id))
        return summaries

    def list_run_ids(self, *, dataset_name: str | None = None) -> list[str]:
        ids = [
            s.run_id
            for s in self.list_summaries()
            if dataset_name is None or s.dataset_name == dataset_name
        ]
        return ids

    def list_benchmark_versions(self) -> list[str]:
        versions = sorted({s.benchmark_version for s in self.list_summaries()})
        return versions

    @property
    def count(self) -> int:
        return len(self.list_summaries())

    def verify_integrity(self, run_ids: list[str] | None = None) -> dict[str, str]:
        """Re-verify every stored run's result hash.

        Returns a mapping of ``run_id -> 'ok'`` (or 'ok') — raises on the
        first corrupted run so callers can surface the failure loudly.
        """
        ids = run_ids or self.list_run_ids()
        status: dict[str, str] = {}
        for run_id in ids:
            self.load(run_id)
            status[run_id] = "ok"
        return status


def _run_to_document(run: BenchmarkRun) -> dict[str, Any]:
    """Serialize a run canonically (entries ordered by company id)."""
    ordered = sorted((e.sorted() for e in run.entries), key=lambda e: e.company_id)
    return {
        "run_id": run.run_id,
        "created_at": run.created_at.isoformat(),
        "engine_version": run.engine_version,
        "benchmark_version": run.benchmark_version,
        "dataset_name": run.dataset_name,
        "dataset_hash": run.dataset_hash,
        "result_hash": run.result_hash,
        "entries": [e.model_dump(mode="json") for e in ordered],
    }


__all__ = [
    "HistoryError",
    "RunSummary",
    "BenchmarkHistory",
    "DEFAULT_HISTORY_DIR",
]
