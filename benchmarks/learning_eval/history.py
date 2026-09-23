"""Append-only store for learning-eval benchmark runs.

Every run is persisted as its own JSON file under a history directory.  The
store is strictly **append-only**: writes never overwrite an existing run,
use an atomic temp-file replacement, and re-verify a stored integrity hash
on both write and read so corrupted runs are detected rather than loaded.
Run ids are deterministic (content-derived), so re-running the identical
suite yields the identical id, and a second write to that id is rejected.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from benchmarks.learning_eval.models import (
    LearningEvalReport,
    LearningEvalRun,
)

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

DEFAULT_HISTORY_DIR = Path(__file__).resolve().parents[1] / "learning_eval_history"


class HistoryError(RuntimeError):
    """Raised for any history-store violation (append, integrity, IO)."""


def _record_hash(document: dict[str, Any]) -> str:
    material = json.dumps(
        document, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class LearningEvalHistory:
    """Append-only, hash-verified store of learning-eval runs."""

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

    # ------------------------------------------------------------------
    # Write path (append-only)
    # ------------------------------------------------------------------

    def append(self, report: LearningEvalReport, *, overwrite: bool = False) -> Path:
        run = report.run
        path = self.run_path(run.run_id)
        if path.exists() and not overwrite:
            raise HistoryError(
                f"run {run.run_id!r} already exists; refusing to overwrite"
            )
        document = {"kind": "learning_eval_run", "run": report.to_dict()}
        record_hash = _record_hash(document)
        document["record_hash"] = record_hash
        payload = json.dumps(
            document, indent=2, sort_keys=True, ensure_ascii=False, default=str
        )
        self._atomic_write(path, payload)
        return path

    def read(self, run_id: str) -> LearningEvalRun:
        """Load a run, re-verifying its integrity hash first."""
        path = self.run_path(run_id)
        if not path.exists():
            raise HistoryError(f"run {run_id!r} not found in history")
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("kind") != "learning_eval_run":
            raise HistoryError(f"record {run_id!r} is not a learning-eval run")
        stored_hash = document.pop("record_hash", None)
        if stored_hash != _record_hash(document):
            raise HistoryError(
                f"run {run_id!r} integrity hash mismatch; refusing to load"
            )
        return LearningEvalRun.from_dict(document["run"]["run"])

    def runs(self) -> list[LearningEvalRun]:
        """All stored runs in insertion (creation) order, verified."""
        result: list[LearningEvalRun] = []
        for path in sorted(self.root.glob("*.json")):
            if path.name == "index.json":
                continue
            result.append(self.read(path.stem))
        return result

    # ------------------------------------------------------------------
    # Integrity helpers
    # ------------------------------------------------------------------

    def verify(self) -> list[str]:
        """Integrity check over the whole store; returns failure messages."""
        failures: list[str] = []
        for path in sorted(self.root.glob("*.json")):
            if path.name == "index.json":
                continue
            try:
                self.read(path.stem)
            except HistoryError as exc:
                failures.append(f"{path.name}: {exc}")
        return failures

    @staticmethod
    def _atomic_write(path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)


__all__ = [
    "DEFAULT_HISTORY_DIR",
    "HistoryError",
    "LearningEvalHistory",
]
