"""Phase 6 — append-only monitor snapshot history (file-backed).

Mirrors the file-backed pattern of ``benchmarks.ground_truth_eval.history``:
one JSON file per snapshot under ``data/monitor_history/<scope>/<period_kind>/``,
named by the deterministic ``snapshot_id``.  No database table is introduced
(Phase 6 is additive; the ledger stays the single source of truth and the
snapshot files are derived, immutable artifacts).

Layout
------
``data/monitor_history/<scope>/<period_kind>/<anchor_date>.json``

* ``<anchor_date>`` is ``YYYY-MM-DD`` so rotating a ``daily`` history keeps
  exactly one snapshot per day (idempotent), while ``weekly`` / ``monthly``
  cadences partition the same directory.
* A record is a full :class:`MonitorSnapshot` JSON document; the integrity
  ``content_hash`` lets consumers verify nothing was mutated after write.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from urllib.parse import quote

from predictron_engine.monitoring.models import (
    MonitorPeriodKind,
    MonitorSnapshot,
    MonitorSnapshotSummary,
)

_SCOPE_SAFE = "_-.a-zA-Z0-9"


class MonitorHistory:
    """Append-only, file-backed repository of monitor snapshots.

    Parameters
    ----------
    root:
        Base directory for the snapshot store.

    Notes
    -----
    A scope label may contain characters that are invalid in directory names
    on some platforms (Windows rejects ``:`` in ``user:<id>``).  Scope labels
    are therefore percent-encoded deterministically when mapped to filesystem
    paths; the canonical scope (not the encoding) is always what is stored in
    the snapshot and summary documents.
    """

    def __init__(self, root: str | Path = "data/monitor_history") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _period_dir(self, scope: str, period_kind: str) -> Path:
        safe_scope = quote(scope, safe=_SCOPE_SAFE)
        directory = self.root / safe_scope / period_kind
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def snapshot_file_path(
        self, scope: str, period_kind: MonitorPeriodKind, anchor_date: date
    ) -> Path:
        snapshot_id = _snapshot_id_for(scope, period_kind, anchor_date)
        return self._period_dir(scope, period_kind.value) / f"{snapshot_id}.json"

    def record_snapshot(
        self,
        snapshot: MonitorSnapshot,
        *,
        keep: int | None = None,
    ) -> Path:
        """Write one snapshot and optionally rotate the history file count.

        ``keep`` bounds the number of retained snapshot files per
        ``(scope, period_kind)``; when ``None`` nothing is deleted (the
        append-only default).
        """
        path = self.snapshot_file_path(
            snapshot.scope, snapshot.period_kind, snapshot.anchor_date
        )
        document = {
            "summary": MonitorSnapshotSummary(
                snapshot_id=snapshot.snapshot_id,
                scope=snapshot.scope,
                period_kind=snapshot.period_kind,
                anchor_date=snapshot.anchor_date,
                engine_version=snapshot.engine_version,
                recorded_at=snapshot.recorded_at,
                content_hash=snapshot.content_hash,
            ).to_dict(),
            "snapshot": json.loads(snapshot.model_dump_json()),
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"))
        if keep is not None:
            self._rotate(snapshot.scope, snapshot.period_kind, keep)
        return path

    def snapshot_files(self, scope: str, period_kind: str) -> list[Path]:
        """Snapshot files in a period directory, ordered by name."""
        directory = self._period_dir(scope, period_kind)
        return sorted(directory.glob("*.json"))

    def load_snapshot(self, path: str | Path) -> MonitorSnapshot:
        """Deserialize a snapshot document (verifying its content hash)."""
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
        snapshot = MonitorSnapshot.model_validate(document["snapshot"])
        if not snapshot.verify():
            raise ValueError(f"integrity failure in snapshot file: {path}")
        return snapshot

    def snapshots(self, scope: str, period_kind: str) -> list[MonitorSnapshot]:
        """All snapshots for a period, ordered by ``anchor_date``."""
        loaded = [self.load_snapshot(path) for path in self.snapshot_files(scope, period_kind)]
        return sorted(loaded, key=lambda entry: entry.anchor_date)

    def latest_snapshot(
        self, scope: str, period_kind: str
    ) -> MonitorSnapshot | None:
        """Most recent snapshot for a period (by ``anchor_date``) or ``None``."""
        snapshots = self.snapshots(scope, period_kind)
        if not snapshots:
            return None
        return snapshots[-1]

    def summarize(self, scope: str, period_kind: str) -> list[MonitorSnapshotSummary]:
        """Cheap index of a period without loading analytic bodies."""
        summaries: list[MonitorSnapshotSummary] = []
        for path in self.snapshot_files(scope, period_kind):
            with open(path, encoding="utf-8") as handle:
                document = json.load(handle)
            summaries.append(MonitorSnapshotSummary.model_validate(document["summary"]))
        return sorted(summaries, key=lambda entry: entry.anchor_date)

    def _rotate(self, scope: str, period_kind: MonitorPeriodKind, keep: int) -> None:
        """Drop the oldest snapshot files once more than ``keep`` exist."""
        if keep < 1:
            raise ValueError("keep must be >= 1")
        files = self.snapshot_files(scope, period_kind.value)
        while len(files) > keep:
            path = files.pop(0)
            path.unlink()


def _snapshot_id_for(
    scope: str, period_kind: MonitorPeriodKind, anchor_date: date
) -> str:
    import hashlib

    material = "|".join((scope, period_kind.value, anchor_date.isoformat()))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


__all__ = ["MonitorHistory"]
