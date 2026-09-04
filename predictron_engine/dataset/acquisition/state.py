"""Acquisition state persistence and checkpoint management.

Tracks what has been acquired, when, and from where.  Supports
resumable imports via checkpoints and prevents duplicate imports
through content-hash idempotency.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class AcquisitionRecord:
    """A single record created by an acquisition run."""

    record_id: str
    startup_name: str
    website: str
    source: str
    acquired_at: str = ""
    source_hash: str = ""


@dataclass
class AcquisitionBatch:
    """Metadata for one acquisition batch (one source + file run)."""

    batch_id: str = ""
    source_name: str = ""
    file_path: str = ""
    file_hash: str = ""
    source_version: str = ""
    imported_at: str = ""
    record_count: int = 0
    records: list[AcquisitionRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.batch_id:
            self.batch_id = str(uuid.uuid4())
        if not self.imported_at:
            self.imported_at = datetime.now(UTC).isoformat()


@dataclass
class CheckpointData:
    """Persisted checkpoint for resumable batch imports."""

    batch_id: str = ""
    source_name: str = ""
    file_path: str = ""
    file_hash: str = ""
    total_records: int = 0
    processed_records: int = 0
    imported_records: int = 0
    failed_records: int = 0
    skipped_records: int = 0
    started_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = datetime.now(UTC).isoformat()
        self.updated_at = datetime.now(UTC).isoformat()

    def advance(
        self,
        *,
        imported: int = 0,
        failed: int = 0,
        skipped: int = 0,
    ) -> None:
        """Advance the checkpoint by the given counts."""
        self.processed_records += imported + failed + skipped
        self.imported_records += imported
        self.failed_records += failed
        self.skipped_records += skipped
        self.updated_at = datetime.now(UTC).isoformat()

    @property
    def is_complete(self) -> bool:
        """Return True if all records have been processed."""
        return self.processed_records >= self.total_records and self.total_records > 0


class AcquisitionStateManager:
    """Manages acquisition state on disk.

    Directory layout::

        dataset_store/acquisition/
            history/
                {batch_id}.json     # completed batch records
            checkpoints/
                {batch_id}.json     # in-progress batch checkpoints
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._history_dir = self._root / "acquisition" / "history"
        self._checkpoint_dir = self._root / "acquisition" / "checkpoints"

    def initialize(self) -> None:
        """Create directories if they do not exist."""
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---- History ----

    def record_batch(self, batch: AcquisitionBatch) -> None:
        """Persist a completed acquisition batch to history."""
        self._history_dir.mkdir(parents=True, exist_ok=True)
        path = self._history_dir / f"{batch.batch_id}.json"
        data = asdict(batch)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def list_batches(self) -> list[AcquisitionBatch]:
        """List all completed acquisition batches."""
        batches: list[AcquisitionBatch] = []
        if not self._history_dir.exists():
            return batches
        for path in sorted(self._history_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                records = [
                    AcquisitionRecord(**r) for r in data.pop("records", [])
                ]
                batch = AcquisitionBatch(**data)
                batch.records = records
                batches.append(batch)
            except (json.JSONDecodeError, TypeError):
                continue
        return batches

    def get_last_batch_for_source(self, source_name: str) -> AcquisitionBatch | None:
        """Return the most recent batch for a given source."""
        batches = self.list_batches()
        for batch in reversed(batches):
            if batch.source_name == source_name:
                return batch
        return None

    def has_file_been_imported(self, file_hash: str) -> bool:
        """Check whether a file with this content hash was already imported."""
        for batch in self.list_batches():
            if batch.file_hash == file_hash:
                return True
        return False

    # ---- Checkpoints ----

    def save_checkpoint(self, checkpoint: CheckpointData) -> None:
        """Persist a checkpoint for a resumable import."""
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_dir / f"{checkpoint.batch_id}.json"
        path.write_text(
            json.dumps(asdict(checkpoint), indent=2, default=str),
            encoding="utf-8",
        )

    def load_checkpoint(self, batch_id: str) -> CheckpointData | None:
        """Load a checkpoint by batch_id."""
        path = self._checkpoint_dir / f"{batch_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return CheckpointData(**data)
        except (json.JSONDecodeError, TypeError):
            return None

    def list_checkpoints(self) -> list[CheckpointData]:
        """List all active (incomplete) checkpoints."""
        checkpoints: list[CheckpointData] = []
        if not self._checkpoint_dir.exists():
            return checkpoints
        for path in sorted(self._checkpoint_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                cp = CheckpointData(**data)
                if not cp.is_complete:
                    checkpoints.append(cp)
            except (json.JSONDecodeError, TypeError):
                continue
        return checkpoints

    def clear_checkpoint(self, batch_id: str) -> None:
        """Remove a completed checkpoint."""
        path = self._checkpoint_dir / f"{batch_id}.json"
        if path.exists():
            path.unlink()

    def find_resume_checkpoint(
        self, source_name: str, file_hash: str
    ) -> CheckpointData | None:
        """Find an existing checkpoint that can be resumed."""
        for cp in self.list_checkpoints():
            if cp.source_name == source_name and cp.file_hash == file_hash:
                return cp
        return None


def compute_file_hash(file_path: Path | str) -> str:
    """Compute SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_batch_id() -> str:
    """Generate a unique batch identifier."""
    ts = int(time.time())
    short = uuid.uuid4().hex[:8]
    return f"batch-{ts}-{short}"
