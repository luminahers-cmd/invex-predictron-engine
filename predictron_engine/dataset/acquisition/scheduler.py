"""Acquisition scheduling utilities.

Provides data structures and logic for managing scheduled imports.
Designed to be driven by external schedulers (cron, GitHub Actions)
via the CLI, rather than running its own event loop.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class ImportSchedule:
    """Configuration for a scheduled import."""

    source_name: str
    frequency: str = "weekly"
    enabled: bool = True
    last_run: str = ""
    next_run: str = ""
    priority: int = 0
    config: dict[str, object] = field(default_factory=dict)


class AcquisitionScheduler:
    """Manages scheduled import configurations.

    Stores schedules as JSON.  Designed to be queried by external
    schedulers (cron, CI pipelines) that call the CLI commands.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._schedules_path = self._root / "acquisition" / "schedules.json"

    def initialize(self) -> None:
        """Create the schedules file if it does not exist."""
        self._schedules_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._schedules_path.exists():
            self._write_schedules([])

    def load_schedules(self) -> list[ImportSchedule]:
        """Load all configured schedules."""
        if not self._schedules_path.exists():
            return []
        try:
            data = json.loads(self._schedules_path.read_text(encoding="utf-8"))
            return [ImportSchedule(**s) for s in data]
        except (json.JSONDecodeError, TypeError):
            return []

    def _write_schedules(self, schedules: list[ImportSchedule]) -> None:
        """Persist schedules to disk."""
        self._schedules_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(s) for s in schedules]
        self._schedules_path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

    def add_schedule(self, schedule: ImportSchedule) -> None:
        """Add or update a schedule for a source."""
        schedules = self.load_schedules()
        existing_idx = None
        for i, s in enumerate(schedules):
            if s.source_name == schedule.source_name:
                existing_idx = i
                break
        if existing_idx is not None:
            schedules[existing_idx] = schedule
        else:
            schedules.append(schedule)
        self._write_schedules(schedules)

    def remove_schedule(self, source_name: str) -> bool:
        """Remove a schedule.  Returns True if found and removed."""
        schedules = self.load_schedules()
        original_len = len(schedules)
        schedules = [s for s in schedules if s.source_name != source_name]
        if len(schedules) < original_len:
            self._write_schedules(schedules)
            return True
        return False

    def get_schedule(self, source_name: str) -> ImportSchedule | None:
        """Get the schedule for a specific source."""
        for s in self.load_schedules():
            if s.source_name == source_name:
                return s
        return None

    def get_due_imports(self) -> list[ImportSchedule]:
        """Return schedules that are due for execution."""
        now = datetime.now(UTC)
        due: list[ImportSchedule] = []
        for schedule in self.load_schedules():
            if not schedule.enabled:
                continue
            if not schedule.next_run:
                due.append(schedule)
                continue
            try:
                next_dt = datetime.fromisoformat(schedule.next_run)
                if next_dt <= now:
                    due.append(schedule)
            except (ValueError, TypeError):
                due.append(schedule)
        return sorted(due, key=lambda s: -s.priority)

    def mark_run(self, source_name: str) -> None:
        """Mark a source as having been run.  Updates last_run and next_run."""
        schedules = self.load_schedules()
        now = datetime.now(UTC)
        for schedule in schedules:
            if schedule.source_name == source_name:
                schedule.last_run = now.isoformat()
                schedule.next_run = _compute_next_run(
                    schedule.frequency, now
                )
                break
        self._write_schedules(schedules)


def _compute_next_run(frequency: str, from_dt: datetime) -> str:
    """Compute the next run time based on frequency."""
    from datetime import timedelta

    deltas = {
        "hourly": timedelta(hours=1),
        "daily": timedelta(days=1),
        "weekly": timedelta(weeks=1),
        "monthly": timedelta(days=30),
        "quarterly": timedelta(days=90),
    }
    delta = deltas.get(frequency, timedelta(weeks=1))
    return (from_dt + delta).isoformat()
