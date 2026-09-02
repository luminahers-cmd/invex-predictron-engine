"""JSON-based dataset storage backend.

Provides persistence for DatasetRecord, OutcomeRecord, and
PredictionEvaluation instances using local JSON files.  The store
is designed for offline analysis and does not require any external
services.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from predictron_engine.dataset.analysis import AnalysisRun
from predictron_engine.dataset.evaluation import PredictionEvaluation
from predictron_engine.dataset.models import DatasetRecord
from predictron_engine.dataset.outcomes import OutcomeRecord


class DatasetStore:
    """Local JSON-file storage for dataset records.

    Stores records as individual JSON files organized by record_id
    within a configurable root directory.  Each record type has its
    own subdirectory.

    Directory layout::

        root/
            records/
                {record_id}.json
            outcomes/
                {outcome_id}.json
            evaluations/
                {evaluation_id}.json
            runs/
                {run_id}.json
            manifest.json
    """

    def __init__(self, root_path: str | Path) -> None:
        self._root = Path(root_path)
        self._records_dir = self._root / "records"
        self._outcomes_dir = self._root / "outcomes"
        self._evaluations_dir = self._root / "evaluations"
        self._runs_dir = self._root / "runs"
        self._manifest_path = self._root / "manifest.json"

    def initialize(self) -> None:
        """Create the directory structure if it does not exist."""
        self._records_dir.mkdir(parents=True, exist_ok=True)
        self._outcomes_dir.mkdir(parents=True, exist_ok=True)
        self._evaluations_dir.mkdir(parents=True, exist_ok=True)
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        if not self._manifest_path.exists():
            self._write_manifest(
                {
                    "version": "1.0.0",
                    "created_at": datetime.now(UTC).isoformat(),
                    "record_count": 0,
                    "outcome_count": 0,
                    "evaluation_count": 0,
                    "run_count": 0,
                }
            )

    # ---- Dataset records ----

    def save_record(self, record: DatasetRecord) -> Path:
        """Persist a DatasetRecord to disk."""
        path = self._records_dir / f"{record.record_id}.json"
        path.write_text(
            record.model_dump_json(indent=2),
            encoding="utf-8",
        )
        self._update_manifest("record_count", 1)
        return path

    def load_record(self, record_id: str) -> DatasetRecord | None:
        """Load a DatasetRecord by its ID."""
        path = self._records_dir / f"{record_id}.json"
        if not path.exists():
            return None
        return DatasetRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_records(self) -> list[str]:
        """List all stored record IDs."""
        return [
            p.stem
            for p in sorted(self._records_dir.glob("*.json"))
        ]

    def find_records_by_startup(self, startup_name: str) -> list[DatasetRecord]:
        """Find all records matching a startup name."""
        results: list[DatasetRecord] = []
        for record_id in self.list_records():
            record = self.load_record(record_id)
            if record and record.startup_name == startup_name:
                results.append(record)
        return results

    def count_records(self) -> int:
        return len(self.list_records())

    def find_distinct_startups(self) -> list[str]:
        """Return the sorted set of distinct startup names in the store."""
        names: set[str] = set()
        for record_id in self.list_records():
            record = self.load_record(record_id)
            if record and record.startup_name:
                names.add(record.startup_name)
        return sorted(names)

    # ---- Outcome records ----

    def save_outcome(self, outcome: OutcomeRecord) -> Path:
        """Persist an OutcomeRecord to disk."""
        path = self._outcomes_dir / f"{outcome.outcome_id}.json"
        path.write_text(
            outcome.model_dump_json(indent=2),
            encoding="utf-8",
        )
        self._update_manifest("outcome_count", 1)
        return path

    def load_outcome(self, outcome_id: str) -> OutcomeRecord | None:
        path = self._outcomes_dir / f"{outcome_id}.json"
        if not path.exists():
            return None
        return OutcomeRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def find_outcome_by_record(self, record_id: str) -> OutcomeRecord | None:
        """Find the outcome linked to a specific dataset record."""
        for outcome_id in self._list_outcome_ids():
            outcome = self.load_outcome(outcome_id)
            if outcome and outcome.record_id == record_id:
                return outcome
        return None

    def list_outcomes(self) -> list[str]:
        return self._list_outcome_ids()

    def count_outcomes(self) -> int:
        return len(self._list_outcome_ids())

    # ---- Evaluations ----

    def save_evaluation(self, evaluation: PredictionEvaluation) -> Path:
        path = self._evaluations_dir / f"{evaluation.evaluation_id}.json"
        path.write_text(
            evaluation.model_dump_json(indent=2),
            encoding="utf-8",
        )
        self._update_manifest("evaluation_count", 1)
        return path

    def load_evaluation(self, evaluation_id: str) -> PredictionEvaluation | None:
        path = self._evaluations_dir / f"{evaluation_id}.json"
        if not path.exists():
            return None
        return PredictionEvaluation.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def find_evaluation_by_record(self, record_id: str) -> PredictionEvaluation | None:
        for eval_id in self._list_evaluation_ids():
            evaluation = self.load_evaluation(eval_id)
            if evaluation and evaluation.record_id == record_id:
                return evaluation
        return None

    def list_evaluations(self) -> list[str]:
        return self._list_evaluation_ids()

    def count_evaluations(self) -> int:
        return len(self._list_evaluation_ids())

    # ---- Analysis runs ----

    def save_run(self, run: AnalysisRun) -> Path:
        """Persist an AnalysisRun to disk."""
        path = self._runs_dir / f"{run.run_id}.json"
        path.write_text(
            run.model_dump_json(indent=2),
            encoding="utf-8",
        )
        self._update_manifest("run_count", 1)
        return path

    def load_run(self, run_id: str) -> AnalysisRun | None:
        """Load an AnalysisRun by its ID."""
        path = self._runs_dir / f"{run_id}.json"
        if not path.exists():
            return None
        return AnalysisRun.model_validate_json(path.read_text(encoding="utf-8"))

    def find_runs_by_record(self, record_id: str) -> list[AnalysisRun]:
        """Find all analysis runs for a specific dataset record."""
        runs: list[AnalysisRun] = []
        for run_id in self._list_run_ids():
            run = self.load_run(run_id)
            if run and run.record_id == record_id:
                runs.append(run)
        return runs

    def list_runs(self) -> list[str]:
        """List all stored analysis run IDs."""
        return self._list_run_ids()

    def count_runs(self) -> int:
        """Count stored analysis runs."""
        return len(self._list_run_ids())

    # ---- Manifest ----

    def get_manifest(self) -> dict[str, object]:
        """Read the current manifest."""
        if not self._manifest_path.exists():
            return {}
        data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return dict(data)
        return {}

    # ---- Internal helpers ----

    def _list_outcome_ids(self) -> list[str]:
        return sorted(p.stem for p in self._outcomes_dir.glob("*.json"))

    def _list_evaluation_ids(self) -> list[str]:
        return sorted(p.stem for p in self._evaluations_dir.glob("*.json"))

    def _list_run_ids(self) -> list[str]:
        return sorted(p.stem for p in self._runs_dir.glob("*.json"))

    def _write_manifest(self, data: dict[str, object]) -> None:
        self._manifest_path.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )

    def _update_manifest(self, key: str, delta: int) -> None:
        manifest = self.get_manifest()
        current = manifest.get(key, 0)
        if isinstance(current, int):
            manifest[key] = current + delta
        else:
            manifest[key] = delta
        self._write_manifest(manifest)
