"""Feature Store persistence layer.

JSON-file based storage for computed feature sets. Supports versioning,
snapshots, lookup, incremental updates, and historical feature history.
Never overwrites previous snapshots.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from predictron_engine.feature_store.models import (
    CompanyFeatureSet,
    FeatureSnapshot,
    FeatureStoreSnapshot,
)


class FeatureStore:
    """JSON-file storage for computed feature sets.

    Directory layout::

        root/
            companies/
                {company_id}.json          # latest CompanyFeatureSet
            snapshots/
                {snapshot_id}.json         # FeatureStoreSnapshot
            history/
                {company_id}/
                    {timestamp}.json       # historical feature sets
            manifest.json
    """

    def __init__(self, root_path: str | Path) -> None:
        self._root = Path(root_path)
        self._companies_dir = self._root / "companies"
        self._snapshots_dir = self._root / "snapshots"
        self._history_dir = self._root / "history"
        self._manifest_path = self._root / "manifest.json"

    def initialize(self) -> None:
        """Create directory structure if needed."""
        self._companies_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._history_dir.mkdir(parents=True, exist_ok=True)
        if not self._manifest_path.exists():
            self._write_manifest({
                "version": "1.0.0",
                "created_at": datetime.now(UTC).isoformat(),
                "company_count": 0,
                "snapshot_count": 0,
            })

    # ---- Company Feature Sets ----

    def save_company_features(self, feature_set: CompanyFeatureSet) -> Path:
        """Persist a CompanyFeatureSet to disk.

        Saves the latest version and appends to history.
        """
        latest_path = self._companies_dir / f"{feature_set.company_id}.json"
        latest_path.write_text(
            feature_set.model_dump_json(indent=2),
            encoding="utf-8",
        )
        history_dir = self._history_dir / feature_set.company_id
        history_dir.mkdir(parents=True, exist_ok=True)
        ts = feature_set.built_at.strftime("%Y%m%dT%H%M%SZ")
        hist_path = history_dir / f"{ts}.json"
        counter = 1
        while hist_path.exists():
            hist_path = history_dir / f"{ts}-{counter}.json"
            counter += 1
        hist_path.write_text(
            feature_set.model_dump_json(indent=2),
            encoding="utf-8",
        )
        self._update_manifest("company_count", 0)
        return latest_path

    def load_company_features(self, company_id: str) -> CompanyFeatureSet | None:
        """Load the latest feature set for a company."""
        path = self._companies_dir / f"{company_id}.json"
        if not path.exists():
            return None
        return CompanyFeatureSet.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def list_companies(self) -> list[str]:
        """Sorted list of company IDs with stored features."""
        return sorted(
            p.stem for p in self._companies_dir.glob("*.json")
        )

    def count_companies(self) -> int:
        return len(self.list_companies())

    # ---- Feature History ----

    def get_company_history(self, company_id: str) -> list[CompanyFeatureSet]:
        """Retrieve all historical feature sets for a company, oldest first."""
        history_dir = self._history_dir / company_id
        if not history_dir.exists():
            return []
        result: list[CompanyFeatureSet] = []
        for path in sorted(history_dir.glob("*.json")):
            fs = CompanyFeatureSet.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            result.append(fs)
        return result

    def get_feature_history(
        self, company_id: str, feature_id: str
    ) -> list[FeatureSnapshot]:
        """Track a single feature's values over time."""
        history = self.get_company_history(company_id)
        snapshots: list[FeatureSnapshot] = []
        for fs in history:
            snap = fs.features.get(feature_id)
            if snap is not None:
                snapshots.append(snap)
        return snapshots

    def history_count(self, company_id: str) -> int:
        """Number of historical snapshots for a company."""
        history_dir = self._history_dir / company_id
        if not history_dir.exists():
            return 0
        return len(list(history_dir.glob("*.json")))

    # ---- Store Snapshots ----

    def save_snapshot(self, snapshot: FeatureStoreSnapshot) -> Path:
        """Persist a full store snapshot."""
        path = self._snapshots_dir / f"{snapshot.snapshot_id}.json"
        path.write_text(
            snapshot.model_dump_json(indent=2),
            encoding="utf-8",
        )
        self._update_manifest("snapshot_count", 1)
        return path

    def load_snapshot(self, snapshot_id: str) -> FeatureStoreSnapshot | None:
        """Load a store snapshot by ID."""
        path = self._snapshots_dir / f"{snapshot_id}.json"
        if not path.exists():
            return None
        return FeatureStoreSnapshot.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def list_snapshots(self) -> list[str]:
        """Sorted list of snapshot IDs."""
        return sorted(
            p.stem for p in self._snapshots_dir.glob("*.json")
        )

    def count_snapshots(self) -> int:
        return len(self.list_snapshots())

    # ---- Feature Lookup ----

    def lookup_feature(
        self, company_id: str, feature_id: str
    ) -> FeatureSnapshot | None:
        """Look up a single feature for a company."""
        fs = self.load_company_features(company_id)
        if fs is None:
            return None
        return fs.features.get(feature_id)

    def lookup_features(
        self, company_id: str, feature_ids: list[str]
    ) -> dict[str, FeatureSnapshot | None]:
        """Look up multiple features for a company."""
        fs = self.load_company_features(company_id)
        if fs is None:
            return {fid: None for fid in feature_ids}
        return {fid: fs.features.get(fid) for fid in feature_ids}

    # ---- Incremental Updates ----

    def update_company_features(
        self, feature_set: CompanyFeatureSet
    ) -> Path:
        """Update (overwrite) a company's feature set.

        Previous versions are preserved in history.
        """
        return self.save_company_features(feature_set)

    # ---- Import / Export ----

    def export_all(self) -> dict[str, object]:
        """Export the entire store as a deterministic dict."""
        companies: dict[str, object] = {}
        for company_id in self.list_companies():
            fs = self.load_company_features(company_id)
            if fs is not None:
                companies[company_id] = fs.model_dump()
        return {
            "version": "1.0.0",
            "exported_at": datetime.now(UTC).isoformat(),
            "company_count": len(companies),
            "companies": companies,
        }

    def import_all(self, data: dict[str, object]) -> int:
        """Import companies from a dict. Returns count imported."""
        companies = data.get("companies", {})
        if not isinstance(companies, dict):
            return 0
        count = 0
        for company_id, fs_data in companies.items():
            if isinstance(fs_data, dict):
                fs = CompanyFeatureSet.model_validate(fs_data)
                self.save_company_features(fs)
                count += 1
        return count

    # ---- Manifest ----

    def get_manifest(self) -> dict[str, object]:
        if not self._manifest_path.exists():
            return {}
        data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        return dict(data) if isinstance(data, dict) else {}

    # ---- Internal ----

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
