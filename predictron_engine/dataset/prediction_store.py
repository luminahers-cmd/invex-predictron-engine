"""Append-only frozen-prediction store (Phase 4 prediction validation).

Every time-scoped prediction is persisted as its own JSON file under a
prediction root.  The store is strictly **append-only**, mirroring the
``BenchmarkHistory`` guarantees of the ground-truth evaluation platform:

* a prediction may never be overwritten — appending an existing
  ``prediction_id`` raises, and writes go through an atomic temp-file
  replacement;
* each stored record re-verifies its ``prediction_hash`` on both write and
  read, so a corrupted/unmodified prediction is detected, not silently
  loaded;
* the on-disk payload is canonical (sorted keys) independently of the
  model's in-memory field order.

Wall-clock timestamps are informational only; hashes never depend on them.
Frozen predictions are the deterministic input to the cohort evaluation
pipeline (:mod:`benchmarks.cohort.execute`).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from predictron_engine.dataset.prediction import TimeScopedPrediction

# A prediction id is used directly as a filename, so keep it strictly safe.
_PREDICTION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

DEFAULT_PREDICTION_ROOT = (
    Path(__file__).resolve().parents[2] / "data" / "frozen_predictions"
)

_PREDICTION_ID_SEPARATOR = "\x00"


class PredictionStoreError(RuntimeError):
    """Raised for any prediction-store violation (append, integrity, IO)."""


@dataclass(frozen=True)
class PredictionIndexEntry:
    """Lightweight header for a stored prediction (index/listing purposes)."""

    prediction_id: str
    company_id: str
    company_name: str
    analysis_timestamp: datetime
    engine_version: str
    prediction_hash: str
    evaluation_horizon_days: int | None
    decision: str
    confidence: float
    recorded_at: datetime

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["analysis_timestamp"] = self.analysis_timestamp.isoformat()
        data["recorded_at"] = self.recorded_at.isoformat()
        return data


def stable_prediction_id(*, company_id: str, analysis_timestamp: datetime) -> str:
    """Deterministic prediction identity from (company, analysis timestamp).

    Re-running the same time-scoped analysis for the same company always
    yields the same ``prediction_id``, so cohort execution can freeze a
    prediction idempotently.
    """
    if analysis_timestamp.tzinfo is None:
        raise PredictionStoreError(
            "analysis_timestamp must be timezone-aware (UTC) for a stable prediction id"
        )
    material = _PREDICTION_ID_SEPARATOR.join(
        [company_id, analysis_timestamp.isoformat()]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def prediction_hash(prediction: TimeScopedPrediction) -> str:
    """Deterministic canonical fingerprint of a time-scoped prediction."""
    payload = json.dumps(
        prediction.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PredictionStore:
    """Append-only, hash-verified store of frozen predictions."""

    def __init__(self, root: Path | str = DEFAULT_PREDICTION_ROOT) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def prediction_path(self, prediction_id: str) -> Path:
        if not _PREDICTION_ID_PATTERN.match(prediction_id):
            raise PredictionStoreError(
                f"prediction_id {prediction_id!r} is not filename-safe"
            )
        return self.root / f"{prediction_id}.json"

    def index_path(self) -> Path:
        return self.root / "index.json"

    # ------------------------------------------------------------------
    # Write path (append-only)
    # ------------------------------------------------------------------

    def save(
        self,
        prediction: TimeScopedPrediction,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Persist a prediction. Refuses to overwrite by default.

        The stored record integrity-verifies its ``prediction_hash`` before
        the write completes, so a corrupted artefact is never persisted
        silently.
        """
        path = self.prediction_path(prediction.prediction_id)
        if path.exists() and not overwrite:
            raise PredictionStoreError(
                f"prediction {prediction.prediction_id!r} already exists; "
                "refusing to overwrite"
            )

        digest = prediction_hash(prediction)
        document: dict[str, Any] = {
            "kind": "frozen_prediction",
            "prediction": prediction.model_dump(mode="json"),
            "prediction_hash": digest,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            document, indent=2, sort_keys=True, ensure_ascii=False, default=str
        )
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload + "\n", encoding="utf-8")
        tmp.replace(path)
        self._update_index(prediction, prediction_hash=digest)
        return path

    def save_if_absent(self, prediction: TimeScopedPrediction) -> tuple[Path, bool]:
        """Persist a prediction only when its id is not already stored.

        Returns ``(path, persisted)`` where ``persisted`` is ``False`` when
        the prediction id already exists — enabling idempotent cohort
        re-execution without mutating prior artefacts.
        """
        if self.has_prediction(prediction.prediction_id):
            return self.prediction_path(prediction.prediction_id), False
        path = self.save(prediction)
        return path, True

    def _update_index(
        self, prediction: TimeScopedPrediction, *, prediction_hash: str
    ) -> None:
        existing = {entry.prediction_id for entry in self.list_summaries()}
        if prediction.prediction_id in existing:
            return
        entry = PredictionIndexEntry(
            prediction_id=prediction.prediction_id,
            company_id=prediction.company_id,
            company_name=prediction.company_name,
            analysis_timestamp=prediction.analysis_timestamp,
            engine_version=prediction.engine_version,
            prediction_hash=prediction_hash,
            evaluation_horizon_days=prediction.evaluation_horizon_days,
            decision=prediction.prediction.decision.value,
            confidence=float(prediction.prediction.confidence),
            recorded_at=prediction.recorded_at,
        )
        entries = self.list_summaries() + [entry]
        entries.sort(key=lambda e: (e.analysis_timestamp, e.prediction_id))
        payload = json.dumps(
            [e.to_dict() for e in entries],
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

    def load(self, prediction_id: str) -> TimeScopedPrediction:
        """Load and integrity-verify a stored prediction."""
        path = self.prediction_path(prediction_id)
        if not path.exists():
            raise PredictionStoreError(
                f"prediction {prediction_id!r} not found in {self.root}"
            )
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PredictionStoreError(
                f"corrupt prediction file {path}: {exc}"
            ) from exc
        if not isinstance(document, dict) or document.get("kind") != "frozen_prediction":
            raise PredictionStoreError(
                f"invalid prediction file {path}: missing 'frozen_prediction' kind"
            )
        prediction_data = document.get("prediction")
        if not isinstance(prediction_data, dict):
            raise PredictionStoreError(
                f"invalid prediction file {path}: missing 'prediction' section"
            )
        try:
            prediction = TimeScopedPrediction.model_validate(prediction_data)
        except Exception as exc:  # noqa: BLE001 - surface malformed records
            raise PredictionStoreError(
                f"prediction file {path} failed validation: {exc}"
            ) from exc
        stored_hash = document.get("prediction_hash")
        computed = prediction_hash(prediction)
        if stored_hash != computed:
            raise PredictionStoreError(
                f"prediction {prediction_id!r} integrity check failed: "
                f"stored {stored_hash!r}, computed {computed!r}"
            )
        return prediction

    def has_prediction(self, prediction_id: str) -> bool:
        return self.prediction_path(prediction_id).exists()

    # ------------------------------------------------------------------
    # Listing / index
    # ------------------------------------------------------------------

    def list_summaries(self) -> list[PredictionIndexEntry]:
        """Return stored predictions sorted by (analysis_timestamp, id)."""
        index = self.index_path()
        if not index.exists():
            return []
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        entries: list[PredictionIndexEntry] = []
        for row in data:
            try:
                d = dict(row)
                d["analysis_timestamp"] = datetime.fromisoformat(
                    str(d["analysis_timestamp"])
                )
                d["recorded_at"] = datetime.fromisoformat(str(d["recorded_at"]))
                entries.append(PredictionIndexEntry(**d))
            except (KeyError, TypeError, ValueError):
                continue
        entries.sort(key=lambda e: (e.analysis_timestamp, e.prediction_id))
        return entries

    def list_prediction_ids(self, *, company_id: str | None = None) -> list[str]:
        ids = [
            entry.prediction_id
            for entry in self.list_summaries()
            if company_id is None or entry.company_id == company_id
        ]
        return ids

    @property
    def count(self) -> int:
        return len(self.list_summaries())

    def verify_integrity(self) -> dict[str, str]:
        """Re-verify every stored prediction's hash.

        Returns a mapping of ``prediction_id -> 'ok'``; raises on the first
        corrupted prediction so callers can surface the failure loudly.
        """
        status: dict[str, str] = {}
        for prediction_id in self.list_prediction_ids():
            self.load(prediction_id)
            status[prediction_id] = "ok"
        return status


__all__ = [
    "DEFAULT_PREDICTION_ROOT",
    "PredictionIndexEntry",
    "PredictionStore",
    "PredictionStoreError",
    "prediction_hash",
    "stable_prediction_id",
]
