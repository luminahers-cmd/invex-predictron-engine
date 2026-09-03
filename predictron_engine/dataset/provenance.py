"""Field-level provenance tracking (Part D).

Tracks the source and retrieval date of every imported field so the
dataset remains auditable.  Provenance is stored per record in a
structured mapping of field name -> (source, retrieval_date).

Provenance data is immutable once recorded and never fabricated.  A
field that was not provided by any source simply has no provenance
entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from predictron_engine.dataset.imports import RawImportRecord
from predictron_engine.dataset.models import DatasetRecord


@dataclass
class FieldProvenance:
    """Provenance for a single field."""

    field_name: str
    source: str
    retrieval_date: datetime

    def to_dict(self) -> dict[str, str | datetime]:
        return {
            "field": self.field_name,
            "source": self.source,
            "retrieval_date": self.retrieval_date,
        }


class ProvenanceTracker:
    """Tracks and attaches provenance for imported records."""

    def __init__(self, retrieval_date: datetime | None = None) -> None:
        self._retrieval_date = retrieval_date or datetime.now(UTC)

    @property
    def retrieval_date(self) -> datetime:
        """The retrieval date recorded for this import run."""
        return self._retrieval_date

    def build_provenance(
        self,
        raw: RawImportRecord,
        field_names: list[str],
    ) -> dict[str, dict[str, str | datetime]]:
        """Build a provenance map for non-empty fields from a raw record.

        Each field that has a non-empty value in the raw record gets a
        provenance entry attributing it to the record's source.  Empty
        fields are excluded.

        Parameters
        ----------
        raw :
            The raw import record.
        field_names :
            The names of the raw record fields to track.

        Returns
        -------
        Mapping of field name -> {source, retrieval_date}.
        """
        provenance: dict[str, dict[str, str | datetime]] = {}

        for field_name in field_names:
            value = getattr(raw, field_name, None)
            if not _is_present(value):
                continue
            provenance[field_name] = {
                "source": raw.source,
                "retrieval_date": self._retrieval_date,
            }

        # Track fields nested under prediction/outcome/metadata dicts.
        for key, value in raw.prediction_data.items():
            if _is_present(value):
                provenance[f"prediction.{key}"] = {
                    "source": raw.source,
                    "retrieval_date": self._retrieval_date,
                }
        for key, value in raw.outcome_data.items():
            if _is_present(value):
                provenance[f"outcome.{key}"] = {
                    "source": raw.source,
                    "retrieval_date": self._retrieval_date,
                }
        for key, value in raw.metadata.items():
            if _is_present(value):
                provenance[f"metadata.{key}"] = {
                    "source": raw.source,
                    "retrieval_date": self._retrieval_date,
                }

        return provenance

    def attach_to_record(
        self,
        record: DatasetRecord,
        provenance: dict[str, dict[str, str | datetime]],
    ) -> DatasetRecord:
        """Attach provenance to a DatasetRecord's analysis_metadata.

        The original record is not mutated; a new instance is returned.
        Provenance is stored under ``analysis_metadata["provenance"]``.
        """
        metadata = dict(record.analysis_metadata)
        metadata["provenance"] = provenance
        return record.model_copy(update={"analysis_metadata": metadata})

    @staticmethod
    def extract_from_record(
        record: DatasetRecord,
    ) -> dict[str, dict[str, str | datetime]]:
        """Extract the provenance map stored on a record, if any."""
        provenance = record.analysis_metadata.get("provenance", {})
        if isinstance(provenance, dict):
            return dict(provenance)
        return {}


def _is_present(value: object) -> bool:
    """Determine whether a value counts as present for provenance."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, list | dict):
        return len(value) > 0
    return True
