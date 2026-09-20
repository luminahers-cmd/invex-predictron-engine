"""Cohort manifest schema and loading (Milestone V1.4).

A *cohort manifest* is the atomic input of the cohort builder: a versioned,
deterministic JSON document describing the real analysis-time inputs and the
best-known verified outcomes for every company in a ground-truth cohort.

Contract guarantees:

* ``analysis_timestamp`` must be timezone-aware (UTC) so the look-ahead guard
  is well defined.
* ``verified_outcome`` entries are stored verbatim; ``verified`` (a
  human-verified flag) never fabricates data — it only decides whether the
  provenance marks the outcome as real ground truth (``is_example=False``).

The schema deliberately mirrors :class:`~benchmarks.ground_truth_eval.models.GoldenEntry`
so a manifest record maps 1:1 onto a golden dataset entry.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from benchmarks.ground_truth.schema import OutcomeEvent, StartupStatus

MANIFEST_SCHEMA_VERSION = 1


class ManifestOutcome(BaseModel):
    """Best-known outcome for one company, with verification status."""

    status: StartupStatus = Field(..., description="Verified startup status")
    verification_date: date = Field(..., description="Date the outcome was last verified/recorded")
    outcome_events: list[OutcomeEvent] = Field(default_factory=list)
    exit_value_usd: float | None = Field(default=None, ge=0.0)
    sources: list[str] = Field(default_factory=list)
    verified: bool = Field(
        default=False,
        description=(
            "True when the outcome was independently verified; controls whether "
            "the built entry is marked as real ground truth (not is_example)"
        ),
    )
    verified_by: str | None = Field(default=None)
    notes: str = Field(default="")


class ManifestSourceRecord(BaseModel):
    """One company in the cohort: inputs, time anchor, and best-known outcome."""

    company_id: str = Field(..., description="Stable unique company identifier")
    company_name: str = Field(..., description="Display name as submitted")
    description: str = Field(..., description="Analysis-time company description")
    website: str | None = Field(default=None, description="Company website URL (if any)")
    pitch_deck_url: str | None = Field(default=None)
    founder_linkedin_urls: list[str] = Field(default_factory=list)
    sector: str | None = Field(default=None)
    country: str | None = Field(default=None)
    stage: str | None = Field(default=None)
    evidence_corpus: str = Field(
        ..., description="Committed offline evidence corpus reference (bare name or path)"
    )
    analysis_timestamp: datetime = Field(
        ..., description="UTC timestamp the analysis is pinned to (look-ahead guard anchor)"
    )
    evaluation_horizon_days: int | None = Field(
        default=None, ge=0, description="Overrides the manifest-level horizon"
    )
    outcome: ManifestOutcome | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CohortManifest(BaseModel):
    """Versioned cohort manifest document."""

    schema_version: int = Field(default=MANIFEST_SCHEMA_VERSION)
    dataset_name: str = Field(..., description="Golden dataset name to produce")
    benchmark_version: str = Field(..., description="Dataset version string (e.g. '1.0')")
    created_at: datetime = Field(
        ..., description="Fixed build timestamp — keeps regeneration deterministic"
    )
    evaluation_horizon_days: int | None = Field(
        default=None, ge=0, description="Minimum days between analysis and outcome observation"
    )
    source_records: list[ManifestSourceRecord] = Field(default_factory=list)
    notes: str = Field(default="")

    @model_validator(mode="after")
    def _validate_determinism(self) -> CohortManifest:
        if self.created_at.tzinfo is None:
            raise ValueError("manifest created_at must be timezone-aware (UTC)")
        seen: set[str] = set()
        for record in self.source_records:
            if record.company_id in seen:
                raise ValueError(f"duplicate company_id {record.company_id!r}")
            seen.add(record.company_id)
        return self


class ManifestError(ValueError):
    """Raised when a cohort manifest cannot be loaded or is invalid."""


def manifest_bytes(manifest: CohortManifest) -> bytes:
    """Deterministic UTF-8 bytes for a manifest (sorted keys)."""
    return json.dumps(
        _sort_keys(manifest.model_dump(mode="json")),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def manifest_hash(manifest: CohortManifest) -> str:
    """Canonical sha256 fingerprint of a cohort manifest."""
    import hashlib

    return hashlib.sha256(manifest_bytes(manifest)).hexdigest()


def dump_manifest(manifest: CohortManifest, path: Path | str, *, overwrite: bool = False) -> Path:
    """Write a manifest to disk (deterministic, atomic, refuse-to-clobber)."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise ManifestError(f"manifest already exists: {path} (pass overwrite=True to replace)")
    payload = json.dumps(
        _sort_keys(manifest.model_dump(mode="json")), indent=2, sort_keys=True
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return path


def parse_manifest(payload: dict[str, Any]) -> CohortManifest:
    """Parse and validate a manifest dict; raises :class:`ManifestError` on failure."""
    schema = payload.get("schema_version")
    if not isinstance(schema, int) or schema != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(f"unsupported manifest schema_version={schema!r}")
    try:
        return CohortManifest.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - surface manifest schema violations
        raise ManifestError(f"manifest failed validation: {exc}") from exc


def load_manifest(path: Path | str) -> CohortManifest:
    """Load and validate a cohort manifest from disk."""
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"cohort manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError(f"manifest {path} is not an object")
    manifest = parse_manifest(payload)
    return manifest


def _sort_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sort_keys(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sort_keys(v) for v in value]
    return value


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ManifestOutcome",
    "ManifestSourceRecord",
    "CohortManifest",
    "ManifestError",
    "manifest_bytes",
    "manifest_hash",
    "dump_manifest",
    "parse_manifest",
    "load_manifest",
]
