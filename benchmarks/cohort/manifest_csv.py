"""CSV cohort manifests (Phase 4 prediction validation).

The JSOM manifest (:mod:`benchmarks.cohort.manifest`) is the canonical,
lossless document format.  This module adds a **flat CSV** surface for the
same schema so a verified cohort can be supplied as either ``.json`` or
``.csv``:

* one row per company (``ManifestSourceRecord``) with the outcome columns
  flattened onto the same row (``ManifestOutcome``);
* a single ``manifest`` section row carrying the document-level fields
  (``dataset_name``, ``benchmark_version``, ``created_at``,
  ``evaluation_horizon_days``, ``notes``, ``schema_version``);
* lists and nested objects are stored as JSON in their cell so the round
  trip is lossless (no delimiter ambiguity).

:func:`load_manifest_any` dispatches on file extension so callers can
hand a path to either format.
"""

from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from benchmarks.cohort.manifest import (
    MANIFEST_SCHEMA_VERSION,
    CohortManifest,
    ManifestError,
    ManifestOutcome,
    ManifestSourceRecord,
)
from benchmarks.ground_truth.schema import OutcomeEvent, StartupStatus

_MANIFEST_SECTION = "manifest"
_RECORD_SECTION = "record"

CSV_COLUMNS: tuple[str, ...] = (
    "section",
    "schema_version",
    "dataset_name",
    "benchmark_version",
    "created_at",
    "evaluation_horizon_days",
    "notes",
    "company_id",
    "company_name",
    "description",
    "website",
    "pitch_deck_url",
    "founder_linkedin_urls_json",
    "sector",
    "country",
    "stage",
    "evidence_corpus",
    "analysis_timestamp",
    "record_evaluation_horizon_days",
    "outcome_status",
    "outcome_verification_date",
    "outcome_exit_value_usd",
    "outcome_sources_json",
    "outcome_verified",
    "outcome_verified_by",
    "outcome_notes",
    "outcome_events_json",
    "record_metadata_json",
)


class ManifestFormatError(ManifestError):
    """Raised when a CSV cohort manifest is malformed."""


# ---------------------------------------------------------------------------
# Encoding helpers (lossless round trip)
# ---------------------------------------------------------------------------


def _optional_str(value: str) -> str | None:
    return value if value else None


def _to_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "y")


def _to_float(value: str) -> float | None:
    if not value:
        return None
    return float(value)


def _to_int(value: str) -> int | None:
    if not value:
        return None
    return int(value)


def _to_datetime(value: str, *, field: str) -> datetime:
    if not value:
        raise ManifestFormatError(f"missing required {field!r} in CSV manifest")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ManifestFormatError(f"invalid {field!r} {value!r}: {exc}") from exc
    if parsed.tzinfo is None:
        raise ManifestFormatError(f"{field!r} in CSV manifest must be timezone-aware (UTC)")
    return parsed


def _to_list(value: str) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ManifestFormatError(f"invalid JSON list cell: {exc}") from exc
    if not isinstance(parsed, list):
        raise ManifestFormatError("expected a JSON list cell")
    return [str(item) for item in parsed]


def _to_obj(value: str, *, field: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ManifestFormatError(f"invalid JSON for {field!r}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ManifestFormatError(f"{field!r} must hold a JSON object")
    return parsed


def _to_date(value: str) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Row <-> record mapping
# ---------------------------------------------------------------------------


def _manifest_row(manifest: CohortManifest) -> dict[str, str]:
    return {
        "section": _MANIFEST_SECTION,
        "schema_version": str(manifest.schema_version),
        "dataset_name": manifest.dataset_name,
        "benchmark_version": manifest.benchmark_version,
        "created_at": manifest.created_at.isoformat(),
        "evaluation_horizon_days": (
            str(manifest.evaluation_horizon_days)
            if manifest.evaluation_horizon_days is not None
            else ""
        ),
        "notes": manifest.notes,
    }


def _record_row(record: ManifestSourceRecord) -> dict[str, str]:
    outcome = record.outcome
    row: dict[str, str] = {
        "section": _RECORD_SECTION,
        "company_id": record.company_id,
        "company_name": record.company_name,
        "description": record.description,
        "website": record.website or "",
        "pitch_deck_url": record.pitch_deck_url or "",
        "founder_linkedin_urls_json": _json_cell(record.founder_linkedin_urls),
        "sector": record.sector or "",
        "country": record.country or "",
        "stage": record.stage or "",
        "evidence_corpus": record.evidence_corpus,
        "analysis_timestamp": record.analysis_timestamp.isoformat(),
        "record_evaluation_horizon_days": (
            str(record.evaluation_horizon_days)
            if record.evaluation_horizon_days is not None
            else ""
        ),
        "record_metadata_json": _json_cell(record.metadata),
        "outcome_status": outcome.status.value if outcome is not None else "",
        "outcome_verification_date": (
            outcome.verification_date.isoformat() if outcome is not None else ""
        ),
        "outcome_exit_value_usd": (
            str(outcome.exit_value_usd)
            if outcome is not None and outcome.exit_value_usd is not None
            else ""
        ),
        "outcome_sources_json": _json_cell(outcome.sources if outcome is not None else []),
        "outcome_verified": "1" if outcome is not None and outcome.verified else "0",
        "outcome_verified_by": (outcome.verified_by or "") if outcome is not None else "",
        "outcome_notes": outcome.notes if outcome is not None else "",
        "outcome_events_json": _json_cell(
            [
                {
                    "kind": event.kind.value,
                    "occurred_at": event.occurred_at.isoformat(),
                    "value_currency_usd": event.value_currency_usd,
                    "amount_nominal_units": event.amount_nominal_units,
                    "description": event.description,
                    "sources": list(event.sources),
                }
                for event in (outcome.outcome_events if outcome is not None else [])
            ]
        ),
    }
    return row


def _record_from_row(row: dict[str, str]) -> ManifestSourceRecord:
    status = _optional_str(row.get("outcome_status", ""))
    verification_date = _to_date(row.get("outcome_verification_date", ""))
    try:
        exit_value_usd = _to_float(row.get("outcome_exit_value_usd", ""))
    except ValueError as exc:
        raise ManifestFormatError(
            f"invalid outcome_exit_value_usd {row.get('outcome_exit_value_usd')!r}: {exc}"
        ) from exc
    sources = _to_list(row.get("outcome_sources_json", ""))
    verified = _to_bool(row.get("outcome_verified", "0"))
    verified_by = _optional_str(row.get("outcome_verified_by", ""))
    notes = row.get("outcome_notes", "")
    outcome = None
    if status:
        if verification_date is None:
            raise ManifestFormatError(
                f"outcome_verification_date required when an outcome is present "
                f"for {row.get('company_id', '?')!r}"
            )
        outcome = ManifestOutcome(
            status=StartupStatus(status),
            verification_date=verification_date,
            outcome_events=_to_events(row.get("outcome_events_json", "")),
            exit_value_usd=exit_value_usd,
            sources=sources,
            verified=verified,
            verified_by=verified_by,
            notes=notes,
        )

    horizon = _to_int(row.get("record_evaluation_horizon_days", ""))
    if horizon is not None and horizon < 0:
        raise ManifestFormatError("record_evaluation_horizon_days must be non-negative")
    return ManifestSourceRecord(
        company_id=row["company_id"],
        company_name=row["company_name"],
        description=row.get("description", ""),
        website=_optional_str(row.get("website", "")),
        pitch_deck_url=_optional_str(row.get("pitch_deck_url", "")),
        founder_linkedin_urls=_to_list(row.get("founder_linkedin_urls_json", "")),
        sector=_optional_str(row.get("sector", "")),
        country=_optional_str(row.get("country", "")),
        stage=_optional_str(row.get("stage", "")),
        evidence_corpus=row["evidence_corpus"],
        analysis_timestamp=_to_datetime(
            row.get("analysis_timestamp", ""), field="analysis_timestamp"
        ),
        evaluation_horizon_days=horizon,
        outcome=outcome,
        metadata=_to_obj(row.get("record_metadata_json", ""), field="record_metadata_json"),
    )


def _to_events(value: str) -> list[OutcomeEvent]:
    """Decode an ``outcome_events_json`` cell (a JSON array of event dicts)."""
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ManifestFormatError(f"invalid JSON for outcome_events_json: {exc}") from exc
    if not isinstance(parsed, list):
        raise ManifestFormatError("outcome_events_json must hold a JSON array")
    events: list[OutcomeEvent] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ManifestFormatError("outcome_events_json entries must be objects")
        occurred = item.get("occurred_at")
        if occurred is None:
            raise ManifestFormatError("outcome_events_json entries require occurred_at")
        events.append(
            OutcomeEvent(
                kind=item["kind"],
                occurred_at=(
                    occurred
                    if isinstance(occurred, date)
                    else date.fromisoformat(str(occurred))
                ),
                value_currency_usd=item.get("value_currency_usd"),
                amount_nominal_units=item.get("amount_nominal_units"),
                description=str(item.get("description", "")),
                sources=[str(s) for s in item.get("sources", [])],
            )
        )
    return events


def _manifest_from_rows(rows: list[dict[str, str]]) -> CohortManifest:
    manifest_row: dict[str, str] | None = None
    records: list[ManifestSourceRecord] = []
    for row in rows:
        if row.get("section") == _MANIFEST_SECTION:
            if manifest_row is not None:
                raise ManifestFormatError("CSV manifest must contain exactly one manifest row")
            manifest_row = row
            continue
        if row.get("section") != _RECORD_SECTION:
            raise ManifestFormatError(f"unknown CSV section {row.get('section')!r}")
        records.append(_record_from_row(row))

    if manifest_row is None:
        raise ManifestFormatError("CSV manifest is missing its manifest section row")

    schema_version = _to_int(manifest_row.get("schema_version", ""))
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ManifestFormatError(
            f"unsupported manifest schema_version={schema_version!r}"
        )
    created_at = _to_datetime(manifest_row.get("created_at", ""), field="created_at")
    doc_horizon = _to_int(manifest_row.get("evaluation_horizon_days", ""))
    if doc_horizon is not None and doc_horizon < 0:
        raise ManifestFormatError("evaluation_horizon_days must be non-negative")

    manifest = CohortManifest(
        schema_version=schema_version,
        dataset_name=manifest_row.get("dataset_name", ""),
        benchmark_version=manifest_row.get("benchmark_version", ""),
        created_at=created_at,
        evaluation_horizon_days=doc_horizon,
        notes=manifest_row.get("notes", ""),
        source_records=records,
    )
    if not manifest.dataset_name or not manifest.benchmark_version:
        raise ManifestFormatError("dataset_name and benchmark_version are required")
    return manifest


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest_to_rows(manifest: CohortManifest) -> list[dict[str, str]]:
    """Flatten a manifest into its deterministic CSV rows (doc row first)."""
    records = sorted(manifest.source_records, key=lambda r: r.company_id)
    return [_manifest_row(manifest)] + [_record_row(r) for r in records]


def manifest_from_rows(rows: list[dict[str, Any]]) -> CohortManifest:
    """Rehydrate a manifest from CSV rows (see :func:`manifest_to_rows`)."""
    normalized = [
        {str(key): (value if isinstance(value, str) else str(value)) for key, value in row.items()}
        for row in rows
    ]
    return _manifest_from_rows(normalized)


def dump_manifest_csv(
    manifest: CohortManifest,
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a manifest to a flat CSV (deterministic, refuse-to-clobber)."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise ManifestError(f"manifest already exists: {path} (pass overwrite=True to replace)")
    rows = manifest_to_rows(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    tmp.replace(path)
    return path


def load_manifest_csv(path: Path | str) -> CohortManifest:
    """Load and validate a flat CSV cohort manifest from disk."""
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"cohort manifest not found: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows: list[dict[str, str]] = []
            required = list(CSV_COLUMNS)
            for row in reader:
                if row is None:
                    continue
                missing = [column for column in required if column not in row]
                if missing:
                    raise ManifestFormatError(
                        f"CSV manifest {path} is missing required columns: "
                        f"{', '.join(missing)}"
                    )
                rows.append(dict(row))
    except ManifestFormatError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface IO/schema violations
        raise ManifestError(f"invalid CSV manifest {path}: {exc}") from exc
    if not rows:
        raise ManifestFormatError(f"CSV manifest {path} contains no rows")
    try:
        return manifest_from_rows(rows)
    except (ManifestError, ValueError) as exc:
        if isinstance(exc, ManifestError):
            raise
        raise ManifestError(f"manifest failed validation: {exc}") from exc


def load_manifest_any(path: Path | str) -> CohortManifest:
    """Load a cohort manifest from a JSON or CSV file (dispatch by suffix)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_manifest_csv(path)
    if suffix in (".json", ""):
        from benchmarks.cohort.manifest import load_manifest

        return load_manifest(path)
    raise ManifestError(
        f"unsupported manifest format {suffix!r} for {path} (expected .json or .csv)"
    )


__all__ = [
    "CSV_COLUMNS",
    "ManifestFormatError",
    "dump_manifest_csv",
    "load_manifest_any",
    "load_manifest_csv",
    "manifest_from_rows",
    "manifest_to_rows",
]
