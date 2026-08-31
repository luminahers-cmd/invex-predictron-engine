"""Offline evidence corpus — versioned, deterministic, commit-friendly.

The corpus (dataset) is a single JSON document that fully describes one
collected :class:`EvidenceBundle` so the engine can replay it offline
without any network access.  It is:

* **Deterministic** — serialized via Pydantic's stable key ordering and
  a pinned schema version, so identical inputs produce identical bytes.
* **Versioned** — every file carries a ``schema_version`` so format
  changes are explicit and migration-friendly.
* **Human-readable & commit-friendly** — plain JSON (indented, sorted
  keys), designed to live in the repository next to the benchmark
  snapshots in ``benchmarks/expected_outputs/``.
* **Self-contained** — no external services, no shared mutable state.

A corpus is produced by serializing a real (or synthesized) bundle and
consumed by :class:`ReplayEvidenceProvider`.  Because Document
Intelligence is pure and deterministic
(:func:`predictron_engine.evidence.document_intelligence.enrich_documents`),
re-enriching the stored documents through the orchestrator reproduces
identical metadata; the corpus intentionally stores the complete bundle
so a full field-for-field reconstruction is also possible via
:func:`rebuild_bundle`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from predictron_engine.evidence.models import EvidenceBundle, EvidenceDocument

SCHEMA_VERSION = 1

_DATASET_ROOT = Path(__file__).resolve().parents[3] / "benchmarks" / "offline_evidence"


class EvidenceCorpusError(RuntimeError):
    """Raised when an evidence corpus cannot be loaded or is invalid."""


def resolve_dataset_path(name: str) -> Path:
    """Resolve a dataset name to its committed JSON path.

    ``name`` may be an absolute/relative path or a bare dataset name. A
    bare name is resolved under the committed corpus root
    ``benchmarks/offline_evidence/`` (``name.json``).
    """
    candidate = Path(name)
    if candidate.is_absolute():
        return candidate
    if name.endswith(".json"):
        path = Path(name)
    else:
        path = Path(f"{name}.json")
    if Path(name).parent != Path("."):
        return path
    return _DATASET_ROOT / path

# Bare dataset names (no extension, no directory) only resolve under the
# committed corpus root, so a bare name never touches arbitrary paths.


def dataset_root() -> Path:
    """Return the committed offline evidence corpus root directory."""
    return _DATASET_ROOT


def serialize_bundle(
    bundle: EvidenceBundle,
    *,
    name: str,
    raw_documents: list[EvidenceDocument] | None = None,
) -> dict[str, Any]:
    """Serialize a bundle into the corpus document dict.

    ``name`` is retained for traceability. Wall-clock fields
    (``duration_ms``, ``collected_at``, ``processing_duration_ms``,
    per-provider ``duration_ms``) are preserved verbatim, since the goal
    is byte-for-byte faithful reconstruction, not re-measurement.

    ``raw_documents`` holds the pre-enrichment documents (i.e. all
    collected pages before Document Intelligence deduplication).  This is
    required for the provider path to reproduce the exact enrichment
    diagnostics (``documents_input`` / ``duplicates_removed``) — the
    enriched bundle alone has already been deduplicated and cannot
    reconstruct those counts.
    """
    raw = raw_documents if raw_documents is not None else bundle.documents
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "bundle": json.loads(bundle.model_dump_json()),
        "raw_documents": json.loads(
            EvidenceBundle(startup_name=bundle.startup_name, documents=raw).model_dump_json()
        )["documents"],
    }


def dump_corpus(
    bundle: EvidenceBundle,
    *,
    name: str,
    path: Path | str,
    raw_documents: list[EvidenceDocument] | None = None,
) -> None:
    """Write a bundle to a corpus file on disk (atomic, deterministic)."""
    document = serialize_bundle(bundle, name=name, raw_documents=raw_documents)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def load_corpus(path: Path) -> dict[str, Any]:
    """Load and validate a corpus document from disk."""
    if not path.exists():
        raise EvidenceCorpusError(f"evidence corpus not found: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceCorpusError(f"invalid evidence corpus {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise EvidenceCorpusError(f"evidence corpus {path} is not an object")
    schema = document.get("schema_version")
    if not isinstance(schema, int) or schema < 1 or schema > SCHEMA_VERSION:
        raise EvidenceCorpusError(
            f"unsupported evidence corpus schema_version={schema!r} in {path}"
        )
    if "bundle" not in document or not isinstance(document["bundle"], dict):
        raise EvidenceCorpusError(f"evidence corpus {path} is missing 'bundle'")
    if "raw_documents" not in document or not isinstance(
        document["raw_documents"], list
    ):
        raise EvidenceCorpusError(f"evidence corpus {path} is missing 'raw_documents'")
    return document


def raw_documents_from(document: dict[str, Any]) -> list[EvidenceDocument]:
    """Return the corpus's pre-enrichment documents.

    These are the documents as collected (before Document Intelligence
    enrichment) and are used by the replay provider so the orchestrator
    recomputes identical enrichment diagnostics.
    """
    try:
        return [EvidenceDocument.model_validate(d) for d in document["raw_documents"]]
    except Exception as exc:  # noqa: BLE001 - surface malformed corpora
        raise EvidenceCorpusError(f"corpus raw_documents failed validation: {exc}") from exc


def rebuild_bundle(document: dict[str, Any]) -> EvidenceBundle:
    """Reconstruct the :class:`EvidenceBundle` from a corpus document.

    Returns a bundle that is field-for-field identical to the bundle the
    corpus was created from, including wall-clock fields.
    """
    try:
        return EvidenceBundle.model_validate(document["bundle"])
    except Exception as exc:  # noqa: BLE001 - surface malformed corpora
        raise EvidenceCorpusError(f"corpus bundle failed validation: {exc}") from exc
