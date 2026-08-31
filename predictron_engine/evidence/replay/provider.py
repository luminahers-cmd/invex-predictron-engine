"""Replay evidence provider — deterministic, offline, no network.

Satisfies the existing :class:`EvidenceProvider` protocol
(:mod:`predictron_engine.evidence.provider_contracts`) and plugs directly
into :class:`EvidenceOrchestrator`.  It replays a stored offline corpus
instead of performing web collection, so a recorded collection run can be
re-executed byte-for-byte (on all deterministic fields) without touching
the network or the rest of the pipeline.

Design notes
------------
* The provider emits the corpus bundle's documents **without** Document
  Intelligence metadata (``metadata=None``).  The orchestrator's existing
  Phase 4 enrichment (:func:`enrich_documents`) is pure and deterministic,
  so it recomputes the identical classification/quality/authority/trust/
  dedup metadata.  This means the replay path exercises the exact same
  enrichment code as a live run.
* No subclassing or monkey-patching of the engine or orchestrator is
  required — the provider is a first-class ``EvidenceProvider``.
* It performs **no network access**; data is read only from the dataset
  on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from predictron_engine.evidence.models import EvidenceBundle, EvidenceDocument
from predictron_engine.evidence.provider_contracts import (
    CollectContext,
    ProviderResult,
)
from predictron_engine.evidence.replay.dataset import (
    EvidenceCorpusError,
    load_corpus,
    raw_documents_from,
    rebuild_bundle,
)

logger = logging.getLogger(__name__)


class ReplayEvidenceProvider:
    """Evidence provider that replays a stored offline corpus.

    Parameters
    ----------
    dataset:
        Path to the corpus file (see :func:`resolve_dataset_path` for the
        accepted name forms).  Resolved lazily on first collection so the
        provider can be constructed even before the file exists.
    name:
        Provider name reported in diagnostics.  Defaults to ``"replay"``.
    """

    def __init__(self, dataset: str | Path, *, name: str = "replay") -> None:
        self._dataset: str | Path = dataset
        self.name = name
        self._corpus: dict[str, Any] | None = None

    @property
    def dataset(self) -> str | Path:
        """Return the configured dataset reference."""
        return self._dataset

    def _load(self) -> dict[str, Any]:
        if self._corpus is None:
            path = _resolve(self._dataset)
            self._corpus = load_corpus(path)
        return self._corpus

    def can_collect(self, context: CollectContext) -> bool:
        """Return True when this provider can replay for the given context.

        Replay is an explicit, opt-in activity: when a dataset is not
        configured this provider is never registered (see
        :func:`build_replay_provider`).  When it is registered, it applies
        to the context that produced the corpus.
        """
        try:
            bundle = self._peek_bundle()
        except EvidenceCorpusError:
            return False
        if bundle is None:
            return False
        if context.startup_name and bundle.startup_name != context.startup_name:
            return False
        return True

    def _peek_bundle(self) -> EvidenceBundle | None:
        """Return the corpus bundle without caching (used by can_collect)."""
        try:
            document = load_corpus(_resolve(self._dataset))
        except EvidenceCorpusError:
            return None
        return rebuild_bundle(document)

    async def collect(self, context: CollectContext) -> ProviderResult:
        """Return the stored evidence as a :class:`ProviderResult`.

        Never performs network access.  Re-enrichment is left to the
        orchestrator so the emitted documents carry no intelligence
        metadata.
        """
        document = self._load()
        bundle = rebuild_bundle(document)

        documents = _strip_metadata(raw_documents_from(document))
        provider_duration_ms = _corpus_provider_duration(document, self.name)

        return ProviderResult(
            provider=self.name,
            website=bundle.website,
            documents=documents,
            sources=bundle.sources,
            attempted_pages=bundle.attempted_pages,
            duration_ms=provider_duration_ms,
            success=True,
        )


def _resolve(dataset: str | Path) -> Path:
    """Resolve a dataset reference to a concrete file path."""
    from predictron_engine.evidence.replay.dataset import resolve_dataset_path

    if isinstance(dataset, Path):
        return dataset
    return resolve_dataset_path(dataset)


def _strip_metadata(documents: list[EvidenceDocument]) -> list[EvidenceDocument]:
    """Return documents with intelligence metadata removed.

    Enrichment is deterministic, so dropping metadata lets the
    orchestrator recompute it identically while guaranteeing no stale
    metadata is replayed.
    """
    return [
        doc.model_copy(update={"metadata": None}) if doc.metadata is not None else doc
        for doc in documents
    ]


def _corpus_provider_duration(
    document: dict[str, Any], provider_name: str
) -> int:
    """Return the recorded duration (ms) for this provider, if present."""
    run = next(
        (r for r in document.get("bundle", {}).get("providers", [])
         if r.get("provider") == provider_name),
        None,
    )
    if run is None:
        return 0
    value = run.get("duration_ms", 0)
    return int(value) if isinstance(value, int | float) else 0
