"""ReasoningContext — structured context passed into reasoning rules.

Sprint 6A introduces an evidence-aware execution context for the
reasoning layer.  Instead of every rule re-implementing the same
lookup logic (filter by domain, find trusted documents, resolve
citations, inspect provenance), the context performs each lookup once,
indexes the results, and exposes them through small deterministic
accessor methods.

The context is a read-only value object: rules must never mutate it.
All accessors are pure functions of the data captured at construction,
so identical inputs always produce identical outputs.

Exposed state:

* ``features``            — ExtractedFeatures for this analysis run
* ``evidence_items``      — flat list of EvidenceItem from the evidence stage
* ``bundle``              — the collected EvidenceBundle (may be empty)
* ``trust_summary``       — aggregate TrustSummary for the bundle
* ``retrieval_diagnostics`` — per-provider ProviderRun records
* ``extraction_diagnostics``— per-provider extraction run metadata
* ``provenance_document_ids`` — document ids recorded during extraction

Backward compatibility: constructing a ReasoningContext is entirely
optional.  Rules that do not implement ``evaluate_context`` continue to
receive ``(features, evidence)`` exactly as before.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.evidence.models import (
        EvidenceBundle,
        EvidenceDocument,
        ProviderRun,
    )
    from predictron_engine.evidence.provenance import TrustSummary
    from predictron_engine.evidence.retrieval import (  # noqa: F401
        retrieve_best_source,
    )
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import EvidenceCitation, EvidenceItem

logger = logging.getLogger(__name__)


class ReasoningContext:
    """Read-only, pre-indexed context for a single reasoning pass.

    Parameters
    ----------
    features:
        Extracted features produced by the extraction stage.
    evidence_items:
        Evidence items gathered by the evidence stage.
    bundle:
        Website evidence bundle collected before extraction.  An empty
        bundle is used when collection was skipped or unavailable.
    """

    def __init__(
        self,
        features: ExtractedFeatures,
        evidence_items: list[EvidenceItem],
        bundle: EvidenceBundle | None = None,
    ) -> None:
        self.features = features
        self.evidence_items: list[EvidenceItem] = list(evidence_items)
        self.bundle = bundle

        # ── Pre-built indexes (avoid repeated scans inside rules) ──
        self._items_by_domain: dict[str, list[EvidenceItem]] = {}
        for item in self.evidence_items:
            self._items_by_domain.setdefault(item.domain, []).append(item)

        self._documents: list[EvidenceDocument] = [
            doc
            for doc in (bundle.documents if bundle else [])
            if doc.status.value == "success"
        ]
        self._docs_by_provider: dict[str, list[EvidenceDocument]] = {}
        for doc in self._documents:
            provider = ""
            if doc.metadata is not None:
                provider = doc.metadata.source_provider
            self._docs_by_provider.setdefault(provider, []).append(doc)

        self._doc_by_id: dict[str, EvidenceDocument] = {
            doc.id: doc for doc in self._documents
        }

        self._trusted_docs: list[EvidenceDocument] = sorted(
            [
                doc
                for doc in self._documents
                if doc.metadata is not None
                and doc.metadata.trust_score is not None
                and doc.metadata.trust_score.overall >= 0.5
            ],
            key=lambda d: (
                d.metadata.trust_score.overall,  # type: ignore[union-attr]
                str(d.url),
            ),
            reverse=True,
        )

        scored_for_best = [
            doc
            for doc in self._documents
            if doc.metadata is not None
            and doc.metadata.trust_score is not None
        ]
        self._best_source: EvidenceDocument | None = (
            max(
                scored_for_best,
                key=lambda d: (
                    d.metadata.trust_score.overall,  # type: ignore[union-attr]
                    d.metadata.quality_score,  # type: ignore[union-attr]
                    str(d.url),
                ),
            )
            if scored_for_best
            else None
        )

        trust_values = [
            doc.metadata.trust_score.overall
            for doc in self._documents
            if doc.metadata is not None
            and doc.metadata.trust_score is not None
        ]
        self._average_trust: float = (
            round(sum(trust_values) / len(trust_values), 4)
            if trust_values
            else 0.0
        )

        self._documents_snapshot: list[EvidenceDocument] = list(self._documents)
        self._retrieval_diagnostics_snapshot: list[ProviderRun] = (
            list(self.bundle.providers) if self.bundle is not None else []
        )
        self._provenance_records: list[dict[str, object]] = self._parse_provenance_records()

    def _parse_provenance_records(self) -> list[dict[str, object]]:
        """Pre-parse all provenance records once at construction time."""
        records: list[dict[str, object]] = []
        for item in self.evidence_items:
            encoded = item.provenance_record
            if not encoded:
                continue
            try:
                parsed = json.loads(encoded)
            except (TypeError, ValueError):
                logger.warning("Skipping unparsable provenance record")
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
        return records

    # ────────────────────────────────────────────────────────────────
    # Evidence item accessors
    # ────────────────────────────────────────────────────────────────

    def evidence_for_domain(self, domain: str) -> list[EvidenceItem]:
        """Return evidence items matching the given domain."""
        return list(self._items_by_domain.get(domain, []))

    def evidence_for_domains(
        self, domains: list[str]
    ) -> list[EvidenceItem]:
        """Return evidence items matching any of the given domains."""
        return [
            item
            for domain in domains
            for item in self._items_by_domain.get(domain, [])
        ]

    # ────────────────────────────────────────────────────────────────
    # Document / retrieval accessors
    # ────────────────────────────────────────────────────────────────

    @property
    def documents(self) -> list[EvidenceDocument]:
        """Successfully collected evidence documents."""
        return self._documents_snapshot

    def documents_by_provider(self, provider: str) -> list[EvidenceDocument]:
        """Return successful documents collected by the given provider."""
        return list(self._docs_by_provider.get(provider, []))

    def document_by_id(self, document_id: str) -> EvidenceDocument | None:
        """Resolve a document id to its EvidenceDocument, when available."""
        return self._doc_by_id.get(document_id)

    def trusted_documents(self, min_trust: float = 0.5) -> list[EvidenceDocument]:
        """Return documents with trust score >= min_trust, highest first."""
        if min_trust == 0.5:
            return list(self._trusted_docs)
        scored = [
            doc
            for doc in self._documents
            if doc.metadata is not None
            and doc.metadata.trust_score is not None
            and doc.metadata.trust_score.overall >= min_trust
        ]
        return sorted(
            scored,
            key=lambda d: (
                d.metadata.trust_score.overall,  # type: ignore[union-attr]
                str(d.url),
            ),
            reverse=True,
        )

    def best_source(self) -> EvidenceDocument | None:
        """Return the highest-trust document, or None when none are scored."""
        return self._best_source

    # ────────────────────────────────────────────────────────────────
    # Trust & diagnostics accessors
    # ────────────────────────────────────────────────────────────────

    @property
    def trust_summary(self) -> TrustSummary | None:
        """Aggregate trust summary for the bundle (None when never scored)."""
        if self.bundle is not None and self.bundle.trust_summary is not None:
            return self.bundle.trust_summary
        return None

    @property
    def average_trust(self) -> float:
        """Average trust score across scored documents (0.0 when unscored)."""
        return self._average_trust

    @property
    def retrieval_diagnostics(self) -> list[ProviderRun]:
        """Per-provider retrieval diagnostic records from the bundle."""
        return self._retrieval_diagnostics_snapshot

    @property
    def extraction_diagnostics(self) -> dict[str, dict[str, object]]:
        """Per-provider extraction run metadata recorded on the features."""
        return dict(getattr(self.features, "provider_run_metadata", {}) or {})

    # ────────────────────────────────────────────────────────────────
    # Provenance accessors
    # ────────────────────────────────────────────────────────────────

    @property
    def provenance_document_ids(self) -> list[str]:
        """Document ids recorded during extraction (deduplicated, ordered)."""
        ids: list[str] = []
        for doc_id in getattr(self.features, "provenance_document_ids", []):
            if doc_id not in ids:
                ids.append(doc_id)
        return ids

    def provenance_records(self) -> list[dict[str, object]]:
        """Return pre-parsed provenance records attached to evidence items.

        Each record is the JSON-encoded ProvenanceRecord stored on an
        EvidenceItem.  Unparsable records are skipped so a single bad
        record can never break reasoning.
        """
        return self._provenance_records

    def citations_for_domain(self, domain: str) -> list[EvidenceCitation]:
        """Collect all citations carried by evidence items in a domain."""
        citations: list[EvidenceCitation] = []
        for item in self.evidence_for_domain(domain):
            citations.extend(item.citations)
        return citations


def build_reasoning_context(
    features: ExtractedFeatures,
    evidence_items: list[EvidenceItem],
    bundle: EvidenceBundle | None = None,
) -> ReasoningContext:
    """Factory function creating a ReasoningContext.

    Pure factory — no side effects, no I/O.
    """
    return ReasoningContext(features, evidence_items, bundle)
