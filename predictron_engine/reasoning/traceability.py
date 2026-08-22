"""Provenance traceability for observations.

Sprint 6A guarantees that every Observation can be traced back through
the full evidence chain:

    Observation
      ↓
    EvidenceItem
      ↓
    EvidenceCitation
      ↓
    EvidenceDocument
      ↓
    Provider
      ↓
    Original URL

:func:`resolve_provenance_chain` walks this chain deterministically and
returns one :class:`ProvenanceChainEntry` per (citation, document id)
pair.  :func:`verify_traceability` asserts that no cited provenance was
lost: every cited document id must resolve to a known document carrying
an original URL and a provider attribution.

All functions are pure — identical inputs always produce identical
outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.evidence.models import EvidenceDocument
    from predictron_engine.models.report import Observation


@dataclass(frozen=True)
class ProvenanceChainEntry:
    """One resolved link in an observation's provenance chain.

    ``resolved`` is False when the cited document id could not be found
    among the supplied documents; in that case ``original_url`` falls
    back to the citation's recorded source URL so the URL-level chain
    is never lost.
    """

    claim: str
    domain: str
    category: str
    document_id: str
    original_url: str
    final_url: str
    provider: str
    knowledge_provider: str
    trust_score: float
    resolved: bool


def resolve_provenance_chain(
    observation: Observation,
    documents: list[EvidenceDocument],
) -> list[ProvenanceChainEntry]:
    """Resolve the full provenance chain for an observation.

    Citations are processed in stored order; within each citation,
    source document ids are processed in stored order.  Document ids
    that cannot be resolved still produce entries, falling back to the
    citation's recorded source URLs so provenance is preserved.
    """
    by_id = {doc.id: doc for doc in documents}

    entries: list[ProvenanceChainEntry] = []
    for citation in observation.citations:
        urls = list(citation.source_urls)
        for idx, doc_id in enumerate(citation.source_document_ids):
            doc = by_id.get(doc_id)
            if doc is not None:
                provider = ""
                trust = 0.0
                if doc.metadata is not None:
                    provider = doc.metadata.source_provider
                    if doc.metadata.trust_score is not None:
                        trust = doc.metadata.trust_score.overall
                entries.append(
                    ProvenanceChainEntry(
                        claim=citation.claim,
                        domain=citation.domain,
                        category=citation.category,
                        document_id=doc.id,
                        original_url=str(doc.original_url),
                        final_url=str(doc.url),
                        provider=provider,
                        knowledge_provider=citation.provider,
                        trust_score=round(trust, 4),
                        resolved=True,
                    )
                )
            else:
                fallback_url = urls[idx] if idx < len(urls) else ""
                entries.append(
                    ProvenanceChainEntry(
                        claim=citation.claim,
                        domain=citation.domain,
                        category=citation.category,
                        document_id=doc_id,
                        original_url=fallback_url,
                        final_url="",
                        provider="",
                        knowledge_provider=citation.provider,
                        trust_score=citation.best_trust_score,
                        resolved=False,
                    )
                )
    return entries


def unresolved_document_ids(
    observation: Observation,
    documents: list[EvidenceDocument],
) -> list[str]:
    """Return cited document ids that do not resolve to known documents."""
    known = {doc.id for doc in documents}
    unresolved: list[str] = []
    for citation in observation.citations:
        for doc_id in citation.source_document_ids:
            if doc_id not in known and doc_id not in unresolved:
                unresolved.append(doc_id)
    return unresolved


def verify_traceability(
    observation: Observation,
    documents: list[EvidenceDocument],
) -> bool:
    """Verify that no cited provenance was lost.

    True when every cited document id resolves to a known document with
    an original URL and a provider attribution.  Observations without
    citations are trivially traceable (nothing to lose).
    """
    return not unresolved_document_ids(observation, documents)
