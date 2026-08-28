"""Evidence Retrieval — queries the EvidenceBundle for relevant documents.

Provides deterministic, pure functions that query the evidence bundle
to retrieve documents by type, provider, trust level, or other metadata.
Every function is side-effect-free: identical inputs always produce
identical outputs.

Design
------
* :func:`retrieve_documents_by_type` filters documents by ``DocumentType``.
* :func:`retrieve_documents_by_provider` filters by source provider name.
* :func:`retrieve_trusted_documents` filters by minimum trust score.
* :func:`retrieve_best_source` returns the highest-trust document.
* :func:`retrieve_evidence_for_domain` filters evidence items by domain.
* :func:`retrieve_citations_for_domain` builds citations for a domain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.evidence.models import DocumentType, EvidenceBundle, EvidenceDocument
    from predictron_engine.models.report import EvidenceCitation, EvidenceItem


def retrieve_documents_by_type(
    bundle: EvidenceBundle,
    doc_type: DocumentType,
) -> list[EvidenceDocument]:
    """Return documents matching the given document type.

    Only documents with successful status and a non-None metadata
    field are considered.
    """
    from predictron_engine.evidence.models import DocumentStatus

    return [
        doc
        for doc in bundle.documents
        if doc.status == DocumentStatus.SUCCESS
        and doc.metadata is not None
        and doc.metadata.document_type == doc_type
    ]


def retrieve_documents_by_provider(
    bundle: EvidenceBundle,
    provider: str,
) -> list[EvidenceDocument]:
    """Return documents collected by the given provider name.

    Matches against the ``source_provider`` field in document metadata.
    Only documents with successful status are included.
    """
    from predictron_engine.evidence.models import DocumentStatus

    return [
        doc
        for doc in bundle.documents
        if doc.status == DocumentStatus.SUCCESS
        and doc.metadata is not None
        and doc.metadata.source_provider == provider
    ]


def retrieve_trusted_documents(
    bundle: EvidenceBundle,
    min_trust: float = 0.5,
) -> list[EvidenceDocument]:
    """Return documents with trust score >= min_trust.

    Documents without a trust score are excluded.  Results are ordered
    by trust score descending.
    """
    from predictron_engine.evidence.models import DocumentStatus

    trusted = [
        doc
        for doc in bundle.documents
        if doc.status == DocumentStatus.SUCCESS
        and doc.metadata is not None
        and doc.metadata.trust_score is not None
        and doc.metadata.trust_score.overall >= min_trust
    ]
    return sorted(
        trusted,
        key=lambda d: d.metadata.trust_score.overall,  # type: ignore[union-attr]
        reverse=True,
    )


def retrieve_best_source(bundle: EvidenceBundle) -> EvidenceDocument | None:
    """Return the document with the highest trust score.

    Returns ``None`` when no documents have trust scores.  Ties are
    broken by quality score, then by URL for determinism.
    """
    from predictron_engine.evidence.models import DocumentStatus

    scored = [
        doc
        for doc in bundle.documents
        if doc.status == DocumentStatus.SUCCESS
        and doc.metadata is not None
        and doc.metadata.trust_score is not None
    ]
    if not scored:
        return None

    return max(
        scored,
        key=lambda d: (
            d.metadata.trust_score.overall
            if d.metadata is not None and d.metadata.trust_score is not None
            else 0.0,
            d.metadata.quality_score if d.metadata is not None else 0.0,
            str(d.url),
        ),
    )


def retrieve_evidence_for_domain(
    evidence_items: list[EvidenceItem],
    domain: str,
) -> list[EvidenceItem]:
    """Return evidence items matching the given domain.

    This is a domain-level convenience wrapper.  For general-purpose
    filtering, use :func:`filter_evidence` from reasoning.rules.base.
    """
    return [e for e in evidence_items if e.domain == domain]


def retrieve_citations_for_domain(
    evidence_items: list[EvidenceItem],
    domain: str,
    documents: list[EvidenceDocument] | None = None,
) -> list[EvidenceCitation]:
    """Build and return citations for all evidence items in a domain.

    Parameters
    ----------
    evidence_items:
        Pool of evidence items to filter.
    domain:
        Target domain to retrieve citations for.
    documents:
        Pool of source documents for citation building.
    """
    from predictron_engine.evidence.citation import build_citations

    domain_items = retrieve_evidence_for_domain(evidence_items, domain)
    return build_citations(domain_items, documents)


def retrieve_citations_for_observation(
    evidence_refs: list[str],
    evidence_items: list[EvidenceItem],
    documents: list[EvidenceDocument] | None = None,
) -> list[EvidenceCitation]:
    """Build citations for evidence referenced by an observation.

    Observations store evidence as reference strings like
    ``"evidence:{domain}/{category}: {statement}"``.  This function
    matches those references back to evidence items and builds citations.
    """
    from predictron_engine.evidence.citation import build_citations

    matched_items = _match_evidence_by_refs(evidence_refs, evidence_items)
    return build_citations(matched_items, documents)


def _match_evidence_by_refs(
    evidence_refs: list[str],
    evidence_items: list[EvidenceItem],
) -> list[EvidenceItem]:
    """Match evidence reference strings to evidence items.

    Reference format: ``"evidence:{domain}/{category}: {statement}"``.
    Matches by domain, category, and statement substring.
    """
    matched: list[EvidenceItem] = []
    for ref in evidence_refs:
        if not ref.startswith("evidence:"):
            continue
        ref_body = ref[len("evidence:"):]
        slash_idx = ref_body.find("/")
        if slash_idx == -1:
            continue
        colon_idx = ref_body.find(": ")
        if colon_idx == -1:
            continue
        ref_domain = ref_body[:slash_idx]
        ref_category = ref_body[slash_idx + 1 : colon_idx]
        ref_statement = ref_body[colon_idx + 2 :]

        for item in evidence_items:
            if (
                item.domain == ref_domain
                and item.category == ref_category
                and ref_statement in item.statement
            ):
                if item not in matched:
                    matched.append(item)
    return matched
