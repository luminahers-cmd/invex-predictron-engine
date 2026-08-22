"""Evidence Citation Builder — links claims to source documents.

Provides deterministic, pure functions that build structured citations
linking evidence claims to the web documents that support them.  Every
function is side-effect-free: identical inputs always produce identical
outputs.

Design
------
* :func:`build_citation` creates a single :class:`EvidenceCitation` from
  an evidence item and a list of supporting documents.
* :func:`build_citations` creates citations in bulk for a list of
  evidence items.
* :func:`format_citation` renders a human-readable citation string.
* :func:`rank_citations_by_trust` sorts citations by trust score
  descending.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.evidence.models import EvidenceDocument
    from predictron_engine.models.report import EvidenceCitation, EvidenceItem


def build_citation(
    evidence_item: EvidenceItem,
    documents: list[EvidenceDocument] | None = None,
) -> EvidenceCitation:
    """Build a structured citation from an evidence item and its source documents.

    The citation links the evidence claim to the documents that support it,
    extracting trust metadata from the documents' intelligence enrichment.

    Parameters
    ----------
    evidence_item:
        The evidence item being cited.
    documents:
        Supporting documents from the evidence bundle.  When ``None`` or
        empty, the citation is built with trust score 0.0 and no source URLs.
    """
    from predictron_engine.models.report import EvidenceCitation

    documents = documents or []

    source_ids: list[str] = []
    source_urls: list[str] = []
    best_trust = 0.0

    for doc in documents:
        if doc.id not in source_ids:
            source_ids.append(doc.id)
        url_str = str(doc.url)
        if url_str not in source_urls:
            source_urls.append(url_str)

        if doc.metadata and doc.metadata.trust_score:
            if doc.metadata.trust_score.overall > best_trust:
                best_trust = doc.metadata.trust_score.overall

    citation_text = _format_citation_text(
        evidence_item.statement,
        evidence_item.domain,
        evidence_item.category,
        evidence_item.source,
        source_urls,
    )

    return EvidenceCitation(
        claim=evidence_item.statement,
        domain=evidence_item.domain,
        category=evidence_item.category,
        source_document_ids=source_ids,
        source_urls=source_urls,
        best_trust_score=round(best_trust, 4),
        citation_text=citation_text,
        provider=evidence_item.source,
    )


def build_citations(
    evidence_items: list[EvidenceItem],
    documents: list[EvidenceDocument] | None = None,
) -> list[EvidenceCitation]:
    """Build citations for a list of evidence items.

    Each evidence item is matched to all provided documents by comparing
    the item's source string against document titles and URLs.  Items
    without matching documents still receive citations with zero trust.

    Parameters
    ----------
    evidence_items:
        Evidence items to build citations for.
    documents:
        Pool of source documents from the evidence bundle.
    """
    documents = documents or []
    return [
        build_citation(item, _match_documents(item, documents))
        for item in evidence_items
    ]


def _match_documents(
    evidence_item: EvidenceItem,
    documents: list[EvidenceDocument],
) -> list[EvidenceDocument]:
    """Match evidence items to documents by URL, title, and source string.

    Matching is deterministic and based on substring containment:
    a document matches when its URL or title appears in the evidence
    source string, or vice versa.
    """
    matched: list[EvidenceDocument] = []
    source_lower = evidence_item.source.lower()

    for doc in documents:
        url_str = str(doc.url).lower()
        title_lower = doc.title.lower()

        if (
            url_str in source_lower
            or source_lower in url_str
            or title_lower in source_lower
            or source_lower in title_lower
        ):
            matched.append(doc)

    return matched


def format_citation(citation: EvidenceCitation) -> str:
    """Render a citation as a human-readable string.

    Returns a formatted string like::

        "[industry/sales_cycle] Enterprise sales cycles are 12-18 months (source: example.com)"
    """
    if citation.citation_text:
        return citation.citation_text

    return _format_citation_text(
        citation.claim,
        citation.domain,
        citation.category,
        citation.provider,
        citation.source_urls,
    )


def format_citation_list(citations: list[EvidenceCitation]) -> list[str]:
    """Render a list of citations as formatted strings."""
    return [format_citation(c) for c in citations]


def rank_citations_by_trust(
    citations: list[EvidenceCitation],
) -> list[EvidenceCitation]:
    """Sort citations by trust score descending.

    Citations with equal trust scores maintain their original order.
    """
    return sorted(citations, key=lambda c: c.best_trust_score, reverse=True)


def deduplicate_citations(citations: list[EvidenceCitation]) -> list[EvidenceCitation]:
    """Remove duplicate citations based on claim and domain.

    When duplicates exist, the citation with the higher trust score
    is retained.
    """
    seen: dict[tuple[str, str], EvidenceCitation] = {}
    for citation in citations:
        key = (citation.claim, citation.domain)
        existing = seen.get(key)
        if existing is None or citation.best_trust_score > existing.best_trust_score:
            seen[key] = citation
    return list(seen.values())


# ────────────────────────────────────────────────────────────────────
# Private helpers
# ────────────────────────────────────────────────────────────────────


def _format_citation_text(
    claim: str,
    domain: str,
    category: str,
    provider: str,
    source_urls: list[str],
) -> str:
    """Format the human-readable citation text."""
    parts = [f"[{domain}/{category}] {claim}"]

    if source_urls:
        domains_list = [_extract_display_domain(u) for u in source_urls[:3]]
        parts.append(f"(source: {', '.join(domains_list)})")
    elif provider:
        parts.append(f"(source: {provider})")

    return " ".join(parts)


def _extract_display_domain(url: str) -> str:
    """Extract a short display domain from a URL."""
    url = url.replace("https://", "").replace("http://", "")
    url = url.rstrip("/")
    slash_idx = url.find("/")
    if slash_idx != -1:
        return url[:slash_idx]
    return url
