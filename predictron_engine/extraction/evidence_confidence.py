"""Evidence-aware extraction confidence scoring.

Computes a deterministic confidence score for extracted features based
on evidence quality, trust, agreement, and diversity.  Every function
is pure: identical inputs always produce identical outputs.

Design
------
* :func:`compute_evidence_confidence` — main entry point.
* :func:`_compute_evidence_quality_score` — evidence quality component.
* :func:`_compute_evidence_trust_score` — evidence trust component.
* :func:`_compute_evidence_agreement_score` — agreement component.
* :func:`_compute_source_diversity_score` — source diversity component.
* :func:`_compute_signal_density_score` — signal density component.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.evidence.models import EvidenceDocument
    from predictron_engine.models.report import EvidenceItem


def compute_evidence_confidence(
    *,
    evidence_items: list[EvidenceItem],
    documents: list[EvidenceDocument],
    keywords_matched: int,
    total_keywords: int,
    description_length: int,
) -> float:
    """Compute a deterministic confidence score for evidence-aware extraction.

    Combines multiple factors into a single [0, 1] confidence score:
      - Evidence quality: fraction of documents with trust scores
      - Evidence trust: average trust score across documents
      - Evidence agreement: fraction of items with corroborating sources
      - Source diversity: number of distinct source providers
      - Signal density: keyword match rate

    All parameters are primitives — no domain objects required for the
    core computation.

    Parameters
    ----------
    evidence_items:
        Evidence items relevant to this extraction domain.
    documents:
        Documents retrieved by the domain's retrieval strategy.
    keywords_matched:
        Number of domain keywords matched in the combined text.
    total_keywords:
        Total number of keywords checked.
    description_length:
        Character length of the combined description + evidence text.
    """
    quality = _compute_evidence_quality_score(documents)
    trust = _compute_evidence_trust_score(documents)
    agreement = _compute_evidence_agreement_score(evidence_items, documents)
    diversity = _compute_source_diversity_score(documents)
    density = _compute_signal_density_score(keywords_matched, total_keywords, description_length)

    weights = {
        "quality": 0.20,
        "trust": 0.25,
        "agreement": 0.20,
        "diversity": 0.15,
        "density": 0.20,
    }

    combined = (
        quality * weights["quality"]
        + trust * weights["trust"]
        + agreement * weights["agreement"]
        + diversity * weights["diversity"]
        + density * weights["density"]
    )

    return round(min(combined, 1.0), 4)


def _compute_evidence_quality_score(
    documents: list[EvidenceDocument],
) -> float:
    """Fraction of documents that have a non-None trust score.

    Documents without trust scores are treated as lower quality.
    """
    if not documents:
        return 0.0

    scored = sum(
        1 for doc in documents
        if doc.metadata is not None and doc.metadata.trust_score is not None
    )
    return scored / len(documents)


def _compute_evidence_trust_score(
    documents: list[EvidenceDocument],
) -> float:
    """Average trust score across all documents with trust scores.

    Returns 0.0 when no documents have trust scores.
    """
    if not documents:
        return 0.0

    trust_values: list[float] = []
    for doc in documents:
        if (
            doc.metadata is not None
            and doc.metadata.trust_score is not None
        ):
            trust_values.append(doc.metadata.trust_score.overall)

    if not trust_values:
        return 0.0

    return sum(trust_values) / len(trust_values)


def _compute_evidence_agreement_score(
    evidence_items: list[EvidenceItem],
    documents: list[EvidenceDocument],
) -> float:
    """Fraction of evidence items backed by at least one document.

    Items with provenance records or citations are considered
    corroborated.  Higher agreement means more items are traceable
    to source documents.
    """
    if not evidence_items:
        return 0.0

    corroborated = 0

    for item in evidence_items:
        if item.provenance_record is not None:
            corroborated += 1
        elif item.citations:
            corroborated += 1

    return corroborated / len(evidence_items)


def _compute_source_diversity_score(
    documents: list[EvidenceDocument],
) -> float:
    """Score based on the number of distinct source providers.

    More diverse sources means higher confidence.  Score saturates
    at 3 distinct providers (returns 1.0).
    """
    if not documents:
        return 0.0

    providers: set[str] = set()
    for doc in documents:
        if doc.metadata is not None and doc.metadata.source_provider:
            providers.add(doc.metadata.source_provider)

    if not providers:
        return 0.0

    return min(len(providers) / 3.0, 1.0)


def _compute_signal_density_score(
    keywords_matched: int,
    total_keywords: int,
    description_length: int,
) -> float:
    """Score based on keyword match rate and text availability.

    Combines keyword coverage with a bonus for having substantial text.
    """
    if total_keywords <= 0:
        return 0.0

    keyword_coverage = min(keywords_matched / total_keywords, 1.0)

    text_bonus = min(description_length / 500.0, 0.5) if description_length > 0 else 0.0

    combined = keyword_coverage * 0.7 + text_bonus * 0.3
    return round(min(combined, 1.0), 4)
