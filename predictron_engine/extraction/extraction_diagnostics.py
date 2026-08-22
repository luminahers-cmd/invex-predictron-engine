"""Structured extraction diagnostics.

Provides deterministic diagnostic data structures that record what
happened during evidence-aware feature extraction — documents examined,
selected, trust scores, citation counts, conflicts, and confidence.

Design
------
* :class:`ExtractionDiagnostic` — per-extractor diagnostic record.
* :func:`build_diagnostic` — factory function to create diagnostics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.evidence.models import EvidenceDocument


class ExtractionDiagnostic:
    """Structured diagnostics for a single extractor's evidence-aware run.

    Records the complete picture of what evidence was available, what
    was selected, and what outcomes resulted.  Every field is populated
    deterministically from the extraction run.
    """

    def __init__(
        self,
        *,
        extractor_name: str,
        documents_examined: int,
        documents_selected: int,
        average_trust_score: float,
        citation_count: int,
        conflict_count: int,
        corroborated_count: int,
        single_source_count: int,
        evidence_confidence: float,
        selected_document_ids: list[str] | None = None,
        keywords_matched: int = 0,
        total_keywords: int = 0,
    ) -> None:
        self.extractor_name = extractor_name
        self.documents_examined = documents_examined
        self.documents_selected = documents_selected
        self.average_trust_score = average_trust_score
        self.citation_count = citation_count
        self.conflict_count = conflict_count
        self.corroborated_count = corroborated_count
        self.single_source_count = single_source_count
        self.evidence_confidence = evidence_confidence
        self.selected_document_ids = selected_document_ids or []
        self.keywords_matched = keywords_matched
        self.total_keywords = total_keywords


def build_diagnostic(
    *,
    extractor_name: str,
    documents_examined: list[EvidenceDocument],
    documents_selected: list[EvidenceDocument],
    citation_count: int,
    conflict_count: int,
    corroborated_count: int,
    single_source_count: int,
    evidence_confidence: float,
    keywords_matched: int = 0,
    total_keywords: int = 0,
) -> ExtractionDiagnostic:
    """Build an ExtractionDiagnostic from extraction run data.

    Pure factory function — no side effects.
    """
    avg_trust = _compute_average_trust(documents_selected)

    return ExtractionDiagnostic(
        extractor_name=extractor_name,
        documents_examined=len(documents_examined),
        documents_selected=len(documents_selected),
        average_trust_score=avg_trust,
        citation_count=citation_count,
        conflict_count=conflict_count,
        corroborated_count=corroborated_count,
        single_source_count=single_source_count,
        evidence_confidence=evidence_confidence,
        selected_document_ids=[doc.id for doc in documents_selected],
        keywords_matched=keywords_matched,
        total_keywords=total_keywords,
    )


def _compute_average_trust(documents: list[EvidenceDocument]) -> float:
    """Compute the average trust score across documents with trust scores."""
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

    return round(sum(trust_values) / len(trust_values), 4)
