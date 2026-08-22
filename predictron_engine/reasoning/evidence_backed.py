"""Evidence-backed observation enrichment.

Sprint 6A requires every Observation to carry deterministic
evidence-aware metadata:

* ``confidence``                 — set by the producing rule
* ``trust_score``                — best trust among cited documents
* ``citations``                  — structured supporting citations
* ``provenance_document_ids``    — document ids enabling full traceability
* ``evidence_agreement_ratio``   — corroboration ratio of supporting items
* ``evidence_conflict_count``    — conflicting signals among support

:func:`enrich_observation` populates these fields deterministically from
the observation's own evidence references and the evidence pool.
Enrichment never mutates its inputs — it returns an updated copy.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from predictron_engine.reasoning.consistency import _ref_matches_item

if TYPE_CHECKING:
    from predictron_engine.evidence.models import EvidenceDocument
    from predictron_engine.models.report import EvidenceCitation, EvidenceItem, Observation


def match_evidence_for_observation(
    observation: Observation,
    evidence_items: list[EvidenceItem],
) -> list[EvidenceItem]:
    """Resolve an observation's ``evidence:`` references to actual items.

    Matching is deterministic: references are processed in stored order
    and each reference matches at most one pool entry per item, with
    duplicates removed while preserving first-seen order.
    """
    matched: list[EvidenceItem] = []
    for ref in observation.evidence:
        for item in evidence_items:
            if _ref_matches_item(ref, item) and item not in matched:
                matched.append(item)
    return matched


def enrich_observation(
    observation: Observation,
    evidence_items: list[EvidenceItem],
    documents: list[EvidenceDocument] | None = None,
) -> Observation:
    """Populate evidence-backed metadata on an observation.

    All values are computed deterministically:

    * ``trust_score`` — highest trust score among the observation's
      citations; falls back to the highest provenance-record trust of
      matched evidence items; 0.0 when neither exists.
    * ``provenance_document_ids`` — ordered, deduplicated union of
      citation source ids and matched-item provenance record ids.
    * ``evidence_agreement_ratio`` — fraction of matched evidence items
      corroborated by ≥2 distinct sources within their (domain, category)
      group; 0.0 when nothing is matched.
    * ``evidence_conflict_count`` — number of conflicting signal pairs
      detected among matched items.
    * ``citations`` — existing citations are preserved; when absent,
      citations carried by matched evidence items are adopted in order.
    """
    matched = match_evidence_for_observation(observation, evidence_items)

    citations = _resolve_citations(observation, matched)
    trust_score = _compute_trust_score(citations, matched)
    provenance_ids = _collect_provenance_ids(citations, matched)
    agreement = _compute_agreement_ratio(matched)
    conflicts = _count_conflicts(matched)

    return observation.model_copy(
        update={
            "citations": citations,
            "trust_score": trust_score,
            "provenance_document_ids": provenance_ids,
            "evidence_agreement_ratio": agreement,
            "evidence_conflict_count": conflicts,
        }
    )


# ────────────────────────────────────────────────────────────────────
# Private helpers
# ────────────────────────────────────────────────────────────────────


def _resolve_citations(
    observation: Observation,
    matched: list[EvidenceItem],
) -> list[EvidenceCitation]:
    """Keep existing citations; adopt matched-item citations otherwise."""
    if observation.citations:
        return list(observation.citations)

    citations: list[EvidenceCitation] = []
    seen: set[tuple[str, str]] = set()
    for item in matched:
        for citation in item.citations:
            key = (citation.claim, citation.domain)
            if key not in seen:
                seen.add(key)
                citations.append(citation)
    return citations


def _compute_trust_score(
    citations: list[EvidenceCitation],
    matched: list[EvidenceItem],
) -> float:
    """Best available trust: citation trust first, then provenance records."""
    best = 0.0
    for citation in citations:
        if citation.best_trust_score > best:
            best = citation.best_trust_score
    if best > 0.0:
        return round(best, 4)

    for item in matched:
        record = _parse_provenance(item)
        if record is not None:
            value = float(record.get("trust_score", 0.0) or 0.0)
            if value > best:
                best = value
    return round(best, 4)


def _collect_provenance_ids(
    citations: list[EvidenceCitation],
    matched: list[EvidenceItem],
) -> list[str]:
    """Ordered, deduplicated document ids from citations and provenance."""
    ids: list[str] = []
    for citation in citations:
        for doc_id in citation.source_document_ids:
            if doc_id not in ids:
                ids.append(doc_id)
    for item in matched:
        record = _parse_provenance(item)
        if record is None:
            continue
        doc_id = record.get("document_id")
        if isinstance(doc_id, str) and doc_id and doc_id not in ids:
            ids.append(doc_id)
    return ids


def _compute_agreement_ratio(matched: list[EvidenceItem]) -> float:
    """Fraction of matched items corroborated by ≥2 distinct sources."""
    if not matched:
        return 0.0

    groups: dict[tuple[str, str], set[str]] = {}
    for item in matched:
        groups.setdefault((item.domain, item.category), set()).add(item.source)

    corroborated = sum(
        1 for item in matched if len(groups[(item.domain, item.category)]) >= 2
    )
    return round(corroborated / len(matched), 4)


def _count_conflicts(matched: list[EvidenceItem]) -> int:
    """Count conflicting signal pairs among matched evidence items."""
    if len(matched) < 2:
        return 0
    from predictron_engine.extraction.evidence_agreement import detect_conflicts

    return len(detect_conflicts(matched))


def _parse_provenance(item: EvidenceItem) -> dict | None:
    """Parse the JSON-encoded provenance record on an evidence item."""
    encoded = item.provenance_record
    if not encoded:
        return None
    try:
        parsed = json.loads(encoded)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None
