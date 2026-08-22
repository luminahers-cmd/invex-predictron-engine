"""Shared utilities for reasoning rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem
    from predictron_engine.models.report import EvidenceCitation, Observation


def filter_evidence(
    evidence: list[EvidenceItem],
    domain: str,
) -> list[EvidenceItem]:
    """Return evidence items matching the given domain."""
    return [e for e in evidence if e.domain == domain]


def evidence_ref(item: EvidenceItem) -> str:
    """Format an evidence item as a human-readable reference string."""
    return f"evidence:{item.domain}/{item.category}: {item.statement}"


def feature_ref(field: str, value: object) -> str:
    """Format a feature field as a human-readable reference string."""
    return f"feature:{field}={value}"


def citation_ref(citation: EvidenceCitation) -> str:
    """Format a citation as a human-readable reference string.

    Returns a string like::

        "cite:industry/sales_cycle: Enterprise sales cycles are 12-18 months (source: example.com)"
    """
    return f"cite:{citation.domain}/{citation.category}: {citation.claim} ({citation.provider})"


def build_citations_for_evidence(
    evidence_items: list[EvidenceItem],
) -> list[EvidenceCitation]:
    """Build citations from evidence items that already carry citation data.

    Evidence items produced by Sprint 5B-enriched providers will have
    ``citations`` populated.  This function extracts them.
    """
    citations: list[EvidenceCitation] = []
    for item in evidence_items:
        citations.extend(item.citations)
    return citations


def attach_citations(
    observation: Observation,
    evidence_items: list[EvidenceItem],
) -> Observation:
    """Attach citations to an observation from matching evidence items.

    Matches evidence items to the observation's evidence references
    and builds citations from the matched items.  Returns a new
    Observation with citations populated.
    """
    from predictron_engine.evidence.citation import build_citations

    matched = [
        item for item in evidence_items
        if any(
            item.domain in ref and item.category in ref and item.statement in ref
            for ref in observation.evidence
            if ref.startswith("evidence:")
        )
    ]
    citations = build_citations(matched)
    return observation.model_copy(update={"citations": citations})
