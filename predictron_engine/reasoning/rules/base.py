"""Shared utilities for reasoning rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.evidence.evidence_models import EvidenceItem


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
