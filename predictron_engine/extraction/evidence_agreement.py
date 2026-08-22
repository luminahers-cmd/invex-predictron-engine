"""Evidence agreement and conflict detection.

Provides deterministic analysis of whether evidence items corroborate
each other or contain conflicting signals.  Every function is pure:
identical inputs always produce identical outputs.

Design
------
* :func:`detect_corroboration` — find items supported by multiple sources.
* :func:`detect_conflicts` — find items with contradictory claims.
* :func:`detect_single_source_claims` — find items with only one source.
* :func:`compute_agreement_ratio` — fraction of items that are corroborated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.models.report import EvidenceItem


def detect_corroboration(
    items: list[EvidenceItem],
) -> list[CorroboratedGroup]:
    """Find evidence items that corroborate each other by category.

    Groups items by (domain, category) and returns groups with
    multiple items — indicating corroboration from different sources.

    Returns a list of CorroboratedGroup, each containing items that
    support the same claim from different sources.
    """
    groups: dict[tuple[str, str], list[EvidenceItem]] = {}
    for item in items:
        key = (item.domain, item.category)
        groups.setdefault(key, []).append(item)

    corroborated: list[CorroboratedGroup] = []
    for (domain, category), group_items in groups.items():
        if len(group_items) < 2:
            continue

        sources = list({item.source for item in group_items})
        avg_relevance = sum(i.relevance_score for i in group_items) / len(group_items)

        corroborated.append(CorroboratedGroup(
            domain=domain,
            category=category,
            items=group_items,
            source_count=len(sources),
            sources=sources,
            average_relevance=round(avg_relevance, 4),
        ))

    return sorted(
        corroborated,
        key=lambda g: g.source_count,
        reverse=True,
    )


def detect_conflicts(
    items: list[EvidenceItem],
) -> list[ConflictingGroup]:
    """Find evidence items with conflicting claims in the same category.

    Conflicts are detected when items in the same (domain, category)
    have significantly different statements.  This implementation
    detects conflicts by checking for opposing keywords within the
    same category.
    """
    groups: dict[tuple[str, str], list[EvidenceItem]] = {}
    for item in items:
        key = (item.domain, item.category)
        groups.setdefault(key, []).append(item)

    conflicts: list[ConflictingGroup] = []
    _conflict_pairs: list[tuple[str, str]] = [
        ("recurring", "transactional"),
        ("enterprise", "consumer"),
        ("b2b", "b2c"),
        ("high", "low"),
        ("strong", "weak"),
        ("growing", "declining"),
        ("emerging", "mature"),
        ("low risk", "high risk"),
    ]

    for (domain, category), group_items in groups.items():
        if len(group_items) < 2:
            continue

        for item_a in group_items:
            for item_b in group_items:
                if item_a is item_b:
                    continue

                stmt_a_lower = item_a.statement.lower()
                stmt_b_lower = item_b.statement.lower()

                for term_a, term_b in _conflict_pairs:
                    if term_a in stmt_a_lower and term_b in stmt_b_lower:
                        conflicts.append(ConflictingGroup(
                            domain=domain,
                            category=category,
                            item_a=item_a,
                            item_b=item_b,
                            conflict_type=f"{term_a}_vs_{term_b}",
                        ))
                        break

    seen: set[tuple[str, str, str]] = set()
    unique: list[ConflictingGroup] = []
    for c in conflicts:
        key = (
            c.domain,
            c.item_a.statement[:50],
            c.item_b.statement[:50],
        )
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def detect_single_source_claims(
    items: list[EvidenceItem],
) -> list[EvidenceItem]:
    """Find evidence items backed by only a single source.

    Single-source claims are less reliable than corroborated claims.
    Items with no citations or provenance are considered single-source.
    """
    single_source: list[EvidenceItem] = []

    for item in items:
        if not item.citations and item.provenance_record is None:
            single_source.append(item)
        elif len(item.citations) == 1:
            single_source.append(item)

    return single_source


def compute_agreement_ratio(
    items: list[EvidenceItem],
) -> float:
    """Compute the fraction of items that are corroborated.

    An item is considered corroborated when it has citations from
    more than one source, or when multiple items exist in the same
    (domain, category) group.

    Returns a value in [0, 1].  Returns 1.0 when no items are provided.
    """
    if not items:
        return 1.0

    groups: dict[tuple[str, str], list[EvidenceItem]] = {}
    for item in items:
        key = (item.domain, item.category)
        groups.setdefault(key, []).append(item)

    corroborated_count = 0
    for group_items in groups.values():
        sources = {i.source for i in group_items}
        if len(sources) >= 2:
            corroborated_count += len(group_items)

    return round(corroborated_count / len(items), 4) if items else 1.0


class CorroboratedGroup:
    """A group of evidence items that corroborate each other.

    Items in the same group share the same domain and category,
    providing corroboration from multiple sources.
    """

    def __init__(
        self,
        *,
        domain: str,
        category: str,
        items: list[EvidenceItem],
        source_count: int,
        sources: list[str],
        average_relevance: float,
    ) -> None:
        self.domain = domain
        self.category = category
        self.items = items
        self.source_count = source_count
        self.sources = sources
        self.average_relevance = average_relevance


class ConflictingGroup:
    """A pair of evidence items with conflicting claims.

    Represents two items in the same (domain, category) that contain
    contradictory signals.
    """

    def __init__(
        self,
        *,
        domain: str,
        category: str,
        item_a: EvidenceItem,
        item_b: EvidenceItem,
        conflict_type: str,
    ) -> None:
        self.domain = domain
        self.category = category
        self.item_a = item_a
        self.item_b = item_b
        self.conflict_type = conflict_type
