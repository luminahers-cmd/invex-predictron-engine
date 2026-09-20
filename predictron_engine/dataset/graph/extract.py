"""Grounded entity extraction from :class:`DatasetRecord` values.

The knowledge graph only ever contains values that literally appear in
the dataset.  This module centralizes the *metadata contract*: the exact
key names under which founders, investors, technologies, products,
organizations, acquirers, and related companies may be supplied, plus
the normalization applied to their values.

The key lists mirror the conventions established by
:mod:`predictron_engine.dataset.enrichment` (``industry_category``,
``country_code``, ...).  Keys that are absent simply contribute nothing —
no value is derived, guessed, or synthesized.
"""

from __future__ import annotations

import re
from typing import Any

from predictron_engine.dataset.enrichment import extract_domain
from predictron_engine.dataset.identity import extract_identifiers
from predictron_engine.dataset.models import DatasetRecord

# Known metadata keys carrying each entity kind.  Advisory: all present
# keys are read; duplicates are removed during normalization.
FOUNDER_KEYS: tuple[str, ...] = (
    "founders",
    "founder",
    "founder_name",
    "founder_names",
    "founders_names",
    "cofounders",
    "co_founders",
    "ceo",
)
INVESTOR_KEYS: tuple[str, ...] = (
    "investors",
    "investor",
    "investor_name",
    "investor_names",
    "lead_investor",
    "lead_investors",
    "backers",
    "funding_investors",
)
TECHNOLOGY_KEYS: tuple[str, ...] = (
    "technologies",
    "technology",
    "tech_stack",
    "technology_stack",
    "stack",
    "technologies_used",
)
PRODUCT_KEYS: tuple[str, ...] = (
    "products",
    "product",
    "product_name",
    "product_names",
)
ORGANIZATION_KEYS: tuple[str, ...] = (
    "organizations",
    "organization",
    "parent_organization",
    "parent_company",
    "parent",
    "parent_org",
)
ACQUIRER_KEYS: tuple[str, ...] = (
    "acquirer",
    "acquired_by",
    "acquiring_company",
    "acquisition_parent",
    "purchaser",
)
RELATED_KEYS: tuple[str, ...] = (
    "related_companies",
    "related",
    "competitors",
    "peers",
    "partners",
)

_PLACEHOLDER_VALUES = frozenset({"", "-", "n/a", "na", "none", "unknown", "null"})
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_entity_value(value: object) -> str:
    """Normalize a raw entity value for display while keeping meaning.

    Collapses whitespace and strips surrounding space.  Casing is
    preserved so a founder label remains human-readable; the node key is
    case-folded separately.
    """
    return _WHITESPACE_RE.sub(" ", str(value).strip())


def entity_key(label: str) -> str:
    """Return the deterministic case-folded key used in node IDs."""
    return normalize_entity_value(label).casefold()


def metadata_values(record: DatasetRecord, keys: tuple[str, ...]) -> list[str]:
    """Return the sorted distinct normalized values for ``keys``.

    Accepts scalar strings, lists, tuples, and sets.  Empty and
    placeholder values are dropped.  Never fabricates a value.
    """
    found: set[str] = set()
    metadata = record.analysis_metadata
    for key in keys:
        value = metadata.get(key)
        for candidate in _iter_values(value):
            normalized = normalize_entity_value(candidate)
            if normalized.casefold() in _PLACEHOLDER_VALUES:
                continue
            found.add(normalized)
    return sorted(found)


def founder_values(record: DatasetRecord) -> list[str]:
    return metadata_values(record, FOUNDER_KEYS)


def investor_values(record: DatasetRecord) -> list[str]:
    return metadata_values(record, INVESTOR_KEYS)


def technology_values(record: DatasetRecord) -> list[str]:
    return metadata_values(record, TECHNOLOGY_KEYS)


def product_values(record: DatasetRecord) -> list[str]:
    return metadata_values(record, PRODUCT_KEYS)


def organization_values(record: DatasetRecord) -> list[str]:
    return metadata_values(record, ORGANIZATION_KEYS)


def acquirer_values(record: DatasetRecord) -> list[str]:
    return metadata_values(record, ACQUIRER_KEYS)


def related_values(record: DatasetRecord) -> list[str]:
    return metadata_values(record, RELATED_KEYS)


def record_domain(record: DatasetRecord) -> str | None:
    """Return the canonical domain grounded in the record's profile/website."""
    if record.profile.domain:
        return record.profile.domain.strip().casefold() or None
    return extract_domain(record.website)


def record_identifiers(record: DatasetRecord) -> dict[str, str]:
    """Return grounded identifier ``kind -> value`` from record metadata.

    Delegates to the entity-resolution extractor so the graph and the
    resolver agree on what counts as a strong identifier.
    """
    return extract_identifiers(record)


def record_city(record: DatasetRecord) -> str | None:
    return _clean(record.profile.city)


def record_region(record: DatasetRecord) -> str | None:
    return _clean(record.profile.region)


def record_country(record: DatasetRecord) -> str | None:
    return _clean(record.profile.country_code)


def _iter_values(value: Any) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [item for item in value if item is not None]
    return []


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = normalize_entity_value(value)
    return cleaned or None
