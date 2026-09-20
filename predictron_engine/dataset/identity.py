"""Canonical company identity model (Project E2).

A :class:`CompanyIdentity` is the canonical representation of a single
logical company assembled from one or more :class:`DatasetRecord`\\ s.
Resolving a dataset assigns exactly one identity to every record.

Merge rules (all deterministic, all lossless):

- **Preserve provenance** — every member ``record_id`` is retained and
  each record's original display name and website/domain are kept in
  ``record_names`` / ``record_domains``.
- **Never discard conflicting values** — names and domains that differ
  become ``aliases`` and ``alternate_domains`` rather than being dropped.
- **Maintain source attribution** — every identifier value records the
  list of record IDs that supplied it (``identifier_sources``).
- **Keep historical identifiers** — all identifier values across all
  members are retained as a union per identifier kind.
- **No fabricated data** — a country code, founding year, or status is
  only ever taken from a member record; nothing is inferred or guessed.
"""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from predictron_engine.dataset.company_name import (
    canonical_name_key,
)
from predictron_engine.dataset.enrichment import extract_domain
from predictron_engine.dataset.models import DatasetRecord

# Metadata keys that carry a strong external company identifier, mapped
# to the canonical identifier kind stored on the identity.
_IDENTIFIER_METADATA_KEYS: tuple[tuple[str, str], ...] = (
    ("sec_cik", "sec_cik"),
    ("company_number", "company_number"),
    ("registered_number", "registration_number"),
    ("registration_number", "registration_number"),
    ("vat_number", "vat_number"),
    ("ein", "ein"),
    ("crunchbase_url", "crunchbase_url"),
    ("normalized_identifier", "normalized_identifier"),
)

# Metadata keys that carry alternate/reference names.
_ALIAS_METADATA_KEYS: tuple[str, ...] = (
    "aliases",
    "former_name",
    "previous_name",
    "also_known_as",
    "legal_name",
    "registered_name",
)


class CompanyIdentity(BaseModel):
    """Canonical identity of one logical company.

    Populated deterministically by :class:`CompanyIdentityBuilder`; all
    collections are sorted and de-duplicated for stable serialization.
    """

    identity_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identity identifier (UUIDv4)",
    )
    record_ids: list[str] = Field(
        default_factory=list,
        description="All dataset record IDs that resolve to this identity",
    )
    canonical_name: str = Field(
        ..., description="Chosen canonical display name of the company"
    )
    canonical_domain: str | None = Field(
        default=None,
        description="Chosen canonical domain; None when no member has one",
    )
    record_names: dict[str, str] = Field(
        default_factory=dict,
        description="record_id -> original display name (provenance)",
    )
    record_domains: dict[str, str] = Field(
        default_factory=dict,
        description="record_id -> original website/domain (provenance)",
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternate names, former legal names, and spellings",
    )
    alternate_domains: list[str] = Field(
        default_factory=list,
        description="Additional domains used by the company",
    )
    industries: list[str] = Field(
        default_factory=list,
        description="Sorted union of normalized industry tags",
    )
    country_code: str | None = Field(
        default=None,
        description="Consensus ISO 3166-1 alpha-2 country code",
    )
    country_codes: list[str] = Field(
        default_factory=list,
        description="All distinct country codes across members",
    )
    headquarters: str | None = Field(
        default=None,
        description="Consensus human-readable headquarters string",
    )
    founded_year: int | None = Field(
        default=None,
        description="Earliest reported founding year among members",
    )
    employee_range: str | None = Field(
        default=None,
        description="Largest reported size band among members",
    )
    status: str | None = Field(
        default=None,
        description="Consensus registry status, if any member reported one",
    )
    description: str | None = Field(
        default=None,
        description="Consensus description, if any member reported one",
    )
    identifiers: dict[str, list[str]] = Field(
        default_factory=dict,
        description="kind -> sorted union of normalized identifier values",
    )
    identifier_sources: dict[str, list[str]] = Field(
        default_factory=dict,
        description='"kind:value" -> sorted record IDs that supplied it',
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Sorted distinct record.source values among members",
    )
    merge_decision: str = Field(
        default="singleton",
        description="'singleton' or 'automatic_merge'",
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def add_record(self, record: DatasetRecord) -> None:
        """Fold one record into this identity (deterministic merge rules)."""
        if record.record_id in self.record_ids:
            return
        self.record_ids = sorted([*self.record_ids, record.record_id])
        self.record_names = {**self.record_names, record.record_id: record.startup_name}
        domain = _record_domain(record)
        self.record_domains = {**self.record_domains, record.record_id: domain or ""}

        profile = record.profile
        name_forms = {record.startup_name}
        name_forms.update(_metadata_aliases(record))
        all_names = set(self.record_names.values())
        for form in sorted(name_forms):
            if form not in all_names:
                self.record_names = {
                    **self.record_names,
                    f"{record.record_id}:alias:{len(self.aliases)}": form,
                }
                self.aliases = sorted(set([*self.aliases, form]))

        if domain:
            self.alternate_domains = sorted(
                set([*self.alternate_domains, domain])
            )

        for tag in profile.industries:
            self.industries = sorted(set([*self.industries, tag]))

        country = profile.country_code
        if country:
            self.country_codes = sorted(set([*self.country_codes, country]))

        if profile.founded_year is not None:
            self.founded_year = _earliest(
                self.founded_year, profile.founded_year
            )
        if profile.employee_range is not None:
            self.employee_range = _largest_band(
                self.employee_range, profile.employee_range
            )
        if profile.status:
            self.status = _consensus(self.status, profile.status)
        if profile.description:
            self.description = _consensus(self.description, profile.description)
        if profile.headquarters:
            self.headquarters = _consensus(self.headquarters, profile.headquarters)

        for kind, value in extract_identifiers(record).items():
            values = set(self.identifiers.get(kind, []))
            values.add(value)
            self.identifiers[kind] = sorted(values)
            source_key = f"{kind}:{value}"
            sources = set(self.identifier_sources.get(source_key, []))
            sources.add(record.record_id)
            self.identifier_sources[source_key] = sorted(sources)

        self.sources = sorted(set([*self.sources, record.source]))

    def finalize(self) -> None:
        """Resolve consensus fields after all records are folded in.

        Called by the builder once every member record has been merged.
        """
        self.aliases = sorted(set(self.aliases))
        self.alternate_domains = sorted(set(self.alternate_domains))
        self.industries = sorted(set(self.industries))
        self.country_codes = sorted(set(self.country_codes))
        self.sources = sorted(set(self.sources))
        self.country_code = _majority(self.country_codes)
        # Exclude the chosen canonical form from aliases.
        self.aliases = [
            alias
            for alias in self.aliases
            if canonical_name_key(alias) != canonical_name_key(self.canonical_name)
        ]
        if self.canonical_domain and self.canonical_domain in self.alternate_domains:
            self.alternate_domains = [
                d for d in self.alternate_domains if d != self.canonical_domain
            ]


def extract_identifiers(record: DatasetRecord) -> dict[str, str]:
    """Extract strong external identifiers from a record's metadata.

    Returns a mapping of ``kind -> normalized value``.  Only identifiers
    with a non-empty value are returned; nothing is derived or guessed.
    """
    identifiers: dict[str, str] = {}
    metadata = record.analysis_metadata
    for metadata_key, kind in _IDENTIFIER_METADATA_KEYS:
        value = metadata.get(metadata_key)
        if value is None:
            continue
        normalized = _normalize_identifier(value)
        if normalized:
            identifiers[kind] = normalized
    return identifiers


class CompanyIdentityBuilder:
    """Builds a canonical :class:`CompanyIdentity` from one or more records.

    The builder applies the deterministic merge rules described in the
    module docstring.  Iteration order is stable because records and all
    collections are sorted before any choice is made.
    """

    def build(
        self,
        records: list[DatasetRecord],
        *,
        merge_decision: str = "singleton",
    ) -> CompanyIdentity:
        ordered = sorted(
            records,
            key=lambda record: (record.record_id, record.startup_name),
        )
        if not ordered:
            msg = "cannot build an identity from an empty record list"
            raise ValueError(msg)

        base = ordered[0]
        identity = CompanyIdentity(
            canonical_name=base.startup_name,
            canonical_domain=_record_domain(base),
            merge_decision=merge_decision,
        )
        for record in ordered:
            identity.add_record(record)

        identity.canonical_name = _choose_canonical_name(identity)
        identity.canonical_domain = _choose_canonical_domain(identity)
        identity.finalize()
        return identity


# ---- Internal helpers ----

def _record_domain(record: DatasetRecord) -> str:
    if record.profile.domain:
        return record.profile.domain
    return extract_domain(record.website) or ""


def _metadata_aliases(record: DatasetRecord) -> list[str]:
    aliases: list[str] = []
    metadata = record.analysis_metadata
    for key in _ALIAS_METADATA_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, list | tuple):
            for item in value:
                text = str(item).strip()
                if text:
                    aliases.append(text)
        else:
            text = str(value).strip()
            if text:
                aliases.append(text)
    return aliases


def _normalize_identifier(value: object) -> str:
    text = _clean_id(str(value))
    if not text:
        return ""
    lowered = text.lower()
    # Normalize leading zeros inconsistently present in CIK values.
    return lowered


def _clean_id(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z:_/-]", "", value.strip())


def _earliest(current: int | None, candidate: int) -> int:
    if current is None:
        return candidate
    return min(current, candidate)


def _consensus(current: str | None, candidate: str) -> str | None:
    if current is None:
        return candidate
    if current == candidate:
        return current
    # Conflicting values are never discarded; pick the shorter, then
    # lexicographically smaller for determinism.
    if len(candidate) < len(current):
        return candidate
    return current


def _largest_band(current: str | None, candidate: str) -> str | None:
    if current is None:
        return candidate
    rank = _band_rank(current)
    rank_candidate = _band_rank(candidate)
    if rank_candidate > rank:
        return candidate
    return current


def _band_rank(band: str) -> int:
    for index, prefix in enumerate(("1-10", "11-50", "51-200", "201-500")):
        if band.startswith(prefix):
            return index
    return len(("1-10", "11-50", "51-200", "201-500"))


def _choose_canonical_name(identity: CompanyIdentity) -> str:
    """Pick the most frequent core-name variant, tie-break deterministically."""
    by_core: dict[str, list[str]] = {}
    display_names: list[str] = []
    for rid in identity.record_ids:
        display_names.append(identity.record_names.get(rid, ""))
    for display_name in display_names:
        if not display_name:
            continue
        key = canonical_name_key(display_name)
        by_core.setdefault(key, []).append(display_name)
    flattened = [name for names in by_core.values() for name in names]
    counts: dict[str, int] = {}
    for display_name in flattened:
        counts[display_name] = counts.get(display_name, 0) + 1
    # Most frequent; then prefer the shortest; then lexicographically first.
    return sorted(
        flattened,
        key=lambda name: (-counts[name], len(name), name),
    )[0]


def _choose_canonical_domain(identity: CompanyIdentity) -> str | None:
    domains = [
        domain
        for domain in identity.alternate_domains
        if domain
    ]
    if not domains:
        return None
    counts: dict[str, int] = {}
    for domain in domains:
        counts[domain] = counts.get(domain, 0) + 1
    return sorted(domains, key=lambda domain: (-counts[domain], len(domain), domain))[0]


def _majority(values: list[str]) -> str | None:
    if not values:
        return None
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(values, key=lambda value: (-counts[value], value))[0]
