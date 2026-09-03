"""Deduplication logic (Part E).

Detects duplicate startup records using website, company name, and
normalized identifiers (e.g. CIK, company number).  Deduplication is
non-destructive: it identifies candidate duplicates and lets the
caller decide how to handle them.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field

from predictron_engine.dataset.models import DatasetRecord


@dataclass
class DeduplicationReport:
    """Report of duplicate groups found in a collection of records."""

    groups: list[list[DatasetRecord]] = field(default_factory=list)
    method_counts: dict[str, int] = field(default_factory=dict)

    @property
    def duplicate_record_count(self) -> int:
        """Total number of records participating in at least one group."""
        return sum(len(group) for group in self.groups)

    @property
    def group_count(self) -> int:
        return len(self.groups)


# Identifiers that uniquely identify a company when present in metadata.
_IDENTIFIER_METADATA_KEYS = (
    "sec_cik",
    "company_number",
    "crunchbase_url",
    "normalized_identifier",
)


def find_duplicates(
    records: list[DatasetRecord],
) -> DeduplicationReport:
    """Find duplicate records within a collection.

    Fields used for matching (in priority order):
      1. Normalized identifiers from metadata (CIK, company number, etc.)
      2. Normalized website
      3. Normalized company name

    A record is considered a duplicate of another when they share at
    least one of these normalized keys.  Records with no usable key are
    never grouped as duplicates.
    """
    report = DeduplicationReport()
    key_to_record: dict[str, DatasetRecord] = {}
    assigned: set[str] = set()
    groups: list[list[DatasetRecord]] = []

    for record in records:
        keys = _record_keys(record)
        # Find an existing group that shares a key with this record.
        group_index = None
        for key in keys:
            owner = key_to_record.get(key)
            if owner is not None:
                # Locate the group already containing the owner.
                for gi, group in enumerate(groups):
                    if any(r.record_id == owner.record_id for r in group):
                        group_index = gi
                        break
                if group_index is not None:
                    break

        if group_index is None:
            # New group.
            new_group = [record]
            groups.append(new_group)
            group_index = len(groups) - 1
            assigned.add(record.record_id)
            for key in keys:
                key_to_record.setdefault(key, record)
        else:
            # Look for the record's group; if the record already belongs,
            # do not add it twice.
            existing = groups[group_index]
            already_there = any(r.record_id == record.record_id for r in existing)
            if not already_there:
                existing.append(record)
                assigned.add(record.record_id)
            for key in keys:
                key_to_record.setdefault(key, record)

    report.groups = [g for g in groups if len(g) > 1]
    report.method_counts = _count_methods(report.groups)
    return report


def _record_keys(record: DatasetRecord) -> list[str]:
    """Return normalized identity keys for a record."""
    keys: list[str] = []

    # 1. Normalized identifiers from metadata.
    metadata = record.analysis_metadata
    for key in _IDENTIFIER_METADATA_KEYS:
        value = metadata.get(key)
        if value:
            normalized = str(value).strip().lower()
            if normalized:
                keys.append(f"id:{normalized}")

    # 2. Normalized website.
    if record.website:
        website_key = _normalize_website(record.website)
        if website_key:
            keys.append(f"site:{website_key}")

    # 3. Normalized company name.
    if record.startup_name:
        name_key = _normalize_name(record.startup_name)
        if name_key:
            keys.append(f"name:{name_key}")

    return keys


def _normalize_website(url: str) -> str:
    """Normalize a website URL for comparison.

    Strips scheme, 'www' prefix, trailing slash, query parameters, and
    fragment; lowercases the host.
    """
    value = url.strip()
    if not value:
        return ""
    if "://" not in value:
        # Try to normalize a scheme-less domain.
        value = f"http://{value}"
    try:
        parsed = urllib.parse.urlparse(value)
    except ValueError:
        return value.lower()
    host = parsed.hostname or ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/")
    return f"{host}{path}".lower()


def _normalize_name(name: str) -> str:
    """Normalize a company name for comparison.

    Lowercases, strips legal suffixes, collapses whitespace, and strips
    punctuation/spaces.
    """
    suffixes = (
        "inc", "inc.", "llc", "llc.", "l.l.c.", "corp", "corp.",
        "corporation", "ltd", "ltd.", "limited", "co", "co.", "company",
        "holdings", "group", "gmbh", "bv", "nv", "pvt", "ltd.",
    )
    value = name.strip().lower()
    # Remove legal suffixes from the end (repeatedly).
    changed = True
    while changed and value:
        changed = False
        for suffix in suffixes:
            if value.endswith(suffix):
                # Only strip when the suffix is a whole trailing token.
                prefix = value[: -len(suffix)].strip()
                if prefix and (not prefix[-1].isalnum()):
                    suffix_char = value[-len(suffix) - 1] if len(value) > len(suffix) else " "
                    if not suffix_char.isalnum():
                        value = prefix.rstrip()
                        changed = True
                break

    # Remove non-alphanumeric characters for a canonical key.
    value = re.sub(r"[^0-9a-z]+", "", value)
    return value


def _count_methods(groups: list[list[DatasetRecord]]) -> dict[str, int]:
    """Count which key type triggered each duplicate group.

    For each group, determine the first shared key type present across
    all group members and increment its counter.
    """
    counts: dict[str, int] = {"identifier": 0, "website": 0, "name": 0}
    for group in groups:
        key_sets = [set(_record_keys(r)) for r in group]
        if not key_sets:
            continue
        common = set.intersection(*key_sets)
        if any(k.startswith("id:") for k in common):
            counts["identifier"] += 1
        elif any(k.startswith("site:") for k in common):
            counts["website"] += 1
        elif any(k.startswith("name:") for k in common):
            counts["name"] += 1
    return counts


def match_record_to_store(
    record: DatasetRecord,
    existing: list[DatasetRecord],
) -> tuple[DatasetRecord | None, str]:
    """Match a single record against existing records.

    Returns (matched_record, method) where method is 'identifier',
    'website', 'name', or 'none'.

    Parameters
    ----------
    record :
        The record to match.
    existing :
        Records already present in the store.

    Returns
    -------
    A tuple of the matched existing record (or None) and the method used.
    """
    keys = _record_keys(record)
    for method_key in ("id:", "site:", "name:"):
        for key in keys:
            if not key.startswith(method_key):
                continue
            for candidate in existing:
                if _record_keys(candidate) and key in _record_keys(candidate):
                    method = {
                        "id:": "identifier",
                        "site:": "website",
                        "name:": "name",
                    }[method_key]
                    return candidate, method
    return None, "none"
