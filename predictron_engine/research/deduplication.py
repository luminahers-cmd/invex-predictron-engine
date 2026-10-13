"""Deterministic duplicate detection for research memory — Phase 8 Sprint 6.

The memory layer accumulates evidence over months of research, so the same
claim will be re-observed many times: by a re-run of the pipeline, by a
second source repeating a filing, or by a report that quotes an earlier
report.  Deduplication is what stops those repeats from inflating the graph.

Keys
----
Every duplicate decision is made from five deterministic keys, all of them
derived from the existing :class:`~predictron_engine.research.models.Evidence`
value object — nothing here re-reads a source or guesses:

===================================  ==========================================
:func:`evidence_fingerprint_key`      the content fingerprint computed by
                                     Sprint 4 (:func:`compute_fingerprint`)
:func:`evidence_source_key`           source category + source identifier +
                                     normalized URL
:func:`evidence_url_key`              the normalized URL alone
:func:`evidence_title_key`            the normalized title alone
:func:`evidence_claim_key`            topic + category + normalized claim
===================================  ==========================================

:func:`duplicate_key` folds all five into one digest, which is the identity
of an observation.  Two evidence items are duplicates when their
``duplicate_key`` matches, and :class:`DeduplicationIndex` makes that
lookup O(1) across the whole memory.

Determinism
-----------
The index keeps keys in a stable order and returns identifiers sorted, so
the same evidence added in the same order always produces the same
duplicate groupings.  Nothing here consults a clock, a network, or a model.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from predictron_engine.canonical import (
    RESEARCH_DIGEST_LENGTH,
    text_digest,
)
from predictron_engine.research.exceptions import (
    DeduplicationError,
    InvalidMemoryInputError,
)
from predictron_engine.research.linking import (
    canonical_key,
    normalize_title,
    normalize_url,
    title_key,
)
from predictron_engine.research.models import Evidence
from predictron_engine.research.validation_models import (
    EvidenceFingerprint,
    compute_fingerprint,
)

#: Serialization schema version for the deduplication index.
DEDUPLICATION_SCHEMA_VERSION = "1.0.0"

#: Prefix for the digest of an evidence fingerprint.
FINGERPRINT_KEY_PREFIX = "fp_"

#: Prefix for a source-identity key.
SOURCE_KEY_PREFIX = "src_"

#: Prefix for a normalized-URL key.
URL_KEY_PREFIX = "url_"

#: Prefix for a normalized-title key.
TITLE_KEY_PREFIX = "ttl_"

#: Prefix for a normalized-claim key.
CLAIM_KEY_PREFIX = "clm_"

#: Prefix for the combined duplicate key.
DUPLICATE_KEY_PREFIX = "dup_"

#: Separator between the parts of a composite key.
KEY_SEPARATOR = "|"


def _key(prefix: str, payload: str) -> str:
    """Return the prefixed digest of ``payload``."""
    digest = text_digest(payload, length=RESEARCH_DIGEST_LENGTH)
    return f"{prefix}{digest}"


def evidence_fingerprint_key(fingerprint: EvidenceFingerprint | str) -> str:
    """Return the deduplication key for an evidence fingerprint.

    Accepts either the
    :class:`~predictron_engine.research.validation_models.EvidenceFingerprint`
    produced by Sprint 4 or a raw digest string, so memory can key on
    fingerprints it received by either route.
    """
    value = (
        fingerprint.value
        if isinstance(fingerprint, EvidenceFingerprint)
        else fingerprint
    )
    if not isinstance(value, str) or not value.strip():
        raise DeduplicationError(
            "an evidence fingerprint is required for a fingerprint key"
        )
    return _key(
        FINGERPRINT_KEY_PREFIX,
        f"{value.strip()}",
    )


def evidence_url_key(url: str) -> str | None:
    """Return the key for a URL, or ``None`` when there is no usable URL.

    The URL is normalized by
    :func:`~predictron_engine.research.linking.normalize_url` first, so
    ``"HTTPS://WWW.Example.com/a/?b=2&a=1"`` and
    ``"https://example.com/a?a=1&b=2"`` share a key.
    """
    if not isinstance(url, str):
        raise DeduplicationError("a URL key requires a string")
    normalized = normalize_url(url)
    if not normalized:
        return None
    return _key(URL_KEY_PREFIX, normalized)


def evidence_title_key(title: str) -> str | None:
    """Return the key for a title, or ``None`` when there is none.

    Titles are normalized but never have legal suffixes or articles
    stripped, because those words are part of a headline's meaning.
    """
    if not isinstance(title, str):
        raise DeduplicationError("a title key requires a string")
    if not normalize_title(title):
        return None
    return _key(TITLE_KEY_PREFIX, title_key(title))


def evidence_source_key(evidence: Evidence) -> str | None:
    """Return the key identifying where ``evidence`` came from.

    The key combines the evidence's source category, its source
    identifier, and the normalized reference URL, so the same page fetched
    under two identifiers is still one source.  Returns ``None`` when the
    reference carries no URL at all.
    """
    reference = evidence.reference
    url = evidence_url_key(reference.url)
    if url is None:
        return None
    payload = KEY_SEPARATOR.join(
        (
            reference.source_category,
            reference.source_identifier,
            url,
        )
    )
    return _key(SOURCE_KEY_PREFIX, payload)


def evidence_claim_key(evidence: Evidence) -> str:
    """Return the key for what ``evidence`` claims, ignoring its source.

    Two collectors asserting the same fact about the same topic produce the
    same key even when they disagree on wording, punctuation, or casing.
    """
    payload = KEY_SEPARATOR.join(
        (
            evidence.topic_id,
            evidence.category,
            canonical_key(evidence.claim),
        )
    )
    return _key(CLAIM_KEY_PREFIX, payload)


@dataclass(frozen=True)
class EvidenceKeys:
    """The complete deterministic key set for one evidence item.

    Attributes
    ----------
    fingerprint:
        Key derived from the evidence's content fingerprint.
    source:
        Key derived from the source identity, or ``None`` when the
        reference carries no URL.
    url:
        Key derived from the normalized URL, or ``None``.
    title:
        Key derived from the normalized title, or ``None``.
    claim:
        Key derived from the topic, category, and normalized claim.
    duplicate:
        The combined digest: the identity of this observation.
    """

    fingerprint: str
    source: str | None
    url: str | None
    title: str | None
    claim: str
    duplicate: str

    def present(self) -> tuple[str, ...]:
        """Return the non-empty component keys, in a stable order."""
        return tuple(
            key
            for key in (
                self.fingerprint,
                self.source,
                self.url,
                self.title,
                self.claim,
            )
            if key
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "fingerprint": self.fingerprint,
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "claim": self.claim,
            "duplicate": self.duplicate,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvidenceKeys:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            fingerprint=_require_str(data, "fingerprint"),
            source=_optional_str(data, "source"),
            url=_optional_str(data, "url"),
            title=_optional_str(data, "title"),
            claim=_require_str(data, "claim"),
            duplicate=_require_str(data, "duplicate"),
        )


def evidence_keys(evidence: Evidence) -> EvidenceKeys:
    """Return every deterministic key for ``evidence``.

    This is the single place key derivation lives, so the index, the
    duplicate groups, and the memory store can never disagree about what
    makes two observations the same.
    """
    if not isinstance(evidence, Evidence):
        raise InvalidMemoryInputError(
            "evidence_keys expects an Evidence instance"
        )
    reference = evidence.reference
    fingerprint = evidence_fingerprint_key(compute_fingerprint(evidence))
    url = evidence_url_key(reference.url)
    title = (
        evidence_title_key(reference.description)
        if reference.description
        else None
    )
    source = evidence_source_key(evidence)
    claim = evidence_claim_key(evidence)
    duplicate = _key(
        DUPLICATE_KEY_PREFIX,
        KEY_SEPARATOR.join(
            (
                fingerprint,
                source or "",
                url or "",
                title or "",
                claim,
            )
        ),
    )
    return EvidenceKeys(
        fingerprint=fingerprint,
        source=source,
        url=url,
        title=title,
        claim=claim,
        duplicate=duplicate,
    )


def duplicate_key(evidence: Evidence) -> str:
    """Return the combined duplicate key for ``evidence``."""
    return evidence_keys(evidence).duplicate


@dataclass(frozen=True)
class DuplicateGroup:
    """A set of evidence identifiers that all resolve to one observation.

    ``reason`` names the key that collapsed them, so a reader can see
    *why* two items were treated as duplicates without re-deriving every
    key.
    """

    key: str
    reason: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise DeduplicationError("key must not be empty for DuplicateGroup")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise DeduplicationError(
                "reason must not be empty for DuplicateGroup"
            )
        ids = tuple(self.evidence_ids)
        if len(ids) < 2:
            raise DeduplicationError(
                "a DuplicateGroup needs at least two evidence identifiers"
            )
        if list(ids) != sorted(ids):
            raise DeduplicationError(
                "DuplicateGroup evidence ids must be sorted"
            )
        object.__setattr__(self, "key", self.key.strip())
        object.__setattr__(self, "reason", self.reason.strip())
        object.__setattr__(self, "evidence_ids", ids)

    @property
    def size(self) -> int:
        """Return how many evidence items collapsed onto this key."""
        return len(self.evidence_ids)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-ready dictionary."""
        return {
            "key": self.key,
            "reason": self.reason,
            "evidence_ids": list(self.evidence_ids),
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> DuplicateGroup:
        """Deserialize from the dictionary produced by :meth:`to_dict`."""
        return cls(
            key=_require_str(data, "key"),
            reason=_require_str(data, "reason"),
            evidence_ids=tuple(_require_str_list(data, "evidence_ids")),
        )


def find_duplicate_groups(
    evidence_items: Iterable[Evidence],
) -> tuple[DuplicateGroup, ...]:
    """Return every duplicate group in ``evidence_items``, key-sorted.

    Grouping is by the combined duplicate key, so an item is reported once
    no matter how many of its individual keys collide with others.  The
    reason reported is the first component key present on every member, in
    the fixed order fingerprint, source, URL, title, claim — deterministic
    even when several components agree.
    """
    index = DeduplicationIndex()
    for evidence in evidence_items:
        index.record(evidence)
    return index.duplicate_groups()


class DeduplicationIndex:
    """A deterministic, in-memory index of every observed evidence item.

    The index answers three questions in constant time: *have I seen this
    before?* (:meth:`is_duplicate`), *which identifier did I see it as?*
    (:meth:`existing_duplicate`), and *what else shares a key with it?*
    (:meth:`shared_with`).

    Registering an item twice is idempotent: the same evidence identifier
    is never stored twice for a key, so re-running the pipeline cannot grow
    the index.
    """

    def __init__(self) -> None:
        self._by_key: dict[str, tuple[str, ...]] = {}
        self._by_evidence: dict[str, EvidenceKeys] = {}
        self._by_id: dict[str, str] = {}

    # -- Key derivation ----------------------------------------------

    def keys_for(self, evidence: Evidence) -> EvidenceKeys:
        """Return every deterministic key for ``evidence``."""
        return evidence_keys(evidence)

    def key_for(self, evidence: Evidence) -> str:
        """Return the combined duplicate key for ``evidence``."""
        return evidence_keys(evidence).duplicate

    # -- Recording ---------------------------------------------------

    def record(self, evidence: Evidence) -> str:
        """Record ``evidence`` and return its duplicate key.

        Recording the same evidence identifier again is a no-op, which is
        what makes the index safe to feed from a replayed pipeline.
        """
        keys = evidence_keys(evidence)
        identifier = evidence.evidence_id
        known = self._by_evidence.get(identifier)
        if known is not None:
            return known.duplicate
        for key in keys.present():
            existing = self._by_key.get(key)
            if existing is None:
                self._by_key[key] = (identifier,)
            elif identifier not in existing:
                self._by_key[key] = (*existing, identifier)
        self._by_evidence[identifier] = keys
        self._by_id[identifier] = keys.duplicate
        return keys.duplicate

    def record_all(self, evidence_items: Iterable[Evidence]) -> tuple[str, ...]:
        """Record every item and return the resulting duplicate keys."""
        return tuple(self.record(evidence) for evidence in evidence_items)

    # -- Queries -----------------------------------------------------

    def is_duplicate(self, evidence: Evidence) -> bool:
        """Return whether ``evidence`` was already recorded."""
        keys = evidence_keys(evidence)
        if evidence.evidence_id in self._by_evidence:
            return True
        return any(key in self._by_key for key in keys.present())

    def existing_duplicate(self, evidence: Evidence) -> str | None:
        """Return the identifier ``evidence`` duplicates, or ``None``.

        The first-seen identifier for a matching key wins, so the answer is
        stable no matter how many later items repeat it.  The item's own
        identifier is never the answer, even when it has already been
        recorded: this is the same question :meth:`shared_with` answers, and
        asking it about a known item must not report that item as its own
        duplicate.
        """
        keys = evidence_keys(evidence)
        matches: list[str] = []
        for key in keys.present():
            for identifier in self._by_key.get(key, ()):
                if identifier != evidence.evidence_id and identifier not in matches:
                    matches.append(identifier)
        matches.sort()
        return matches[0] if matches else None

    def evidence_ids_for_key(self, key: str) -> tuple[str, ...]:
        """Return the sorted identifiers recorded under ``key``."""
        return tuple(sorted(self._by_key.get(key, ())))

    def keys_for_evidence(self, evidence_id: str) -> EvidenceKeys | None:
        """Return the recorded keys for ``evidence_id`` or ``None``."""
        return self._by_evidence.get(evidence_id)

    def shared_with(self, evidence: Evidence) -> tuple[str, ...]:
        """Return every other identifier that shares a key with ``evidence``.

        The identifier of ``evidence`` itself is excluded, so the result is
        exactly "what else have I already seen this as".
        """
        keys = evidence_keys(evidence)
        matches: set[str] = set()
        for key in keys.present():
            matches.update(self._by_key.get(key, ()))
        matches.discard(evidence.evidence_id)
        return tuple(sorted(matches))

    def duplicate_groups(self) -> tuple[DuplicateGroup, ...]:
        """Return every duplicate group, ordered by key."""
        groups: list[DuplicateGroup] = []
        for key in sorted(self._by_key):
            identifiers = self._by_key[key]
            if len(identifiers) < 2:
                continue
            groups.append(
                DuplicateGroup(
                    key=key,
                    reason=_reason_for(key),
                    evidence_ids=tuple(sorted(identifiers)),
                )
            )
        return tuple(groups)

    def duplicate_count(self) -> int:
        """Return how many identifiers repeat an observation seen earlier.

        An identifier counts once if it shares *any* key with another
        identifier, so an item that happens to collide on several keys is
        still reported a single time rather than once per key.
        """
        repeated: set[str] = set()
        for identifiers in self._by_key.values():
            if len(identifiers) < 2:
                continue
            repeated.update(identifiers)
        return len(repeated)

    def indexed_evidence_ids(self) -> tuple[str, ...]:
        """Return every recorded evidence identifier, sorted."""
        return tuple(sorted(self._by_evidence))

    def __len__(self) -> int:
        """Return how many distinct evidence items are indexed."""
        return len(self._by_evidence)

    def __contains__(self, evidence_id: object) -> bool:
        """Return whether ``evidence_id`` has been indexed."""
        return evidence_id in self._by_evidence

    # -- Serialization ------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Serialize the index into a JSON-ready dictionary.

        Every list is emitted in sorted order, so the document is byte-stable
        for a given index and a round trip restores exactly the groupings and
        duplicate groups the original reported.
        """
        return {
            "schema_version": DEDUPLICATION_SCHEMA_VERSION,
            "keys": {
                key: sorted(self._by_key[key]) for key in sorted(self._by_key)
            },
            "evidence": [
                {
                    "evidence_id": evidence_id,
                    **self._by_evidence[evidence_id].to_dict(),
                }
                for evidence_id in sorted(self._by_evidence)
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> DeduplicationIndex:
        """Rebuild an index from :meth:`to_dict` output.

        Entries are restored in sorted order, which reproduces the original
        index's groupings and its duplicate groupings exactly.
        """
        if not isinstance(data, Mapping):
            raise DeduplicationError("from_dict expects a mapping")
        keys = data.get("keys", {})
        if not isinstance(keys, Mapping):
            raise DeduplicationError("expected mapping for 'keys'")
        entries = data.get("evidence", [])
        if not isinstance(entries, list):
            raise DeduplicationError("expected list for 'evidence'")
        index = cls()
        for raw_key in sorted(keys):
            identifiers = keys[raw_key]
            if not isinstance(identifiers, list):
                raise DeduplicationError(
                    f"expected list of identifiers for '{raw_key}'"
                )
            resolved = tuple(
                item for item in identifiers if isinstance(item, str)
            )
            if not resolved:
                raise DeduplicationError(
                    f"'{raw_key}' must reference at least one identifier"
                )
            index._by_key[raw_key] = resolved
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise DeduplicationError(
                    "expected mappings for 'evidence' entries"
                )
            evidence_id = entry.get("evidence_id")
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                raise DeduplicationError(
                    "expected non-empty 'evidence_id' for an evidence entry"
                )
            resolved_keys = EvidenceKeys.from_dict(entry)
            index._by_evidence[evidence_id.strip()] = resolved_keys
            index._by_id[evidence_id.strip()] = resolved_keys.duplicate
        return index


def _reason_for(key: str) -> str:
    """Return the human-readable reason for a component key prefix."""
    if key.startswith(FINGERPRINT_KEY_PREFIX):
        return "evidence_fingerprint"
    if key.startswith(SOURCE_KEY_PREFIX):
        return "source_identifier"
    if key.startswith(URL_KEY_PREFIX):
        return "normalized_url"
    if key.startswith(TITLE_KEY_PREFIX):
        return "normalized_title"
    if key.startswith(CLAIM_KEY_PREFIX):
        return "normalized_claim"
    if key.startswith(DUPLICATE_KEY_PREFIX):
        return "combined"
    return "unknown"


def _require_str(data: Mapping[str, object], key: str) -> str:
    """Read and validate a string field."""
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DeduplicationError(f"expected non-empty string for '{key}'")
    return value


def _optional_str(data: Mapping[str, object], key: str) -> str | None:
    """Read and validate an optional string field."""
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise DeduplicationError(f"expected string or null for '{key}'")
    return value


def _require_str_list(
    data: Mapping[str, object],
    key: str,
) -> list[str]:
    """Read and validate a string-list field."""
    value = data.get(key)
    if not isinstance(value, list):
        raise DeduplicationError(f"expected list for '{key}'")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise DeduplicationError(
                f"expected list of strings for '{key}'"
            )
        result.append(item)
    return result


__all__ = [
    "CLAIM_KEY_PREFIX",
    "DEDUPLICATION_SCHEMA_VERSION",
    "DUPLICATE_KEY_PREFIX",
    "FINGERPRINT_KEY_PREFIX",
    "KEY_SEPARATOR",
    "SOURCE_KEY_PREFIX",
    "TITLE_KEY_PREFIX",
    "URL_KEY_PREFIX",
    "DeduplicationIndex",
    "DuplicateGroup",
    "EvidenceKeys",
    "duplicate_key",
    "evidence_claim_key",
    "evidence_fingerprint_key",
    "evidence_keys",
    "evidence_source_key",
    "evidence_title_key",
    "evidence_url_key",
    "find_duplicate_groups",
]
