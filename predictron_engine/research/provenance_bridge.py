"""Versioned evidence identity and the provenance bridge.

Research claims are *linked* to documents produced by the evidence
subsystem — they never become documents, and they never re-derive what
that subsystem already knows.  This module is the seam:

* **Identity versioning.**  :data:`EVIDENCE_ID_VERSION` ("1") is the
  frozen identity of an ungrounded claim, defined before document
  linkage existed.  :data:`GROUNDED_EVIDENCE_ID_VERSION` ("2") covers
  the same claim plus its grounded provenance fields.  The split exists
  so adding linkage cannot silently renumber identifiers that are
  already stored.
* **Citation linkage.**  :func:`citation_id` and
  :func:`document_ids_for` turn ``EvidenceCitation`` output into
  reference fields.  They read; they do not score.

Trust is never computed here.  A ``TrustScore`` arrives from the
evidence subsystem as a value object and is carried through untouched —
see :func:`trust_score_payload`, which serialises the object it was
handed rather than re-deriving anything from it.

Deliberately absent: any trust banding, freshness calculation,
authority estimate, or URL/fetch metadata.  Those belong to the
evidence subsystem.  Importing them here would create a second,
eventually-disagreeing opinion about the same document.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from predictron_engine.canonical import RESEARCH_DIGEST_LENGTH, content_digest
from predictron_engine.research.exceptions import InvalidEvidence

if TYPE_CHECKING:
    from predictron_engine.evidence.provenance import TrustScore
    from predictron_engine.models.report import EvidenceCitation

#: Identity version of an ungrounded claim.  Frozen: the field set this
#: version covers must never change, because stored ``evidence_id``
#: values were computed with it.
EVIDENCE_ID_VERSION = "1"

#: Identity version covering an ungrounded claim *plus* its document
#: linkage, trust reference, and citation references.
GROUNDED_EVIDENCE_ID_VERSION = "2"

#: Every identity version this release can compute.  Adding one is an
#: additive, versioned change; changing what an existing version covers
#: is not.
SUPPORTED_EVIDENCE_ID_VERSIONS: frozenset[str] = frozenset(
    {EVIDENCE_ID_VERSION, GROUNDED_EVIDENCE_ID_VERSION}
)

#: The exact fields :data:`EVIDENCE_ID_VERSION` covers.  Anything not
#: listed here is invisible to a v1 identifier — that is the whole
#: reason a second version was needed.
V1_IDENTITY_FIELDS: tuple[str, ...] = (
    "task_id",
    "topic_id",
    "collector_id",
    "category",
    "claim",
    "confidence",
    "reference",
)

#: Identity version of a citation identifier.  Pinned from the first
#: release so a future change to the citation shape cannot silently
#: renumber ids either.
CITATION_ID_VERSION = "1"

CITATION_ID_PREFIX = "cit_"
CITATION_ID_LENGTH = RESEARCH_DIGEST_LENGTH


def require_supported_version(version: str) -> str:
    """Validate an identity version, returning it stripped.

    Raises
    ------
    InvalidEvidence:
        When ``version`` is empty or not a supported version.
    """
    if not isinstance(version, str):
        raise InvalidEvidence("evidence_id_version must be a string")
    stripped = version.strip()
    if stripped not in SUPPORTED_EVIDENCE_ID_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_EVIDENCE_ID_VERSIONS))
        raise InvalidEvidence(
            f"unsupported evidence_id_version {version!r}; "
            f"supported versions: {supported}"
        )
    return stripped


def evidence_identity_payload(
    version: str,
    fields: Mapping[str, object],
) -> dict[str, object]:
    """Return the field subset that identity ``version`` actually covers.

    Parameters
    ----------
    version:
        One of :data:`SUPPORTED_EVIDENCE_ID_VERSIONS`.
    fields:
        Every candidate field of the claim, keyed by name.  Fields the
        version does not cover are dropped rather than hashed as
        defaults, so an ungrounded claim produces exactly the same v1
        payload it produced before linkage existed.

    Raises
    ------
    InvalidEvidence:
        When ``version`` is not a supported identity version.
    """
    version = require_supported_version(version)
    if version == EVIDENCE_ID_VERSION:
        return {name: fields[name] for name in V1_IDENTITY_FIELDS}
    payload = dict(fields)
    payload["evidence_id_version"] = version
    return payload


def citation_id(citation: EvidenceCitation) -> str:
    """Return the content-addressed identifier of one citation.

    The id covers the citation's own content as the evidence subsystem
    serialises it, plus :data:`CITATION_ID_VERSION`.  Delegates to
    :func:`predictron_engine.canonical.content_digest`, so citations are
    encoded by the same canonicaliser as every other identifier in the
    repository.
    """
    payload = citation.model_dump(mode="json")
    payload["citation_id_version"] = CITATION_ID_VERSION
    return content_digest(
        payload, length=CITATION_ID_LENGTH, prefix=CITATION_ID_PREFIX
    )


def citation_ids(citations: Sequence[EvidenceCitation]) -> tuple[str, ...]:
    """Return the sorted, de-duplicated ids of ``citations``.

    Citations are treated as a set of references, so identity does not
    depend on the order they happened to be collected in.
    """
    return tuple(sorted({citation_id(c) for c in citations}))


def document_ids_for(citations: Sequence[EvidenceCitation]) -> tuple[str, ...]:
    """Return every document id cited by ``citations``, sorted and unique.

    Reads ``EvidenceCitation.source_document_ids`` — the ids the
    evidence subsystem already minted — without validating, re-fetching,
    or otherwise second-guessing them.
    """
    ids: set[str] = set()
    for citation in citations:
        ids.update(citation.source_document_ids)
    return tuple(sorted(ids))


def trust_score_payload(trust_score: TrustScore | None) -> dict[str, object] | None:
    """Serialise a trust score exactly as it was imported.

    The object is dumped, never recomputed: no factor is reweighted, no
    band is assigned, no freshness term is evaluated.  ``None`` stays
    ``None`` so "no trust reference" remains distinguishable from "trust
    of zero".
    """
    if trust_score is None:
        return None
    return dict(trust_score.model_dump(mode="json"))


def normalise_link_ids(values: Sequence[str], field: str) -> tuple[str, ...]:
    """Strip, verify, de-duplicate, and sort a sequence of reference ids.

    References are sets, so ordering is normalised away: two claims
    citing the same documents in a different order carry the same
    identity.

    Raises
    ------
    InvalidEvidence:
        When an entry is not a string or is blank.
    """
    normalised: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise InvalidEvidence(f"{field} entries must be strings")
        stripped = value.strip()
        if not stripped:
            raise InvalidEvidence(f"{field} entries must not be empty")
        normalised.append(stripped)
    return tuple(sorted(dict.fromkeys(normalised)))


__all__ = [
    "CITATION_ID_LENGTH",
    "CITATION_ID_PREFIX",
    "CITATION_ID_VERSION",
    "EVIDENCE_ID_VERSION",
    "GROUNDED_EVIDENCE_ID_VERSION",
    "SUPPORTED_EVIDENCE_ID_VERSIONS",
    "V1_IDENTITY_FIELDS",
    "citation_id",
    "citation_ids",
    "document_ids_for",
    "evidence_identity_payload",
    "normalise_link_ids",
    "require_supported_version",
    "trust_score_payload",
]
