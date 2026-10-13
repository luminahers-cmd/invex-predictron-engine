"""Canonical serialization for every content-addressed identifier.

Every deterministic identifier in this engine — ``plan_``,
``collection_``, ``validation_``, ``evidence_``, ``analysis_``, snapshot
ids, learning fingerprints — is a SHA-256 over a *canonical* JSON
encoding of its payload.  If two call sites encode the same logical
content differently, they compute different digests for identical
content, and replay stops being able to prove anything.

This module is the single definition of that encoding.  It exists
because the codebase had eleven compact-canonical call sites speaking
three dialects at three digest widths, which made digest equality a
coincidence of payload shape rather than a guarantee.

The encoding
------------
``sort_keys=True`` so mapping order cannot leak into the digest,
``separators=(",", ":")`` so whitespace cannot either, and
``ensure_ascii=True`` so the bytes are identical on every platform and
under every locale.  That last one is not cosmetic:
``ensure_ascii=False`` produces different bytes for the same logical
content the moment a payload contains a non-ASCII character, which
silently breaks equality between two digests of the same fact.

There is deliberately **no** ``default=`` fallback.  An earlier helper
in :mod:`predictron_engine.monitoring.models` passed ``default=str``,
which meant a ``datetime`` in a payload was silently stringified into
the digest instead of raising.  A digest that quietly accepts a
lossy encoding of its input is worse than one that refuses: the
persisted bytes and the hashed bytes can then disagree with no error
anywhere.  Here, a non-JSON-native payload raises.

Digest width
------------
Truncation is a *call-site* decision, not a property of the payload,
so it stays an explicit parameter.  The repository uses three widths
(12 for reasoning trace ids, 16 for research artifacts, 64 for
monitoring and learning), and changing any of them would invalidate
stored identifiers.  Call sites keep passing what they always passed.

Compatibility
-------------
This module is additive.  Existing helpers in
:mod:`predictron_engine.monitoring.models` and
:mod:`predictron_engine.learning.models` keep their signatures and
their exact output; they delegate here.  No stored identifier changes
value as a result of this module existing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#: Digest width used by research content-addressed identifiers
#: (``plan_``, ``collection_``, ``validation_``, ``ev_``, ``analysis_``).
RESEARCH_DIGEST_LENGTH = 16

#: Digest width used by monitoring and learning fingerprints.
FULL_DIGEST_LENGTH = 64

#: Digest width used by reasoning trace node ids.
TRACE_DIGEST_LENGTH = 12

#: The canonical JSON separators.  Compact on purpose: whitespace is not
#: content, so it must not reach the digest.
_SEPARATORS = (",", ":")


def canonical_json(payload: Any) -> str:
    """Return the canonical JSON encoding of ``payload``.

    Parameters
    ----------
    payload:
        Any JSON-native value: ``dict``, ``list``, ``str``, ``int``,
        ``float``, ``bool``, or ``None``.  Mappings that are not
        ``dict`` instances must already be plain JSON types.

    Returns
    -------
    str
        Sorted-key, whitespace-free, ASCII-escaped JSON.

    Raises
    ------
    TypeError
        When ``payload`` contains a value JSON cannot represent.  This
        is intentional: a digest over a lossy rendering of its input is
        not reproducible.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=_SEPARATORS,
        ensure_ascii=True,
    )


def content_digest(
    payload: Any,
    *,
    length: int = RESEARCH_DIGEST_LENGTH,
    prefix: str = "",
) -> str:
    """Return the truncated SHA-256 digest of ``payload``'s canonical form.

    Parameters
    ----------
    payload:
        The value to digest; encoded by :func:`canonical_json`.
    length:
        Number of hex characters to keep.  Call sites pass the width
        their identifiers have always used.
    prefix:
        Optional identifier prefix, e.g. ``"plan_"``.

    Returns
    -------
    str
        ``prefix`` followed by ``length`` lowercase hex characters.
    """
    canonical = canonical_json(payload)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:length]}"


def text_digest(text: str, *, length: int = FULL_DIGEST_LENGTH) -> str:
    """Return the truncated SHA-256 digest of a raw string.

    For call sites that hash an already-canonical string rather than a
    structure.  Kept separate from :func:`content_digest` so a caller
    cannot accidentally skip canonicalization by accident — reaching
    for this one is a deliberate choice.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


__all__ = [
    "FULL_DIGEST_LENGTH",
    "RESEARCH_DIGEST_LENGTH",
    "TRACE_DIGEST_LENGTH",
    "canonical_json",
    "content_digest",
    "text_digest",
]
