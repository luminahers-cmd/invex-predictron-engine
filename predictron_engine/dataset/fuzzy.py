"""Deterministic string-similarity functions (Project E2).

Provides dependency-free, pure-Python implementations of the string
similarity measures used by entity resolution.  Every function is
deterministic, side-effect free, and bounded in memory.

Implemented measures:

- Levenshtein edit distance (O(m*n) worst case, single row in memory)
- Jaro and Jaro-Winkler similarity
- Token overlap coefficient and Jaccard similarity
- ``name_similarity`` — the composite similarity used for fuzzy name
  matching, a weighted blend of Jaro-Winkler, token overlap, and
  normalized Levenshtein.

These are intentionally local and free of machine-learning so resolution
results are reproducible on every machine and in every run.
"""

from __future__ import annotations

import re
from typing import Final

_TOKEN_SPLIT_RE: Final = re.compile(r"[^0-9a-z]+")


def levenshtein(a: str, b: str) -> int:
    """Return the Levenshtein edit distance between ``a`` and ``b``.

    A single-row dynamic-programming pass uses O(min(m, n)) working
    memory.  Deterministic: equal inputs always return 0.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Keep the shorter string as the row for working-memory efficiency.
    if len(a) > len(b):
        a, b = b, a

    previous = list(range(len(a) + 1))
    for index_b, char_b in enumerate(b, start=1):
        current = [index_b]
        for index_a, char_a in enumerate(a, start=1):
            if char_a == char_b:
                substitution = previous[index_a - 1]
            else:
                substitution = previous[index_a - 1] + 1
            current.append(
                min(
                    current[index_a - 1] + 1,  # deletion
                    previous[index_a] + 1,  # insertion
                    substitution,
                )
            )
        previous = current
    return previous[-1]


def normalized_levenshtein(a: str, b: str) -> float:
    """Return Levenshtein distance normalized into [0, 1].

    1.0 means identical; 0.0 means the strings share nothing (one being
    empty while the other is non-empty maps to 0.0).
    """
    if a == b:
        return 1.0
    length = max(len(a), len(b))
    if length == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / length


def jaro(a: str, b: str) -> float:
    """Return the Jaro similarity in [0, 1]."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    if len(a) > len(b):
        a, b = b, a

    len_a, len_b = len(a), len(b)
    search_range = max(0, len_b // 2 - 1)

    matched_b = [False] * len_b
    matched_a = [False] * len_a

    matches = 0
    for i in range(len_a):
        start = max(0, i - search_range)
        end = min(i + search_range + 1, len_b)
        for j in range(start, end):
            if not matched_b[j] and a[i] == b[j]:
                matched_b[j] = True
                matched_a[i] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    ordered_b: list[str] = []
    for j, is_matched in enumerate(matched_b):
        if is_matched:
            ordered_b.append(b[j])

    mismatches = 0
    index = 0
    for i, is_matched in enumerate(matched_a):
        if is_matched:
            if a[i] != ordered_b[index]:
                mismatches += 1
            index += 1

    m = float(matches)
    transpositions = mismatches / 2.0
    return (m / len_a + m / len_b + (m - transpositions) / m) / 3.0


def jaro_winkler(a: str, b: str) -> float:
    """Return the Jaro-Winkler similarity in [0, 1].

    Boosts the Jaro score for agreement on the common prefix (up to the
    first four characters), weighting initials — which are unusually
    discriminative for company names — more heavily.
    """
    if a == b:
        return 1.0
    jaro_score = jaro(a, b)
    if jaro_score <= 0.7:
        return jaro_score
    prefix = 0
    for char_a, char_b in zip(a, b):
        if char_a != char_b:
            break
        prefix += 1
        if prefix == 4:
            break
    return jaro_score + prefix * 0.1 * (1.0 - jaro_score)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens."""
    return [token for token in _TOKEN_SPLIT_RE.split(text.lower()) if token]


def token_overlap(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Return the overlap coefficient (0..1) between two token lists.

    Uses the smaller token set as the denominator, which rewards one
    name being contained in the other (e.g. "Acme Corp" vs
    "Acme Corp LLC").
    """
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    return intersection / min(len(set_a), len(set_b))


def jaccard(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Return the Jaccard similarity (0..1) between two token lists."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def name_similarity(a: str, b: str) -> float:
    """Return the composite company-name similarity in [0, 1].

    Weights, in order of importance:

      0.5 * Jaro-Winkler similarity   — character-level agreement
      0.3 * token overlap             — shared significant words
      0.2 * normalized Levenshtein    — edit-distance fidelity

    To tolerate tokenization variants ("Data Stream" vs "Datastream",
    "Acme Rentals" vs "Acme-Rentals"), the Jaro-Winkler term uses the
    better of the space-preserving and space-compressed forms.
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    jw = max(
        jaro_winkler(a, b),
        jaro_winkler(_compact(a), _compact(b)),
    )
    overlap = token_overlap(tokenize(a), tokenize(b))
    normalized = normalized_levenshtein(a, b)
    return round(0.5 * jw + 0.3 * overlap + 0.2 * normalized, 6)


def _compact(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())
