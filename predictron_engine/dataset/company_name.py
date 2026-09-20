"""Company name normalization and corporate-suffix stripping (Project E2).

Entity resolution compares *canonical core names* rather than raw display
names.  This module reduces any display name to a deterministic canonical
form in four deterministic steps:

  1. Unicode normalization (NFC) and case folding
  2. corporate suffix stripping (Inc, LLC, Ltd, PLC, GmbH, BV, SA, SAS,
     Oy, AB, ...)
  3. punctuation removal and whitespace collapse
  4. tokenization against a small stopword vocabulary (for blocking)

Two entry points matter downstream:

- ``core_name`` — the suffix-stripped, punctuation-free canonical name
  used by fuzzy similarity.
- ``canonical_name_key`` — a maximal-compression exact-match key used by
  the exact-name blocking bucket.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

# Corporate-suffix phrases, longest-first so that "gmbh & co. kg" is
# matched before "gmbh".  Each entry maps the phrase to its canonical
# suffix label (not used in scoring, only kept for provenance).
_CORPORATE_SUFFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("gmbh & co. kg", "GMBH"),
    ("gmbh & co.kg", "GMBH"),
    ("gmbh & co kg", "GMBH"),
    ("incorporated", "INC"),
    ("corporation", "CORP"),
    ("limited liability company", "LLC"),
    ("limited liability partnership", "LLP"),
    ("societe anonyme", "SA"),
    ("anonima", "ANA"),
    ("aktiengesellschaft", "AG"),
    ("osakeyhtio", "OY"),
    ("o/y", "OY"),
    ("aktiebolag", "AB"),
    ("kabushiki kaisha", "KK"),
    ("i.l.c.", "LLC"),
    ("l.l.c.", "LLC"),
    ("l.l.p.", "LLP"),
    ("p.l.c.", "PLC"),
    ("p.t.y.", "PTY"),
    ("p.t.e.", "PTE"),
    ("ltd.", "LTD"),
    ("inc.", "INC"),
    ("llc", "LLC"),
    ("ltd", "LTD"),
    ("corp", "CORP"),
    ("co.", "CO"),
    ("co", "CO"),
    ("company", "CO"),
    ("gmbh", "GMBH"),
    ("b.v.", "BV"),
    ("bv", "BV"),
    ("n.v.", "NV"),
    ("nv", "NV"),
    ("plc", "PLC"),
    ("s.a.", "SA"),
    ("sa", "SA"),
    ("sas", "SAS"),
    ("sarl", "SARL"),
    ("srl", "SRL"),
    ("spa", "SPA"),
    ("oy", "OY"),
    ("oyj", "OY"),
    ("ab", "AB"),
    ("as", "AS"),
    ("ag", "AG"),
    ("kg", "KG"),
    ("llp", "LLP"),
    ("lp", "LP"),
    ("l.p.", "LP"),
    ("l.p", "LP"),
    ("inc", "INC"),
    ("llc.", "LLC"),
    ("pty", "PTY"),
    ("pte", "PTE"),
    ("pvt", "PVT"),
    ("pt", "PT"),
    ("kk", "KK"),
)

# Tokens that carry no identity signal and are dropped from fuzzy/blocking
# token sets.  Small and explicit so behavior stays explainable.
_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "co",
        "corp",
        "for",
        "incorporated",
        "inc",
        "ltd",
        "llc",
        "the",
        "of",
        "on",
        "to",
    }
)

_ALNUM_RE: Final = re.compile(r"[^0-9a-z]+")


def fold_name(name: str) -> str:
    """Unicode-normalize (NFC) and case-fold a name."""
    return unicodedata.normalize("NFC", name.strip()).casefold()


def strip_corporate_suffixes(name: str) -> tuple[str, list[str]]:
    """Strip recognized corporate suffixes from a name.

    Returns a tuple of ``(core_name, stripped_labels)``.  Suffixes are
    stripped only when they appear as whole trailing tokens.  The result
    is deterministic: "Acme Inc" -> ("acme", ["INC"]), and
    "Acme GmbH & Co. KG" -> ("acme", ["GMBH"]).
    """
    tokens: list[str] = fold_name(name).split()
    removed: list[str] = []
    changed = True
    while changed and tokens:
        changed = False
        for phrase, label in _CORPORATE_SUFFIXES:
            phrase_tokens = tuple(phrase.split())
            if len(tokens) < len(phrase_tokens):
                continue
            if " ".join(tokens[-len(phrase_tokens):]) == phrase:
                del tokens[-len(phrase_tokens):]
                removed.append(label)
                changed = True
                break
    return " ".join(tokens).strip(), removed


def core_name(name: str) -> str:
    """Return the canonical core name for comparison.

    Applies suffix stripping, then removes all non-alphanumeric
    characters and collapses whitespace.  "Acme, Inc." and
    "Acme Incorporated" both reduce to "acme".
    """
    core, _ = strip_corporate_suffixes(name)
    compact = _ALNUM_RE.sub(" ", core)
    return " ".join(compact.split())


def canonical_name_key(name: str) -> str:
    """Return a maximal-compression exact-match key.

    Removes every non-alphanumeric character so that "Acme Corp" and
    "AcmeCORP" collide.  Used for the exact-name blocking bucket.
    """
    core, _ = strip_corporate_suffixes(name)
    return _ALNUM_RE.sub("", core)


def name_tokens(name: str) -> list[str]:
    """Return the significant tokens of a name, stopwords removed."""
    tokens: list[str] = []
    seen: set[str] = set()
    for token in core_name(name).split():
        if token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens
