"""Deterministic entity linking and name normalization — Phase 8 Sprint 6.

Entity resolution here is **rule-based, not fuzzy**.  There is no edit
distance, no embedding, no model call, and no randomness: every alias is
mapped onto a single canonical form by a documented pipeline of pure
string transformations, so ``"Open AI"``, ``"OpenAI"``, and
``"OpenAI Inc."`` all reduce to the same key and therefore resolve to the
same graph entity.

The same normalization rules apply uniformly to companies, founders,
products, technologies, and investors — resolution never depends on the
entity type, only on the text.

Normalization pipeline
----------------------
Applied strictly in this order:

1. :func:`strip_accents` — fold accented characters to their ASCII base so
   ``"José"`` and ``"Jose"`` agree.
2. :func:`normalize_whitespace` — casefold and collapse every whitespace run
   (including non-breaking spaces) to a single space.
3. :func:`strip_directional_noise` — delete zero-width and bidi characters
   and turn decorative punctuation (apostrophes, quotes, dashes,
   ampersands, slashes, plus signs, and every period) into word
   separators.
4. :func:`strip_legal_suffixes` — drop trailing legal-form tokens.
5. :func:`fuse_tokens` — fuse the ``"a"``/``"i"`` initialism and a trailing
   domain acronym onto the token before it.
6. :func:`strip_boundaries` — drop leading/trailing articles and
   conjunctions.

Deliberate non-merge rules
--------------------------
Some differences are *meaningful* in venture research and must never be
erased by a normalization rule:

* the domain token ``"ai"`` is *fused*, never dropped: ``"a"`` + ``"i"``
  collapses to ``"ai"`` so ``"A.I."`` and ``"AI"`` agree, and a trailing
  ``"ai"`` fuses onto the token before it so ``"Open AI"`` and ``"OpenAI"``
  agree while ``"Acme AI Labs"`` keeps both tokens;
* the domain token is therefore *not* a legal form — it never appears in
  :data:`LEGAL_SUFFIX_TOKENS`, so it can never be stripped;
* ``"and"`` is only dropped at the *edges* of a name, so ``"The Hertz
  Corporation"`` becomes ``"hertz"`` while ``"Sand and Gravel"`` keeps its
  interior conjunction;
* a name that consists only of legal tokens is reduced to a single token
  rather than emptied, so normalization never returns ``""`` for a
  non-empty input.

Canonical keys
--------------
:func:`canonical_key` is the pure function every identifier is derived
from, and it embeds :data:`LINKING_VERSION` in its digest.  :func:`entity_id`
additionally embeds the entity type, so ``"OpenAI"`` is the company
``ent_company_<digest>`` and never collides with a product that happens to
share the name.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from urllib.parse import urlsplit, urlunsplit

from predictron_engine.canonical import RESEARCH_DIGEST_LENGTH, text_digest

#: Version of the normalization ruleset.  Bump this only when a rule
#: changes, because the value participates in the canonical-key digest and
#: therefore in every stable entity identifier.
LINKING_VERSION = "1.0.0"

#: Fixed version segment of every canonical key.
IDENTIFIER_VERSION = "v1"

#: Prefix applied to canonical entity keys, before the digest.
ENTITY_ID_PREFIX = "ent_"

#: Separator between an entity type and the digest in an entity id.
ENTITY_ID_SEPARATOR = "_"

#: Separator between an alias and the digest of its canonical key.
ALIAS_ID_SEPARATOR = "_"

#: Trailing legal-form tokens removed during normalization.
#:
#: Multi-word forms need no special handling: :func:`strip_legal_suffixes`
#: removes trailing tokens one at a time, so ``"Acme S.A.R.L."`` and
#: ``"Acme SARL"`` both reduce to ``"acme"``.  The entries ``"l l c"`` and
#: ``"s a r l"`` are reachable because :func:`strip_directional_noise` turns
#: every period into a separator, turning ``"L.L.C."`` and ``"S.A.R.L."`` into
#: their spaced-out spellings.  Words that are part of a company's real name
#: -- ``holdings``, ``group`` -- are deliberately absent: dropping them would
#: merge genuinely different firms.
LEGAL_SUFFIX_TOKENS: tuple[str, ...] = (
    "ag",
    "aps",
    "ab",
    "bv",
    "co",
    "corp",
    "corporation",
    "gmbh",
    "inc",
    "incorporated",
    "kk",
    "l l c",
    "limited",
    "limited liability company",
    "limited liability partnership",
    "llc",
    "ltd",
    "nv",
    "oy",
    "plc",
    "professional corporation",
    "pte",
    "pty",
    "pvt",
    "sarl",
    "s a r l",
    "sas",
    "sa",
    "srl",
    "venture",
    "ventures",
)

#: Adjacent token pairs fused into a single token, keyed as
#: ``"left|right"``.  The only entry fuses the ``"A.I."`` initialism so it
#: agrees with the plain ``"AI"`` spelling.
INITIALISM_FUSIONS: Mapping[str, str] = MappingProxyType({"a|i": "ai"})

#: Trailing domain acronyms fused onto the token before them.  This is what
#: makes ``"Open AI"`` resolve to the same entity as ``"OpenAI"`` without
#: stripping the token and losing a real part of the name.
TRAILING_ACRONYM_FUSIONS: frozenset[str] = frozenset({"ai"})

#: Edge tokens removed before suffix stripping: pure articles and a small
#: closed set of conjunctions.
EDGE_TOKENS: frozenset[str] = frozenset({"a", "an", "and", "the"})

#: Characters treated as word separators during tokenization.
_DECORATIVE_PUNCTUATION: frozenset[str] = frozenset(
    set("'\"`´’‘“”-/\\|+&*_~^<>()[]{}#!?;:,=")
    | {"."}
)

#: Whitespace runs, including non-breaking and zero-width spaces.
_WHITESPACE_RUN = re.compile(r"\s+")

#: Unicode format characters (zero-width joiners, bidi marks, …).
_FORMAT_CHARACTERS: frozenset[int] = frozenset(
    {0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0xFEFF, 0x2060}
)

#: Lookup form of :data:`LEGAL_SUFFIX_TOKENS`.
_LEGAL_SUFFIX_SET: frozenset[str] = frozenset(LEGAL_SUFFIX_TOKENS)

#: Width of the widest multi-token entry, so the longest possible trailing
#: phrase is tried before a shorter one.
_LEGAL_SUFFIX_MAX_WORDS: int = max(
    len(token.split()) for token in LEGAL_SUFFIX_TOKENS
)


def strip_accents(value: str) -> str:
    """Fold accented characters onto their ASCII base letters.

    ``"José"`` becomes ``"Jose"`` and ``"Ångström"`` becomes
    ``"Angstrom"``.  Characters with no ASCII decomposition are returned
    unchanged.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )


def normalize_whitespace(value: str) -> str:
    """Casefold ``value`` and collapse every whitespace run to one space."""
    return _WHITESPACE_RUN.sub(" ", value.casefold()).strip()


def strip_directional_noise(value: str) -> str:
    """Delete format characters and split ``value`` on decorative punctuation.

    Zero-width and bidi characters are removed outright.  Apostrophes,
    quotes, dashes, ampersands, slashes, plus signs, brackets, and every
    period become spaces, so ``"J. R. Smith"``, ``"J R Smith"``, and
    ``"J.R.Smith"`` all reduce to ``"j r smith"``.
    """
    without_format = "".join(
        char for char in value if ord(char) not in _FORMAT_CHARACTERS
    )
    separated = "".join(
        " " if char in _DECORATIVE_PUNCTUATION else char
        for char in without_format
    )
    return collapse_spaces(separated)


def strip_legal_suffixes(value: str) -> str:
    """Remove trailing legal-form tokens from an already-normalized name.

    Matching is exact-token and greedy, so ``"Acme Ltd Inc."`` reduces to
    ``"acme"``.  The longest multi-token entry is tried first, which is what
    lets the spaced-out spellings produced from dotted forms -- ``"L.L.C."``
    becoming ``"l l c"`` -- reduce the same way as their plain spellings.  A
    name that consists only of legal tokens is reduced to a single token
    rather than emptied, so normalization never returns an empty string for a
    non-empty input.
    """
    tokens = value.split()
    while len(tokens) > 1:
        matched = 0
        for width in range(min(_LEGAL_SUFFIX_MAX_WORDS, len(tokens) - 1), 0, -1):
            if " ".join(tokens[-width:]) in _LEGAL_SUFFIX_SET:
                matched = width
                break
        if not matched:
            break
        del tokens[-matched:]
    return " ".join(tokens)


def fuse_tokens(value: str) -> str:
    """Fuse acronym tokens onto their neighbours.

    Two documented, deterministic fusions are applied:

    * adjacent ``"a"`` and ``"i"`` become ``"ai"`` (see
      :data:`INITIALISM_FUSIONS`), so ``"A.I. Labs"`` and ``"AI Labs"``
      agree;
    * a trailing acronym listed in :data:`TRAILING_ACRONYM_FUSIONS` is
      concatenated onto the token before it, so ``"Open AI"``, ``"OpenAI"``,
      and ``"Open AI Inc."`` all agree.

    Fusing never deletes a token, so a domain acronym can never be lost the
    way a legal suffix is.  Interior acronyms are left alone, which keeps
    ``"Acme AI Labs"`` distinct from ``"Acme Labs"``.
    """
    tokens = value.split()
    if not tokens:
        return ""
    index = 0
    while index + 1 < len(tokens):
        fused = INITIALISM_FUSIONS.get(f"{tokens[index]}|{tokens[index + 1]}")
        if fused is not None:
            tokens[index : index + 2] = [fused]
            continue
        index += 1
    if len(tokens) > 1 and tokens[-1] in TRAILING_ACRONYM_FUSIONS:
        tokens[-2] = f"{tokens[-2]}{tokens[-1]}"
        tokens.pop()
    return " ".join(tokens)


def strip_boundaries(value: str) -> str:
    """Drop leading and trailing articles and conjunctions.

    Interior conjunctions are preserved: ``"Sand and Gravel"`` keeps
    ``"and"`` while ``"The Hertz Corporation"`` becomes ``"hertz"``.
    """
    tokens = value.split()
    while len(tokens) > 1 and tokens[0] in EDGE_TOKENS:
        tokens = tokens[1:]
    while len(tokens) > 1 and tokens[-1] in EDGE_TOKENS:
        tokens = tokens[:-1]
    return " ".join(tokens)


def collapse_spaces(value: str) -> str:
    """Collapse whitespace runs and strip the boundaries."""
    return _WHITESPACE_RUN.sub(" ", value).strip()


def normalize_name(value: str) -> str:
    """Return the canonical form of an entity name.

    The result is empty only when ``value`` is empty or consists solely of
    punctuation and whitespace.  The transformation is total,
    deterministic, and idempotent:
    ``normalize_name(normalize_name(x)) == normalize_name(x)``.
    """
    if not isinstance(value, str):
        raise TypeError("normalize_name expects a string")
    folded = strip_accents(value)
    folded = normalize_whitespace(folded)
    folded = strip_directional_noise(folded)
    folded = strip_legal_suffixes(folded)
    folded = fuse_tokens(folded)
    folded = strip_boundaries(folded)
    return collapse_spaces(folded)


def canonical_key(value: str) -> str:
    """Return the stable canonical key for ``value``.

    Two names resolve to the same entity exactly when their canonical keys
    are equal.  The key embeds the normalization ruleset version, so a rule
    change can never silently collide with a previously stored identifier.
    """
    normalized = normalize_name(value)
    if not normalized:
        raise ValueError("cannot derive a canonical key from an empty name")
    digest = text_digest(
        f"{LINKING_VERSION}\x1f{normalized}",
        length=RESEARCH_DIGEST_LENGTH,
    )
    return f"{IDENTIFIER_VERSION}_{digest}"


def entity_id(entity_type: str, value: str) -> str:
    """Return the stable identifier for an entity of ``entity_type``.

    The type is part of the digest payload, so a company and a product that
    share a name never share an identifier.
    """
    payload_type = _validated_type(entity_type)
    digest = text_digest(
        f"{LINKING_VERSION}\x1f{payload_type}\x1f{canonical_key(value)}",
        length=RESEARCH_DIGEST_LENGTH,
    )
    return f"{ENTITY_ID_PREFIX}{payload_type}{ENTITY_ID_SEPARATOR}{digest}"


def alias_id(value: str) -> str:
    """Return the stable identifier for an alias of ``value``."""
    digest = text_digest(
        f"{LINKING_VERSION}\x1f{canonical_key(value)}",
        length=RESEARCH_DIGEST_LENGTH,
    )
    return f"alias{ALIAS_ID_SEPARATOR}{digest}"


def normalize_url(value: str) -> str:
    """Return the canonical form of a URL for duplicate detection.

    Normalization is purely structural: the scheme and host are
    lower-cased, a leading ``www.`` and default ports are dropped, a
    trailing slash is removed, query parameters are sorted, and the
    fragment is discarded.  No request is issued and no host is resolved.
    """
    if not isinstance(value, str):
        raise TypeError("normalize_url expects a string")
    candidate = value.strip()
    if not candidate:
        return ""
    parts = urlsplit(candidate)
    if not parts.scheme and not parts.netloc:
        return collapse_spaces(normalize_whitespace(candidate))
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    for suffix, matching_scheme in ((":80", "http"), (":443", "https")):
        if scheme == matching_scheme and netloc.endswith(suffix):
            netloc = netloc[: -len(suffix)]
            break
    if netloc.startswith("www."):
        netloc = netloc[len("www.") :]
    path = parts.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, _canonical_query(parts.query), ""))


def normalize_title(value: str) -> str:
    """Return the canonical form of a title for duplicate detection.

    Titles normalize like names but never drop legal suffixes, articles, or
    conjunctions, because those words carry meaning in a headline.
    """
    if not isinstance(value, str):
        raise TypeError("normalize_title expects a string")
    folded = strip_accents(value)
    folded = normalize_whitespace(folded)
    folded = strip_directional_noise(folded)
    return collapse_spaces(folded)


def title_key(value: str) -> str:
    """Return the stable digest identifying a normalized title."""
    normalized = normalize_title(value)
    if not normalized:
        raise ValueError("cannot derive a title key from an empty title")
    digest = text_digest(
        f"{LINKING_VERSION}\x1f{normalized}",
        length=RESEARCH_DIGEST_LENGTH,
    )
    return f"{IDENTIFIER_VERSION}_t_{digest}"


class EntityLinker:
    """Deterministic, type-aware entity resolution over name variants.

    The linker keeps a ``(entity_type, canonical key)`` index and, for every
    resolved entity, the ordered set of variants that collapsed onto it.
    Two guarantees matter to the rest of the layer:

    * **Content-derived identifiers** — :meth:`resolve` returns
      :func:`entity_id`, a pure function of the type and the name, so two
      linkers fed the same names agree without any coordination.
    * **Idempotence** — resolving a name that is already known returns the
      existing identifier and records no new alias.
    """

    def __init__(self) -> None:
        self._identities: dict[tuple[str, str], str] = {}
        self._aliases: dict[str, tuple[str, ...]] = {}
        self._types: dict[str, str] = {}

    def resolve(self, entity_type: str, name: str) -> str:
        """Return the identifier for ``name`` within ``entity_type``.

        Repeated calls with variants of the same name return the same
        identifier.  An empty or punctuation-only name raises
        :class:`ValueError` rather than minting an unusable identifier.
        """
        payload_type = _validated_type(entity_type)
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        variant = name.strip()
        if not normalize_name(variant):
            raise ValueError("name must contain at least one usable character")
        index = (payload_type, canonical_key(variant))
        identifier = self._identities.get(index)
        if identifier is None:
            identifier = entity_id(payload_type, variant)
            self._identities[index] = identifier
            self._aliases[identifier] = (variant,)
            self._types[identifier] = payload_type
        else:
            known = self._aliases[identifier]
            if variant not in known:
                self._aliases[identifier] = known + (variant,)
        return identifier

    def peek(self, entity_type: str, name: str) -> str | None:
        """Return the identifier for ``name`` without recording an alias.

        Returns ``None`` when the name has not been resolved, and never
        raises for a noise-only name.
        """
        payload_type = _validated_type(entity_type)
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        normalized = normalize_name(name)
        if not normalized:
            return None
        return self._identities.get((payload_type, canonical_key(name)))

    def is_known(self, entity_type: str, name: str) -> bool:
        """Return whether ``name`` has already been resolved."""
        return self.peek(entity_type, name) is not None

    def aliases_for(self, identifier: str) -> tuple[str, ...]:
        """Return the recorded name variants for ``identifier``.

        Variants are stored in first-seen order, which is deterministic
        because resolution is a pure function of the call order a caller
        chose.
        """
        return self._aliases.get(identifier, ())

    def type_for(self, identifier: str) -> str | None:
        """Return the entity type recorded for ``identifier`` or ``None``."""
        return self._types.get(identifier)

    def known_names(self) -> tuple[str, ...]:
        """Return every resolved canonical name, sorted."""
        return tuple(sorted(key for _type, key in self._identities))

    def identifiers(self) -> tuple[str, ...]:
        """Return every resolved identifier, sorted."""
        return tuple(sorted(self._identities.values()))

    def __len__(self) -> int:
        """Return how many distinct entities have been resolved."""
        return len(self._identities)

    def to_dict(self) -> dict[str, object]:
        """Serialize the resolved aliases into a JSON-ready dictionary."""
        return {
            "version": LINKING_VERSION,
            "entities": [
                {
                    "entity_id": identifier,
                    "entity_type": self._types.get(identifier, ""),
                    "aliases": list(self._aliases.get(identifier, ())),
                }
                for identifier in sorted(self._identities.values())
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EntityLinker:
        """Rebuild a linker from :meth:`to_dict` output.

        Because identifiers are content-derived, a restored linker resolves
        the recorded aliases to exactly the same identifiers the original
        produced; a payload whose aliases do not reproduce their stored
        identifier is rejected.
        """
        if not isinstance(data, Mapping):
            raise TypeError("from_dict expects a mapping")
        entries = data.get("entities", [])
        if not isinstance(entries, list):
            raise ValueError("expected list for 'entities'")
        linker = cls()
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("expected list of mappings for 'entities'")
            stored_id = entry.get("entity_id", "")
            entity_type = entry.get("entity_type", "")
            aliases = entry.get("aliases", [])
            if not isinstance(stored_id, str):
                raise ValueError("expected string for 'entity_id'")
            if not isinstance(entity_type, str):
                raise ValueError("expected string for 'entity_type'")
            if not isinstance(aliases, list):
                raise ValueError("expected list of strings for 'aliases'")
            variants = tuple(
                alias
                for alias in aliases
                if isinstance(alias, str) and alias.strip()
            )
            if not variants:
                continue
            if not entity_type.strip():
                raise ValueError(
                    "entity_type must not be empty for a stored entity"
                )
            for variant in variants:
                if linker.resolve(entity_type, variant) != stored_id:
                    raise ValueError(
                        "alias does not reproduce its stored entity_id: "
                        f"{variant!r}"
                    )
        return linker


def resolve_aliases(entity_type: str, names: Iterable[str]) -> tuple[str, ...]:
    """Resolve ``names`` to the distinct identifiers, in first-seen order.

    Convenience wrapper for resolving a batch of variants that are expected
    to describe the same entity.
    """
    linker = EntityLinker()
    resolved: list[str] = []
    for name in names:
        identifier = linker.resolve(entity_type, name)
        if identifier not in resolved:
            resolved.append(identifier)
    return tuple(resolved)


def _validated_type(entity_type: str) -> str:
    """Validate and lower-case an entity-type label."""
    if not isinstance(entity_type, str) or not entity_type.strip():
        raise ValueError("entity_type must be a non-empty string")
    return entity_type.strip().lower()


def _canonical_query(query: str) -> str:
    """Return ``query`` with its parameters in deterministic order."""
    if not query:
        return ""
    return "&".join(sorted(pair for pair in query.split("&") if pair))


__all__ = [
    "ALIAS_ID_SEPARATOR",
    "EDGE_TOKENS",
    "ENTITY_ID_PREFIX",
    "ENTITY_ID_SEPARATOR",
    "IDENTIFIER_VERSION",
    "INITIALISM_FUSIONS",
    "LEGAL_SUFFIX_TOKENS",
    "LINKING_VERSION",
    "TRAILING_ACRONYM_FUSIONS",
    "EntityLinker",
    "alias_id",
    "canonical_key",
    "collapse_spaces",
    "entity_id",
    "fuse_tokens",
    "normalize_name",
    "normalize_title",
    "normalize_url",
    "normalize_whitespace",
    "resolve_aliases",
    "strip_accents",
    "strip_boundaries",
    "strip_directional_noise",
    "strip_legal_suffixes",
    "title_key",
]
