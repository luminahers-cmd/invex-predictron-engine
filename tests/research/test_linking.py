"""Tests for deterministic entity linking — Phase 8 Sprint 6.

Covers name normalization, canonical keys, stable identifiers, URL and title
normalization, and the alias-resolving :class:`EntityLinker`.
"""

from __future__ import annotations

import pytest

from predictron_engine.research.linking import (
    ALIAS_ID_SEPARATOR,
    EDGE_TOKENS,
    ENTITY_ID_PREFIX,
    ENTITY_ID_SEPARATOR,
    IDENTIFIER_VERSION,
    INITIALISM_FUSIONS,
    LEGAL_SUFFIX_TOKENS,
    LINKING_VERSION,
    TRAILING_ACRONYM_FUSIONS,
    EntityLinker,
    alias_id,
    canonical_key,
    collapse_spaces,
    entity_id,
    fuse_tokens,
    normalize_name,
    normalize_title,
    normalize_url,
    normalize_whitespace,
    resolve_aliases,
    strip_accents,
    strip_boundaries,
    strip_directional_noise,
    strip_legal_suffixes,
    title_key,
)


def test_normalize_name_casefolds_and_collapses_whitespace() -> None:
    assert normalize_name("  ACME   Corp ") == "acme"


def test_normalize_name_strips_legal_suffixes() -> None:
    assert normalize_name("Acme, Inc.") == "acme"
    assert normalize_name("Acme Ltd") == "acme"
    assert normalize_name("Acme LLC") == "acme"
    assert normalize_name("Acme Corporation") == "acme"


def test_normalize_name_strips_greedy_suffix_chains() -> None:
    assert normalize_name("Acme Ltd Inc.") == "acme"
    assert normalize_name("Acme S.A.R.L.") == "acme"


def test_normalize_name_keeps_words_that_are_part_of_a_real_name() -> None:
    assert normalize_name("Acme Holdings LLC") == "acme holdings"


def test_normalize_name_never_empties_a_real_name() -> None:
    assert normalize_name("Inc") == "inc"


def test_normalize_name_is_idempotent() -> None:
    once = normalize_name("Café Holdings, S.A.")
    assert normalize_name(once) == once


def test_normalize_name_removes_accents() -> None:
    assert strip_accents("Café") == "Cafe"
    assert normalize_name("Société Générale") == "societe generale"


def test_normalize_name_fuses_a_trailing_acronym() -> None:
    assert normalize_name("Open AI") == "openai"
    assert normalize_name("OpenAI") == "openai"
    assert normalize_name("Open AI Inc.") == "openai"


def test_normalize_name_fuses_the_ai_initialism() -> None:
    assert normalize_name("A.I. Labs") == "ai labs"
    assert normalize_name("AI Labs") == "ai labs"


def test_normalize_name_keeps_interior_acronyms() -> None:
    assert normalize_name("Acme AI Labs") == "acme ai labs"
    assert normalize_name("Acme AI Labs") != normalize_name("Acme Labs")


def test_normalize_name_does_not_strip_a_real_trailing_ai() -> None:
    assert normalize_name("Airbnb") == "airbnb"


def test_strip_directional_noise_splits_on_punctuation() -> None:
    assert strip_directional_noise("J.R.Smith") == "J R Smith"
    assert normalize_name("J. R. Smith") == "j r smith"
    assert normalize_name("J.R.Smith") == "j r smith"


def test_strip_directional_noise_removes_zero_width_characters() -> None:
    assert strip_directional_noise("A\u200bCme") == "ACme"
    assert normalize_name("A\u200bCme") == "acme"


def test_strip_legal_suffixes_keeps_interior_tokens() -> None:
    assert strip_legal_suffixes("corp acme inc") == "corp acme"


def test_strip_boundaries_drops_edge_articles() -> None:
    assert strip_boundaries("the hertz corporation") == "hertz corporation"
    assert strip_boundaries("sand and gravel") == "sand and gravel"
    assert normalize_name("The Hertz Corporation") == "hertz"


def test_strip_boundaries_keeps_a_name_that_is_entirely_an_article() -> None:
    assert strip_boundaries("the") == "the"


def test_collapse_spaces_strips_the_boundaries() -> None:
    assert collapse_spaces("  a   b  ") == "a b"


def test_normalize_whitespace_casefolds() -> None:
    assert normalize_whitespace("A \t b") == "a b"


def test_normalize_name_of_blank_is_empty() -> None:
    assert normalize_name("   ") == ""
    assert normalize_name("") == ""
    assert normalize_name("///") == ""


def test_normalize_name_rejects_non_strings() -> None:
    with pytest.raises(TypeError):
        normalize_name(None)  # type: ignore[arg-type]


def test_canonical_key_is_stable_across_spellings() -> None:
    assert canonical_key("Acme, Inc.") == canonical_key("  ACME ")
    assert canonical_key("Open AI") == canonical_key("OpenAI")


def test_canonical_key_is_namespaced_by_ruleset_version() -> None:
    assert canonical_key("Acme").startswith(f"{IDENTIFIER_VERSION}_")


def test_canonical_key_rejects_empty_names() -> None:
    with pytest.raises(ValueError, match="empty name"):
        canonical_key("   ")


def test_entity_id_is_type_scoped() -> None:
    assert entity_id("company", "Acme") != entity_id("competitor", "Acme")


def test_entity_id_is_stable_and_prefixed() -> None:
    first = entity_id("company", "Acme, Inc.")
    assert first == entity_id("company", "ACME")
    assert first.startswith(ENTITY_ID_PREFIX)
    assert ENTITY_ID_SEPARATOR in first


def test_entity_id_rejects_a_blank_type() -> None:
    with pytest.raises(ValueError):
        entity_id("   ", "Acme")


def test_entity_id_rejects_a_non_string_type() -> None:
    with pytest.raises(ValueError):
        entity_id(None, "Acme")  # type: ignore[arg-type]


def test_entity_id_rejects_an_empty_value() -> None:
    with pytest.raises(ValueError):
        entity_id("company", "  ")


def test_alias_id_is_distinct_from_an_entity_id() -> None:
    assert alias_id("Acme").startswith(f"alias{ALIAS_ID_SEPARATOR}")
    assert alias_id("Acme") != entity_id("company", "Acme")


def test_normalize_url_lowercases_host_and_drops_fragment() -> None:
    assert normalize_url("HTTPS://Example.COM/a?b=1#c") == "https://example.com/a?b=1"


def test_normalize_url_drops_www_and_default_ports() -> None:
    assert normalize_url("https://www.example.com:443/a") == "https://example.com/a"
    assert normalize_url("http://example.com:80/a") == "http://example.com/a"


def test_normalize_url_sorts_query_parameters() -> None:
    assert normalize_url("https://x.test/?b=2&a=1") == "https://x.test?a=1&b=2"


def test_normalize_url_drops_a_trailing_slash() -> None:
    assert normalize_url("https://example.com/a/") == "https://example.com/a"


def test_normalize_url_of_blank_is_empty() -> None:
    assert normalize_url("   ") == ""


def test_normalize_url_of_bare_text_is_preserved() -> None:
    assert normalize_url("not a url") == "not a url"


def test_normalize_url_rejects_non_strings() -> None:
    with pytest.raises(TypeError):
        normalize_url(None)  # type: ignore[arg-type]


def test_normalize_title_keeps_meaningful_words() -> None:
    assert normalize_title("  The   Quick, Brown-Fox!  ") == "the quick brown fox"


def test_normalize_title_keeps_legal_suffixes() -> None:
    assert normalize_title("Acme Inc") == "acme inc"
    assert normalize_title("Acme Inc") != normalize_name("Acme Inc")


def test_title_key_ignores_punctuation() -> None:
    assert title_key("The Quick, Brown-Fox!") == title_key("the quick brown fox")


def test_title_key_rejects_an_empty_title() -> None:
    with pytest.raises(ValueError, match="empty title"):
        title_key("   ")


def test_normalization_tables_are_exported_and_frozen() -> None:
    assert "llc" in LEGAL_SUFFIX_TOKENS
    assert "ai" in TRAILING_ACRONYM_FUSIONS
    assert INITIALISM_FUSIONS["a|i"] == "ai"
    assert "the" in EDGE_TOKENS


def test_linking_version_is_exported() -> None:
    assert LINKING_VERSION == "1.0.0"


def test_linker_resolves_one_identifier_per_canonical_name() -> None:
    linker = EntityLinker()
    first = linker.resolve("company", "Acme Inc")
    assert linker.resolve("company", "ACME") == first
    assert len(linker) == 1


def test_linker_separates_types() -> None:
    linker = EntityLinker()
    company = linker.resolve("company", "Acme")
    competitor = linker.resolve("competitor", "Acme")
    assert company != competitor
    assert linker.type_for(company) == "company"
    assert linker.type_for(competitor) == "competitor"


def test_linker_peek_does_not_create() -> None:
    linker = EntityLinker()
    assert linker.peek("company", "Acme") is None
    assert len(linker) == 0


def test_linker_peek_finds_a_known_name() -> None:
    linker = EntityLinker()
    identifier = linker.resolve("company", "Acme Inc")
    assert linker.peek("company", "acme") == identifier
    assert linker.is_known("company", "Acme") is True


def test_linker_is_type_strict_when_peeking() -> None:
    linker = EntityLinker()
    linker.resolve("company", "Acme")
    assert linker.is_known("technology", "Acme") is False


def test_linker_tracks_aliases_in_first_seen_order() -> None:
    linker = EntityLinker()
    linker.resolve("company", "Acme Inc")
    linker.resolve("company", "ACME")
    identifier = linker.peek("company", "acme")
    assert identifier is not None
    assert linker.aliases_for(identifier) == ("Acme Inc", "ACME")


def test_linker_repeating_a_name_records_no_second_alias() -> None:
    linker = EntityLinker()
    identifier = linker.resolve("company", "Acme Inc")
    linker.resolve("company", "Acme Inc")
    assert linker.aliases_for(identifier) == ("Acme Inc",)


def test_linker_rejects_an_unusable_name() -> None:
    linker = EntityLinker()
    with pytest.raises(ValueError):
        linker.resolve("company", "///")


def test_linker_rejects_a_non_string_name() -> None:
    linker = EntityLinker()
    with pytest.raises(TypeError):
        linker.resolve("company", 42)  # type: ignore[arg-type]


def test_linker_reports_unknown_identifiers() -> None:
    linker = EntityLinker()
    assert linker.type_for("ent_company_missing") is None
    assert linker.aliases_for("ent_company_missing") == ()


def test_linker_lists_identifiers_and_names_sorted() -> None:
    linker = EntityLinker()
    linker.resolve("company", "Zeta")
    linker.resolve("company", "Alpha")
    assert linker.identifiers() == tuple(sorted(linker.identifiers()))
    assert linker.known_names() == tuple(sorted(linker.known_names()))
    assert len(linker.identifiers()) == 2


def test_linker_round_trips_through_a_dictionary() -> None:
    linker = EntityLinker()
    linker.resolve("company", "Acme Inc")
    linker.resolve("technology", "GPT")
    document = linker.to_dict()
    restored = EntityLinker.from_dict(document)
    assert restored.to_dict() == document
    assert restored.peek("company", "acme") == linker.peek("company", "acme")
    assert len(restored) == 2


def test_linker_from_dict_rejects_a_foreign_shape() -> None:
    with pytest.raises(ValueError):
        EntityLinker.from_dict({"entities": "nope"})


def test_linker_from_dict_rejects_a_non_mapping() -> None:
    with pytest.raises(TypeError):
        EntityLinker.from_dict([])  # type: ignore[arg-type]


def test_resolve_aliases_deduplicates_and_resolves() -> None:
    linker = EntityLinker()
    linker.resolve("company", "Acme Inc")
    resolved = resolve_aliases("company", ["ACME", "Acme Inc", "Beta"])
    assert len(resolved) == 2
    assert linker.peek("company", "acme") in resolved


def test_resolve_aliases_of_nothing_is_empty() -> None:
    assert resolve_aliases("company", []) == ()


def test_normalization_pipeline_is_case_and_accent_stable() -> None:
    assert normalize_name("J. R. Smith") == "j r smith"
    assert normalize_name("J.R.Smith") == "j r smith"


def test_fuse_tokens_leaves_a_single_token_alone() -> None:
    assert fuse_tokens("ai") == "ai"
    assert fuse_tokens("") == ""
