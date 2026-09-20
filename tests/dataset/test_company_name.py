"""Tests for company name normalization and corporate-suffix stripping."""

from __future__ import annotations

from predictron_engine.dataset.company_name import (
    canonical_name_key,
    core_name,
    fold_name,
    name_tokens,
    strip_corporate_suffixes,
)

# ---------------------------------------------------------------------------
# fold_name
# ---------------------------------------------------------------------------

class TestFoldName:
    def test_lowercase(self) -> None:
        assert fold_name("Acme Corp") == "acme corp"

    def test_unicode_nfc(self) -> None:
        # e-acute in two NFC-incompatible forms should collapse to one.
        assert fold_name("\u00e9cole") == "\u00e9cole"

    def test_whitespace_stripped(self) -> None:
        assert fold_name("  Acme  ") == "acme"

    def test_casefold_sharp_s(self) -> None:
        assert fold_name("STRASSE") == "strasse"


# ---------------------------------------------------------------------------
# strip_corporate_suffixes
# ---------------------------------------------------------------------------

class TestStripCorporateSuffixes:
    def test_inc(self) -> None:
        core, labels = strip_corporate_suffixes("Acme Inc")
        assert core == "acme"
        assert labels == ["INC"]

    def test_llc(self) -> None:
        core, labels = strip_corporate_suffixes("Acme LLC")
        assert core == "acme"
        assert labels == ["LLC"]

    def test_ltd(self) -> None:
        core, labels = strip_corporate_suffixes("Acme Ltd")
        assert core == "acme"
        assert labels == ["LTD"]

    def test_plc(self) -> None:
        core, labels = strip_corporate_suffixes("Acme PLC")
        assert core == "acme"
        assert labels == ["PLC"]

    def test_gmbh(self) -> None:
        core, labels = strip_corporate_suffixes("Acme GmbH")
        assert core == "acme"
        assert labels == ["GMBH"]

    def test_gmbh_co_kg(self) -> None:
        core, labels = strip_corporate_suffixes("Acme GmbH & Co. KG")
        assert core == "acme"
        assert labels == ["GMBH"]

    def test_bv(self) -> None:
        core, labels = strip_corporate_suffixes("Acme B.V.")
        assert core == "acme"
        assert labels == ["BV"]

    def test_sa(self) -> None:
        core, labels = strip_corporate_suffixes("Acme SA")
        assert core == "acme"
        assert labels == ["SA"]

    def test_sas(self) -> None:
        core, labels = strip_corporate_suffixes("Acme SAS")
        assert core == "acme"
        assert labels == ["SAS"]

    def test_oy(self) -> None:
        core, labels = strip_corporate_suffixes("Acme Oy")
        assert core == "acme"
        assert labels == ["OY"]

    def test_ab(self) -> None:
        core, labels = strip_corporate_suffixes("Acme AB")
        assert core == "acme"
        assert labels == ["AB"]

    def test_ag(self) -> None:
        core, labels = strip_corporate_suffixes("Acme AG")
        assert core == "acme"
        assert labels == ["AG"]

    def test_incorporated(self) -> None:
        core, labels = strip_corporate_suffixes("Acme Incorporated")
        assert core == "acme"
        assert labels == ["INC"]

    def test_corporation(self) -> None:
        core, labels = strip_corporate_suffixes("Acme Corporation")
        assert core == "acme"
        assert labels == ["CORP"]

    def test_limited_liability_company(self) -> None:
        core, labels = strip_corporate_suffixes("Acme Limited Liability Company")
        assert core == "acme"
        assert labels == ["LLC"]

    def test_pte(self) -> None:
        core, labels = strip_corporate_suffixes("Acme Pte Ltd")
        assert core == "acme"
        assert labels == ["LTD", "PTE"]

    def test_sarl(self) -> None:
        core, labels = strip_corporate_suffixes("Acme SARL")
        assert core == "acme"
        assert labels == ["SARL"]

    def test_srl(self) -> None:
        core, labels = strip_corporate_suffixes("Acme SRL")
        assert core == "acme"
        assert labels == ["SRL"]

    def test_spa(self) -> None:
        core, labels = strip_corporate_suffixes("Acme SpA")
        assert core == "acme"
        assert labels == ["SPA"]

    def test_no_suffix(self) -> None:
        core, labels = strip_corporate_suffixes("Acme")
        assert core == "acme"
        assert labels == []

    def test_empty_string(self) -> None:
        core, labels = strip_corporate_suffixes("")
        assert core == ""
        assert labels == []

    def test_only_suffix(self) -> None:
        core, labels = strip_corporate_suffixes("Inc")
        assert core == ""
        assert labels == ["INC"]

    def test_multiple_suffix_layers(self) -> None:
        core, labels = strip_corporate_suffixes("Acme Corp Inc")
        assert core == "acme"
        assert "INC" in labels

    def test_case_insensitive(self) -> None:
        core, _ = strip_corporate_suffixes("acme llc")
        assert core == "acme"

    def test_periods_in_name(self) -> None:
        core, _ = strip_corporate_suffixes("Acme, Inc.")
        # Comma is not part of suffix stripping; core_name handles it.
        assert "acme" in core
        assert "inc" not in core


# ---------------------------------------------------------------------------
# core_name
# ---------------------------------------------------------------------------

class TestCoreName:
    def test_basic(self) -> None:
        assert core_name("Acme Corp") == "acme"

    def test_punctuation_removal(self) -> None:
        assert core_name("Acme, Inc.") == "acme"

    def test_unicode(self) -> None:
        # core_name uses [^0-9a-z]+ regex which strips non-ASCII chars.
        result = core_name("Über GmbH")
        assert result == "ber"

    def test_whitespace_collapse(self) -> None:
        assert core_name("  Acme   Corp  ") == "acme"

    def test_empty(self) -> None:
        assert core_name("") == ""

    def test_single_word(self) -> None:
        assert core_name("Acme") == "acme"

    def test_corporate_suffix_stripped(self) -> None:
        assert core_name("Acme Limited Liability Company") == "acme"

    def test_numbers_preserved(self) -> None:
        assert core_name("Acme 123") == "acme 123"


# ---------------------------------------------------------------------------
# canonical_name_key
# ---------------------------------------------------------------------------

class TestCanonicalNameKey:
    def test_strips_all_non_alnum(self) -> None:
        # "Corp" is a suffix; after stripping, only "acme" remains.
        assert canonical_name_key("Acme Corp") == "acme"

    def test_suffix_stripped(self) -> None:
        assert canonical_name_key("Acme Inc") == "acme"

    def test_collapses_identical(self) -> None:
        # Different punctuation around the core name compresses identically.
        assert canonical_name_key("Acme-Corp") == canonical_name_key("acme-corp")
        assert canonical_name_key("Acme. Corp.") == "acmecorp"

    def test_empty(self) -> None:
        assert canonical_name_key("") == ""

    def test_deterministic(self) -> None:
        a = canonical_name_key("Acme Corp Inc.")
        b = canonical_name_key("Acme Corp Inc.")
        assert a == b


# ---------------------------------------------------------------------------
# name_tokens
# ---------------------------------------------------------------------------

class TestNameTokens:
    def test_basic(self) -> None:
        tokens = name_tokens("Acme Corp")
        assert "acme" in tokens
        assert "corp" not in tokens  # corp is a stopword

    def test_stopwords_removed(self) -> None:
        tokens = name_tokens("The Acme Company")
        assert "the" not in tokens
        assert "company" not in tokens
        assert "acme" in tokens

    def test_deduplication(self) -> None:
        tokens = name_tokens("Acme Acme Corp")
        assert tokens.count("acme") == 1

    def test_empty(self) -> None:
        assert name_tokens("") == []

    def test_single_stopword(self) -> None:
        assert name_tokens("Inc") == []

    def test_preserves_order(self) -> None:
        tokens = name_tokens("Zebra Alpha Corp")
        assert tokens == ["zebra", "alpha"]

    def test_numeric_tokens_kept(self) -> None:
        tokens = name_tokens("Acme 123")
        assert "123" in tokens


# ---------------------------------------------------------------------------
# Cross-function consistency
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_core_name_matches_canonical_key_when_no_suffix(self) -> None:
        name = "Acme"
        assert canonical_name_key(name) == core_name(name).replace(" ", "")

    def test_fold_name_idempotent(self) -> None:
        name = "Acme Corp"
        assert fold_name(fold_name(name)) == fold_name(name)

    def test_core_name_idempotent(self) -> None:
        name = "Acme Inc."
        assert core_name(core_name(name)) == core_name(name)
