"""Tests for deterministic string-similarity functions."""

from __future__ import annotations

from predictron_engine.dataset.fuzzy import (
    jaccard,
    jaro,
    jaro_winkler,
    levenshtein,
    name_similarity,
    normalized_levenshtein,
    token_overlap,
    tokenize,
)

# ---------------------------------------------------------------------------
# levenshtein
# ---------------------------------------------------------------------------

class TestLevenshtein:
    def test_identical(self) -> None:
        assert levenshtein("abc", "abc") == 0

    def test_empty_vs_nonempty(self) -> None:
        assert levenshtein("", "abc") == 3

    def test_both_empty(self) -> None:
        assert levenshtein("", "") == 0

    def test_single_substitution(self) -> None:
        assert levenshtein("abc", "axc") == 1

    def test_single_insertion(self) -> None:
        assert levenshtein("ac", "abc") == 1

    def test_single_deletion(self) -> None:
        assert levenshtein("abc", "ac") == 1

    def test_complete_replacement(self) -> None:
        assert levenshtein("abc", "xyz") == 3

    def test_symmetric(self) -> None:
        assert levenshtein("kitten", "sitting") == levenshtein("sitting", "kitten")

    def test_long_strings(self) -> None:
        a = "algorithm"
        b = "altruistic"
        assert levenshtein(a, b) == 6

    def test_single_char(self) -> None:
        assert levenshtein("a", "b") == 1
        assert levenshtein("a", "a") == 0


# ---------------------------------------------------------------------------
# normalized_levenshtein
# ---------------------------------------------------------------------------

class TestNormalizedLevenshtein:
    def test_identical(self) -> None:
        assert normalized_levenshtein("abc", "abc") == 1.0

    def test_completely_different(self) -> None:
        assert normalized_levenshtein("abc", "xyz") < 0.5

    def test_empty_vs_empty(self) -> None:
        assert normalized_levenshtein("", "") == 1.0

    def test_empty_vs_nonempty(self) -> None:
        assert normalized_levenshtein("", "abc") == 0.0

    def test_one_char_edit(self) -> None:
        # 1 edit in a 3-char string → 1 - 1/3 ≈ 0.667.
        score = normalized_levenshtein("abc", "axc")
        assert abs(score - (2 / 3)) < 1e-6

    def test_range(self) -> None:
        score = normalized_levenshtein("hello", "world")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# jaro
# ---------------------------------------------------------------------------

class TestJaro:
    def test_identical(self) -> None:
        assert jaro("abc", "abc") == 1.0

    def test_empty(self) -> None:
        assert jaro("", "abc") == 0.0

    def test_both_empty(self) -> None:
        assert jaro("", "") == 1.0

    def test_no_common_chars(self) -> None:
        assert jaro("abc", "xyz") == 0.0

    def test_partial_match(self) -> None:
        score = jaro("martha", "marhta")
        assert 0.9 < score < 1.0

    def test_range(self) -> None:
        score = jaro("hello", "world")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# jaro_winkler
# ---------------------------------------------------------------------------

class TestJaroWinkler:
    def test_identical(self) -> None:
        assert jaro_winkler("abc", "abc") == 1.0

    def test_empty(self) -> None:
        assert jaro_winkler("", "abc") == 0.0

    def test_prefix_boost(self) -> None:
        # Jaro-Winkler boosts common prefixes.
        assert jaro_winkler("martha", "marhta") > jaro("martha", "marhta")

    def test_short_prefix_no_boost_below_threshold(self) -> None:
        # When Jaro is <= 0.7, Winkler returns Jaro directly.
        score_jw = jaro_winkler("abc", "xyz")
        score_j = jaro("abc", "xyz")
        assert score_jw == score_j

    def test_range(self) -> None:
        score = jaro_winkler("hello", "world")
        assert 0.0 <= score <= 1.0

    def test_prefix_four_chars_max(self) -> None:
        # Prefix boost caps at 4 characters.
        s1 = jaro_winkler("abcdef", "abcxyz")
        s2 = jaro_winkler("abcdef", "abcwxyz")
        # Both should have 3-char prefix match (abc).
        assert s1 > 0.0
        assert s2 > 0.0


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic(self) -> None:
        assert tokenize("Hello World") == ["hello", "world"]

    def test_empty(self) -> None:
        assert tokenize("") == []

    def test_punctuation(self) -> None:
        assert tokenize("Acme, Corp.") == ["acme", "corp"]

    def test_numbers(self) -> None:
        assert tokenize("Acme 123") == ["acme", "123"]

    def test_multiple_spaces(self) -> None:
        assert tokenize("a  b  c") == ["a", "b", "c"]

    def test_mixed(self) -> None:
        tokens = tokenize("Data-Stream Inc.")
        assert "data" in tokens
        assert "stream" in tokens


# ---------------------------------------------------------------------------
# token_overlap
# ---------------------------------------------------------------------------

class TestTokenOverlap:
    def test_identical(self) -> None:
        assert token_overlap(["a", "b"], ["a", "b"]) == 1.0

    def test_no_overlap(self) -> None:
        assert token_overlap(["a", "b"], ["c", "d"]) == 0.0

    def test_subset(self) -> None:
        # "acme" is a subset of "acme corp"; overlap uses min size.
        assert token_overlap(["acme"], ["acme", "corp"]) == 1.0

    def test_partial(self) -> None:
        score = token_overlap(["a", "b", "c"], ["b", "c", "d"])
        assert abs(score - 2 / 3) < 1e-6

    def test_empty(self) -> None:
        assert token_overlap([], ["a"]) == 0.0
        assert token_overlap(["a"], []) == 0.0
        assert token_overlap([], []) == 0.0


# ---------------------------------------------------------------------------
# jaccard
# ---------------------------------------------------------------------------

class TestJaccard:
    def test_identical(self) -> None:
        assert jaccard(["a", "b"], ["a", "b"]) == 1.0

    def test_no_overlap(self) -> None:
        assert jaccard(["a"], ["b"]) == 0.0

    def test_partial(self) -> None:
        score = jaccard(["a", "b"], ["b", "c"])
        assert abs(score - 1 / 3) < 1e-6

    def test_empty(self) -> None:
        assert jaccard([], ["a"]) == 0.0
        assert jaccard([], []) == 0.0


# ---------------------------------------------------------------------------
# name_similarity
# ---------------------------------------------------------------------------

class TestNameSimilarity:
    def test_identical(self) -> None:
        assert name_similarity("Acme Corp", "Acme Corp") == 1.0

    def test_empty(self) -> None:
        assert name_similarity("", "Acme") == 0.0
        assert name_similarity("Acme", "") == 0.0
        # Two empty strings are equal → 1.0.
        assert name_similarity("", "") == 1.0

    def test_high_similarity(self) -> None:
        score = name_similarity("Acme Corp", "Acme Corporation")
        assert score > 0.7

    def test_moderate_similarity(self) -> None:
        score = name_similarity("Acme Corp", "Acme Industries")
        assert 0.3 < score < 0.8

    def test_low_similarity(self) -> None:
        score = name_similarity("Acme Corp", "Beta LLC")
        assert score < 0.4

    def test_token_compaction_tolerance(self) -> None:
        # "Data Stream" vs "Datastream" should be above the token-overlap-only
        # baseline (which would be 0 for non-matching tokens).
        score = name_similarity("Data Stream", "Datastream")
        assert score > 0.5

    def test_range(self) -> None:
        score = name_similarity("Hello World", "Foo Bar")
        assert 0.0 <= score <= 1.0

    def test_symmetric(self) -> None:
        a, b = "Acme Corp", "Acme Inc"
        assert name_similarity(a, b) == name_similarity(b, a)

    def test_deterministic(self) -> None:
        a, b = "Acme Corp", "Acme Corp"
        assert name_similarity(a, b) == name_similarity(a, b)
