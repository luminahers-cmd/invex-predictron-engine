"""Tests for the shared canonical serialization helper.

Every content-addressed identifier in the engine is a SHA-256 over a
canonical JSON encoding.  Before this module existed there were eleven
compact-canonical call sites speaking three dialects at three digest
widths, so digest equality between two subsystems was a coincidence of
payload shape rather than a guarantee.

These tests pin the encoding itself, and — more importantly — pin the
compatibility properties that let the consolidation happen without
invalidating a single stored identifier.
"""

from __future__ import annotations

import datetime
import hashlib
import json

import pytest

from predictron_engine.canonical import (
    FULL_DIGEST_LENGTH,
    RESEARCH_DIGEST_LENGTH,
    TRACE_DIGEST_LENGTH,
    canonical_json,
    content_digest,
    text_digest,
)
from predictron_engine.monitoring.models import json_dumps_canonical
from predictron_engine.research.validation_models import (
    canonical_json as research_canonical_json,
)
from predictron_engine.research.validation_models import (
    content_digest as research_content_digest,
)

# Payloads that must encode identically everywhere.  The non-ASCII cases
# are the point: an ``ensure_ascii=False`` encoder produces different
# bytes for the same logical content as soon as a payload leaves ASCII,
# which silently breaks equality between two digests of one fact.
VECTORS: tuple[object, ...] = (
    {},
    {"a": 1},
    {"b": 2, "a": 1},
    {"z": [], "a": {"y": 1, "x": [1, 2, 3]}, "s": "plain"},
    {"claim": "café naïve", "topic": "market", "n": 0.5},
    {"unicode_key_é": "v", "plain": "w"},
    {"nested": {"deep": {"deeper": "value"}}},
    {"escaped": 'quote" backslash\\ newline\n tab\t'},
    {"unicode_escape": "café"},
    [1, 2, {"k": "é"}],
)

# ``VECTORS`` deliberately mixes in list payloads; key-order invariance
# only makes sense for mappings, so that one test filters them out.
MAPPING_VECTORS = tuple(v for v in VECTORS if isinstance(v, dict))


class TestCanonicalJson:
    """The encoding itself."""

    @pytest.mark.parametrize("payload", VECTORS)
    def test_sorts_keys_and_is_compact(self, payload: object) -> None:
        encoded = canonical_json(payload)
        assert ", " not in encoded
        assert '": ' not in encoded

    @pytest.mark.parametrize("payload", MAPPING_VECTORS)
    def test_key_order_does_not_change_the_bytes(self, payload: dict) -> None:
        shuffled = dict(reversed(list(payload.items())))
        assert canonical_json(shuffled) == canonical_json(payload)

    @pytest.mark.parametrize("payload", VECTORS)
    def test_list_order_does_change_the_bytes(self, payload: object) -> None:
        """Ordering inside a list is meaning, so it must survive."""
        assert canonical_json([1, 2]) != canonical_json([2, 1])

    @pytest.mark.parametrize("payload", VECTORS)
    def test_non_ascii_is_escaped(self, payload: object) -> None:
        """ASCII output is what makes the bytes platform-independent."""
        assert canonical_json(payload).isascii()

    def test_non_ascii_differs_from_unescaped_encoding(self) -> None:
        """Guards the specific defect that motivated ensure_ascii=True."""
        payload = {"k": "café"}
        assert canonical_json(payload) == '{"k":"caf\\u00e9"}'
        unescaped = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        assert unescaped != canonical_json(payload)


class TestStrictness:
    """A digest over a lossy rendering of its input is not reproducible."""

    def test_non_json_native_raises(self) -> None:
        with pytest.raises(TypeError):
            canonical_json({"t": datetime.datetime(2020, 1, 1)})

    def test_set_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            canonical_json({"s": {1, 2}})


class TestMonitoringCompatibility:
    """The monitoring wrapper must keep producing the same bytes.

    Its previous ``default=str`` fallback silently stringified a
    ``datetime`` into the digest, which meant a snapshot's hash and its
    persisted bytes could disagree with no error anywhere.  Delegating to
    the shared helper removes that, but only for JSON-native payloads may
    the output change — and it must not.
    """

    @pytest.mark.parametrize("payload", VECTORS)
    def test_matches_previous_monitoring_output(self, payload: object) -> None:
        previous = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        assert json_dumps_canonical(payload) == previous

    def test_monitoring_no_longer_swallows_datetimes(self) -> None:
        with pytest.raises(TypeError):
            json_dumps_canonical({"t": datetime.datetime(2020, 1, 1)})

    def test_legacy_fingerprint_expression_is_preserved(self) -> None:
        """Learning and monitoring hashed with ``default=str``.

        ``analytic_payload`` already returns ``model_dump(mode="json")``,
        so that fallback is unreachable for JSON-native input; routing it
        through the strict helper must therefore be byte-identical.
        """
        payload = {"metric": 0.5, "label": "café", "tags": ["a", "b"]}
        legacy = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        ).hexdigest()
        assert (
            content_digest(payload, length=FULL_DIGEST_LENGTH) == legacy
        ), "delegating must not change any stored snapshot hash"

    @pytest.mark.parametrize("payload", VECTORS)
    def test_legacy_expression_matches_for_every_vector(self, payload: object) -> None:
        legacy = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        ).hexdigest()
        assert content_digest(payload, length=FULL_DIGEST_LENGTH) == legacy


class TestResearchCompatibility:
    """The research wrapper must keep producing the same bytes too."""

    @pytest.mark.parametrize("payload", VECTORS)
    def test_matches_previous_research_output(self, payload: object) -> None:
        assert research_canonical_json(payload) == canonical_json(payload)

    @pytest.mark.parametrize("payload", VECTORS)
    def test_research_digest_is_sixteen_chars(self, payload: object) -> None:
        assert len(research_content_digest(payload)) == RESEARCH_DIGEST_LENGTH

    def test_research_digest_is_insertion_order_insensitive(self) -> None:
        assert research_content_digest({"x": 1, "y": 2}) == research_content_digest(
            {"y": 2, "x": 1}
        )


class TestContentDigest:
    """Prefix and width are call-site decisions, not payload properties."""

    def test_prefix_is_applied(self) -> None:
        assert content_digest({"a": 1}, prefix="ev_").startswith("ev_")

    def test_width_is_explicit(self) -> None:
        payload = {"a": 1}
        assert len(content_digest(payload, length=RESEARCH_DIGEST_LENGTH)) == 16
        assert len(content_digest(payload, length=FULL_DIGEST_LENGTH)) == 64
        assert len(content_digest(payload, length=TRACE_DIGEST_LENGTH)) == 12

    def test_prefixed_and_bare_digests_share_a_body(self) -> None:
        payload = {"a": 1}
        assert content_digest(payload, prefix="plan_") == (
            "plan_" + content_digest(payload)
        )

    def test_distinct_content_yields_distinct_digests(self) -> None:
        assert content_digest({"a": 1}) != content_digest({"a": 2})


class TestTextDigest:
    """Raw-string hashing is kept separate on purpose.

    Reaching for this instead of ``content_digest`` is a deliberate
    choice to skip canonicalization, so it is a separate function rather
    than a flag.
    """

    def test_matches_hashlib(self) -> None:
        assert text_digest("payload") == hashlib.sha256(b"payload").hexdigest()

    def test_respects_width(self) -> None:
        assert len(text_digest("payload", length=RESEARCH_DIGEST_LENGTH)) == 16

    def test_agrees_with_content_digest_on_canonical_text(self) -> None:
        payload = {"a": 1, "b": "é"}
        assert text_digest(
            canonical_json(payload), length=FULL_DIGEST_LENGTH
        ) == content_digest(payload, length=FULL_DIGEST_LENGTH)
