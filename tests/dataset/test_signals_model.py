"""Tests for predictron_engine.dataset.signals.model — Project E4."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest

from predictron_engine.dataset.signals.model import (
    EXTERNAL_ONLY_SIGNAL_TYPES,
    SIGNAL_TYPES,
    CompanySignal,
    EvidenceReference,
    SignalType,
    compute_signal_id,
    signal_canonical_dict,
)
from tests.dataset.signals_helpers import dt, make_signal

# ── SignalType / vocabulary ────────────────────────────────────────────

class TestSignalType:
    def test_has_sixteen_members(self) -> None:
        assert len(SignalType) == 16

    def test_signal_types_tuple_matches(self) -> None:
        assert SIGNAL_TYPES == tuple(SignalType)

    def test_values_are_lowercase_snake(self) -> None:
        for member in SignalType:
            assert member.value == member.value.lower()
            assert " " not in member.value

    def test_str_enum_behaviour(self) -> None:
        assert SignalType.FUNDING_ROUND == "funding_round"

    def test_external_only_subset(self) -> None:
        assert EXTERNAL_ONLY_SIGNAL_TYPES <= frozenset(SignalType)
        assert len(EXTERNAL_ONLY_SIGNAL_TYPES) == 5
        for name in (
            "founder_change", "product_launch", "partnership",
            "regulatory_approval", "revenue_milestone",
        ):
            assert SignalType(name) in EXTERNAL_ONLY_SIGNAL_TYPES
        assert SignalType.LAYOFFS not in EXTERNAL_ONLY_SIGNAL_TYPES


# ── EvidenceReference ──────────────────────────────────────────────────

class TestEvidenceReference:
    def test_to_dict_round_trip(self) -> None:
        er = EvidenceReference(
            reference="https://example.com",
            description="test",
            retrieved_at=dt("2024-06-01T12:00:00+00:00"),
        )
        d = er.to_dict()
        restored = EvidenceReference.from_dict(d)
        assert restored == er

    def test_from_dict_missing_retrieved(self) -> None:
        er = EvidenceReference.from_dict({"reference": "x"})
        assert er.retrieved_at is None

    def test_from_dict_invalid_retrieved(self) -> None:
        er = EvidenceReference.from_dict(
            {"reference": "x", "retrieved_at": "not-a-date"}
        )
        assert er.retrieved_at is None

    def test_frozen(self) -> None:
        er = EvidenceReference(reference="x")
        with pytest.raises(AttributeError):
            er.reference = "y"  # type: ignore[misc]


# ── CompanySignal construction ─────────────────────────────────────────

class TestCompanySignal:
    def test_frozen(self) -> None:
        s = make_signal()
        with pytest.raises(AttributeError):
            s.company_id = "other"  # type: ignore[misc]

    def test_coerces_string_signal_type(self) -> None:
        s = make_signal(signal_type="hiring_growth")  # type: ignore[arg-type]
        assert s.signal_type == SignalType.HIRING_GROWTH

    def test_rejects_unknown_signal_type(self) -> None:
        with pytest.raises(ValueError, match="not a valid SignalType"):
            make_signal(signal_type="not_a_real_type")  # type: ignore[arg-type]

    def test_rejects_naive_timestamp(self) -> None:

        with pytest.raises(ValueError, match="tz-aware"):
            CompanySignal(
                company_id="c1",
                signal_type=SignalType.FUNDING_ROUND,
                timestamp=datetime(2024, 1, 15),
                source="test",
            )

    def test_normalizes_to_utc(self) -> None:
        s = make_signal(at="2024-01-15T10:00:00+05:00")
        assert s.timestamp.tzinfo is UTC
        assert s.timestamp.hour == 5

    def test_rejects_confidence_below_zero(self) -> None:
        with pytest.raises(ValueError):
            make_signal(confidence=-0.1)

    def test_rejects_confidence_above_one(self) -> None:
        with pytest.raises(ValueError):
            make_signal(confidence=1.1)

    def test_boundary_confidence_zero(self) -> None:
        s = make_signal(confidence=0.0)
        assert s.confidence == 0.0

    def test_boundary_confidence_one(self) -> None:
        s = make_signal(confidence=1.0)
        assert s.confidence == 1.0

    def test_metadata_becomes_immutable(self) -> None:
        s = make_signal(x="y")
        assert type(s.metadata).__name__ == "mappingproxy"
        with pytest.raises(TypeError):
            s.metadata["z"] = "w"  # type: ignore[index]

    def test_metadata_value(self) -> None:
        s = make_signal(key="val")
        assert s.metadata_value("key") == "val"
        assert s.metadata_value("missing", "default") == "default"

    def test_with_metadata_returns_new(self) -> None:
        s1 = make_signal(a=1)
        s2 = s1.with_metadata(b=2)
        assert s1 is not s2
        assert "a" in s1.metadata
        assert "b" in s2.metadata

    def test_deterministic_signal_id(self) -> None:
        s1 = make_signal(at="2024-06-01T00:00:00+00:00", x="y")
        s2 = make_signal(at="2024-06-01T00:00:00+00:00", x="y")
        assert s1.signal_id == s2.signal_id

    def test_different_payload_different_id(self) -> None:
        s1 = make_signal(at="2024-06-01T00:00:00+00:00")
        s2 = make_signal(at="2024-06-02T00:00:00+00:00")
        assert s1.signal_id != s2.signal_id

    def test_signal_id_is_sha256_hex(self) -> None:
        s = make_signal()
        assert len(s.signal_id) == 64
        int(s.signal_id, 16)  # must not raise

    def test_as_provenance(self) -> None:
        s = make_signal(at="2024-03-01T00:00:00+00:00")
        p = s.as_provenance()
        assert "funding_round" in p
        assert "2024-03-01" in p
        assert "test" in p


# ── Serialization ──────────────────────────────────────────────────────

class TestSerialization:
    def test_to_dict_has_signal_id(self) -> None:
        s = make_signal()
        d = s.to_dict()
        assert d["signal_id"] == s.signal_id
        assert d["signal_type"] == "funding_round"

    def test_round_trip(self) -> None:
        s = make_signal(at="2024-07-01T00:00:00+00:00", amount=42.0)
        d = s.to_dict()
        s2 = CompanySignal.from_dict(d)
        assert s2.signal_id == s.signal_id
        assert s2.company_id == s.company_id

    def test_from_dict_recomputes_id(self) -> None:
        s = make_signal()
        d = s.to_dict()
        d["signal_id"] = "fake_id"
        s2 = CompanySignal.from_dict(d)
        assert s2.signal_id == s.signal_id  # recomputed

    def test_from_dict_requires_timestamp(self) -> None:
        with pytest.raises(ValueError, match="timestamp"):
            CompanySignal.from_dict({"signal_type": "ipo"})

    def test_from_dict_requires_signal_type(self) -> None:
        with pytest.raises(ValueError, match="signal_type"):
            CompanySignal.from_dict({"timestamp": "2024-01-01T00:00:00+00:00"})

    def test_from_dict_rejects_bool_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence must be numeric"):
            CompanySignal.from_dict({
                "timestamp": "2024-01-01T00:00:00+00:00",
                "signal_type": "ipo",
                "confidence": True,
            })

    def test_from_dict_accepts_string_confidence(self) -> None:
        s = CompanySignal.from_dict({
            "timestamp": "2024-01-01T00:00:00+00:00",
            "signal_type": "ipo",
            "confidence": "0.75",
        })
        assert s.confidence == 0.75

    def test_canonical_dict_deterministic(self) -> None:
        s = make_signal()
        d1 = signal_canonical_dict(s)
        d2 = signal_canonical_dict(s)
        j1 = json.dumps(d1, sort_keys=True, separators=(",", ":"))
        j2 = json.dumps(d2, sort_keys=True, separators=(",", ":"))
        assert j1 == j2

    def test_compute_signal_id_matches_hash(self) -> None:
        s = make_signal()
        canonical = json.dumps(
            signal_canonical_dict(s),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert compute_signal_id(s) == expected
        assert s.signal_id == expected
