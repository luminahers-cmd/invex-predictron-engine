"""Tests for the Evidence Provenance & Trust Framework (Sprint 5A).

Covers trust scoring, freshness decay, duplicate penalties, provenance
generation, trust summary aggregation, edge cases, determinism,
serialization, and backward compatibility.  Every test is
pure-deterministic — identical inputs always produce identical outputs.
No network, no LLMs, no side effects.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from predictron_engine.evidence.provenance import (
    _FRESHNESS_HALF_LIFE_DAYS,
    _TRUST_WEIGHTS,
    ProvenanceRecord,
    SourceTrustLevel,
    TrustFactor,
    TrustScore,
    TrustSummary,
    _compute_freshness_score,
    _compute_source_type_score,
    _source_trust_level_from_score,
    build_provenance_record,
    compute_document_trust,
    compute_trust_summary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FROZEN_NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
_FROZEN_FETCHED = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)


def _trust(
    *,
    authority: float = 0.7,
    quality: float = 0.6,
    is_official: bool = False,
    is_tp: bool = False,
    is_dup: bool = False,
    fetched_at: datetime = _FROZEN_FETCHED,
    now: datetime = _FROZEN_NOW,
) -> TrustScore:
    """Convenience wrapper for compute_document_trust with frozen time."""
    return compute_document_trust(
        authority_score=authority,
        quality_score=quality,
        fetched_at=fetched_at,
        is_official=is_official,
        is_third_party_credible=is_tp,
        is_duplicate=is_dup,
        now=now,
    )


# ===================================================================
# Part 1 — SourceTrustLevel enum
# ===================================================================


class TestSourceTrustLevel:
    def test_all_values_are_strings(self) -> None:
        for level in SourceTrustLevel:
            assert isinstance(level.value, str)

    def test_official_is_highest(self) -> None:
        assert SourceTrustLevel.OFFICIAL.value == "official"

    def test_unknown_is_lowest(self) -> None:
        assert SourceTrustLevel.UNKNOWN.value == "unknown"

    def test_exact_member_count(self) -> None:
        assert len(SourceTrustLevel) == 4


# ===================================================================
# Part 2 — TrustFactor model
# ===================================================================


class TestTrustFactor:
    def test_construction(self) -> None:
        f = TrustFactor(name="authority", weight=0.3, value=0.8)
        assert f.name == "authority"
        assert f.weight == 0.3
        assert f.value == 0.8
        assert f.explanation == ""

    def test_with_explanation(self) -> None:
        f = TrustFactor(name="freshness", weight=0.15, value=0.5, explanation="90d old")
        assert f.explanation == "90d old"

    def test_weight_bounds(self) -> None:
        f = TrustFactor(name="t", weight=0.0, value=1.0)
        assert f.weight == 0.0
        f2 = TrustFactor(name="t", weight=1.0, value=0.0)
        assert f2.weight == 1.0

    def test_weight_rejects_out_of_range(self) -> None:
        with pytest.raises(Exception):
            TrustFactor(name="t", weight=-0.1, value=0.5)
        with pytest.raises(Exception):
            TrustFactor(name="t", weight=1.1, value=0.5)

    def test_value_rejects_out_of_range(self) -> None:
        with pytest.raises(Exception):
            TrustFactor(name="t", weight=0.5, value=-0.1)
        with pytest.raises(Exception):
            TrustFactor(name="t", weight=0.5, value=1.1)


# ===================================================================
# Part 3 — TrustScore model
# ===================================================================


class TestTrustScore:
    def test_construction(self) -> None:
        s = TrustScore(overall=0.75)
        assert s.overall == 0.75
        assert s.factors == []
        assert s.source_trust_level == SourceTrustLevel.UNKNOWN
        assert s.freshness_days == 0

    def test_full_construction(self) -> None:
        factors = [TrustFactor(name="a", weight=0.5, value=0.8)]
        s = TrustScore(
            overall=0.75,
            factors=factors,
            source_trust_level=SourceTrustLevel.OFFICIAL,
            freshness_days=30,
        )
        assert len(s.factors) == 1
        assert s.source_trust_level == SourceTrustLevel.OFFICIAL
        assert s.freshness_days == 30

    def test_overall_bounds(self) -> None:
        s = TrustScore(overall=0.0)
        assert s.overall == 0.0
        s2 = TrustScore(overall=1.0)
        assert s2.overall == 1.0

    def test_overall_rejects_out_of_range(self) -> None:
        with pytest.raises(Exception):
            TrustScore(overall=-0.1)
        with pytest.raises(Exception):
            TrustScore(overall=1.1)


# ===================================================================
# Part 4 — Freshness decay
# ===================================================================


class TestFreshnessDecay:
    def test_zero_days(self) -> None:
        now = datetime(2025, 1, 1, tzinfo=UTC)
        score, days = _compute_freshness_score(now, now)
        assert days == 0
        assert score == pytest.approx(1.0, abs=0.01)

    def test_half_life(self) -> None:
        fetched = datetime(2025, 1, 1, tzinfo=UTC)
        now = fetched + timedelta(days=_FRESHNESS_HALF_LIFE_DAYS)
        score, days = _compute_freshness_score(fetched, now)
        assert days == _FRESHNESS_HALF_LIFE_DAYS
        assert score == pytest.approx(0.5, abs=0.01)

    def test_two_half_lives(self) -> None:
        fetched = datetime(2025, 1, 1, tzinfo=UTC)
        now = fetched + timedelta(days=2 * _FRESHNESS_HALF_LIFE_DAYS)
        score, days = _compute_freshness_score(fetched, now)
        assert days == 2 * _FRESHNESS_HALF_LIFE_DAYS
        assert score == pytest.approx(0.25, abs=0.01)

    def test_very_old_document(self) -> None:
        fetched = datetime(2020, 1, 1, tzinfo=UTC)
        now = datetime(2025, 1, 1, tzinfo=UTC)
        score, days = _compute_freshness_score(fetched, now)
        assert days >= 1825
        assert score < 0.01

    def test_future_date_clamps(self) -> None:
        fetched = datetime(2025, 6, 1, tzinfo=UTC)
        now = datetime(2025, 1, 1, tzinfo=UTC)
        score, days = _compute_freshness_score(fetched, now)
        assert days == 0
        assert score == pytest.approx(1.0, abs=0.01)


# ===================================================================
# Part 5 — Source type scoring
# ===================================================================


class TestSourceTypeScoring:
    def test_official(self) -> None:
        assert _compute_source_type_score(True, False) == 1.0

    def test_third_party_credible(self) -> None:
        assert _compute_source_type_score(False, True) == 0.7

    def test_unknown(self) -> None:
        assert _compute_source_type_score(False, False) == 0.2

    def test_official_overrides_tp(self) -> None:
        assert _compute_source_type_score(True, True) == 1.0


class TestSourceTrustLevelMapping:
    def test_official(self) -> None:
        assert _source_trust_level_from_score(1.0) == SourceTrustLevel.OFFICIAL

    def test_third_party_credible(self) -> None:
        assert _source_trust_level_from_score(0.7) == SourceTrustLevel.THIRD_PARTY_CREDIBLE

    def test_third_party_unverified(self) -> None:
        assert _source_trust_level_from_score(0.4) == SourceTrustLevel.THIRD_PARTY_UNVERIFIED

    def test_unknown(self) -> None:
        assert _source_trust_level_from_score(0.2) == SourceTrustLevel.UNKNOWN

    def test_boundary_official(self) -> None:
        assert _source_trust_level_from_score(0.9) == SourceTrustLevel.OFFICIAL
        assert _source_trust_level_from_score(0.89) == SourceTrustLevel.THIRD_PARTY_CREDIBLE


# ===================================================================
# Part 6 — compute_document_trust
# ===================================================================


class TestComputeDocumentTrust:
    def test_official_high_quality_fresh(self) -> None:
        s = _trust(authority=0.85, quality=0.8, is_official=True)
        assert s.overall > 0.8
        assert s.source_trust_level == SourceTrustLevel.OFFICIAL

    def test_unknown_low_quality_old(self) -> None:
        s = _trust(authority=0.2, quality=0.2, is_official=False, is_tp=False)
        assert s.overall < 0.4
        assert s.source_trust_level == SourceTrustLevel.UNKNOWN

    def test_third_party_credible(self) -> None:
        s = _trust(authority=0.6, quality=0.6, is_tp=True)
        assert s.source_trust_level == SourceTrustLevel.THIRD_PARTY_CREDIBLE

    def test_duplicate_penalty(self) -> None:
        s_normal = _trust(authority=0.7, quality=0.6)
        s_dup = _trust(authority=0.7, quality=0.6, is_dup=True)
        assert s_dup.overall < s_normal.overall

    def test_five_factors(self) -> None:
        s = _trust()
        assert len(s.factors) == 5
        factor_names = {f.name for f in s.factors}
        assert factor_names == {
            "authority", "freshness", "quality", "source_type", "duplicate_penalty",
        }

    def test_weights_sum_to_one(self) -> None:
        total = sum(_TRUST_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_score_is_clamped(self) -> None:
        s = _trust(authority=1.0, quality=1.0, is_official=True, is_tp=False)
        assert 0.0 <= s.overall <= 1.0

    def test_score_clamp_zero(self) -> None:
        s = _trust(authority=0.0, quality=0.0, is_official=False, is_dup=True)
        assert 0.0 <= s.overall <= 1.0

    def test_freshness_days_propagated(self) -> None:
        s = _trust(fetched_at=_FROZEN_FETCHED, now=_FROZEN_NOW)
        assert s.freshness_days > 0


# ===================================================================
# Part 7 — ProvenanceRecord
# ===================================================================


class TestProvenanceRecord:
    def test_construction(self) -> None:
        r = build_provenance_record(
            document_id="abc-123",
            url="https://example.com",
            final_url="https://example.com",
            document_type="homepage",
            source_provider="website",
            trust_level="official",
            trust_score=0.85,
            fetched_at=_FROZEN_FETCHED,
            is_official=True,
            is_duplicate=False,
            duplicate_of=None,
            freshness_days=165,
        )
        assert r.document_id == "abc-123"
        assert r.url == "https://example.com"
        assert r.trust_score == 0.85
        assert r.is_official is True
        assert r.is_duplicate is False
        assert r.duplicate_of is None

    def test_duplicate_record(self) -> None:
        r = build_provenance_record(
            document_id="dup-1",
            url="https://example.com/about",
            final_url="https://example.com/about",
            document_type="about",
            source_provider="website",
            trust_level="official",
            trust_score=0.6,
            fetched_at=_FROZEN_FETCHED,
            is_official=True,
            is_duplicate=True,
            duplicate_of="canonical-1",
            freshness_days=165,
        )
        assert r.is_duplicate is True
        assert r.duplicate_of == "canonical-1"


# ===================================================================
# Part 8 — TrustSummary
# ===================================================================


class TestTrustSummary:
    def test_empty_summary(self) -> None:
        s = compute_trust_summary([])
        assert s.total_documents == 0
        assert s.average_trust_score == 0.0
        assert s.high_trust_count == 0

    def test_single_document(self) -> None:
        t = _trust(authority=0.8, quality=0.8, is_official=True)
        s = compute_trust_summary([t])
        assert s.total_documents == 1
        assert s.average_trust_score > 0
        assert s.high_trust_count == 1

    def test_mixed_scores(self) -> None:
        scores = [
            _trust(authority=0.9, quality=0.9, is_official=True),   # high
            _trust(authority=0.5, quality=0.5, is_tp=True),         # medium
            _trust(authority=0.2, quality=0.2),                     # low/unknown
        ]
        s = compute_trust_summary(scores)
        assert s.total_documents == 3
        assert s.high_trust_count >= 1
        assert s.medium_trust_count >= 1

    def test_average_calculation(self) -> None:
        scores = [
            TrustScore(overall=0.8),
            TrustScore(overall=0.4),
        ]
        s = compute_trust_summary(scores)
        assert s.average_trust_score == pytest.approx(0.6, abs=0.01)

    def test_range(self) -> None:
        scores = [
            TrustScore(overall=0.2),
            TrustScore(overall=0.8),
            TrustScore(overall=0.5),
        ]
        s = compute_trust_summary(scores)
        assert s.trust_score_range[0] == pytest.approx(0.2, abs=0.01)
        assert s.trust_score_range[1] == pytest.approx(0.8, abs=0.01)

    def test_source_trust_levels_counted(self) -> None:
        scores = [
            TrustScore(overall=0.9, source_trust_level=SourceTrustLevel.OFFICIAL),
            TrustScore(overall=0.9, source_trust_level=SourceTrustLevel.OFFICIAL),
            TrustScore(overall=0.5, source_trust_level=SourceTrustLevel.THIRD_PARTY_CREDIBLE),
        ]
        s = compute_trust_summary(scores)
        assert s.source_trust_levels["official"] == 2
        assert s.source_trust_levels["third_party_credible"] == 1

    def test_all_unknown(self) -> None:
        scores = [
            TrustScore(overall=0.0, source_trust_level=SourceTrustLevel.UNKNOWN),
        ]
        s = compute_trust_summary(scores)
        assert s.unknown_trust_count == 1


# ===================================================================
# Part 9 — Determinism
# ===================================================================


class TestDeterminism:
    def test_identical_inputs_produce_identical_trust(self) -> None:
        kwargs = dict(
            authority_score=0.7,
            quality_score=0.6,
            fetched_at=_FROZEN_FETCHED,
            is_official=True,
            is_third_party_credible=False,
            is_duplicate=False,
            now=_FROZEN_NOW,
        )
        s1 = compute_document_trust(**kwargs)
        s2 = compute_document_trust(**kwargs)
        assert s1.overall == s2.overall
        assert s1.factors == s2.factors
        assert s1.freshness_days == s2.freshness_days

    def test_identical_inputs_produce_identical_summary(self) -> None:
        scores = [_trust(authority=0.7, quality=0.6, is_official=True) for _ in range(5)]
        s1 = compute_trust_summary(scores)
        s2 = compute_trust_summary(scores)
        assert s1 == s2

    def test_deterministic_across_orders(self) -> None:
        a = _trust(authority=0.8, quality=0.7, is_official=True)
        b = _trust(authority=0.3, quality=0.4, is_tp=True)
        s1 = compute_trust_summary([a, b])
        s2 = compute_trust_summary([b, a])
        assert s1.average_trust_score == s2.average_trust_score


# ===================================================================
# Part 10 — Serialization
# ===================================================================


class TestSerialization:
    def test_trust_score_json_roundtrip(self) -> None:
        original = _trust(authority=0.7, quality=0.6, is_official=True)
        data = original.model_dump()
        restored = TrustScore.model_validate(data)
        assert restored.overall == original.overall
        assert len(restored.factors) == len(original.factors)
        assert restored.source_trust_level == original.source_trust_level

    def test_provenance_record_json_roundtrip(self) -> None:
        original = build_provenance_record(
            document_id="doc-1",
            url="https://example.com",
            final_url="https://example.com",
            document_type="homepage",
            source_provider="website",
            trust_level="official",
            trust_score=0.85,
            fetched_at=_FROZEN_FETCHED,
            is_official=True,
            is_duplicate=False,
            duplicate_of=None,
            freshness_days=165,
        )
        json_str = original.model_dump_json()
        restored = ProvenanceRecord.model_validate_json(json_str)
        assert restored.document_id == original.document_id
        assert restored.trust_score == original.trust_score

    def test_trust_summary_json_roundtrip(self) -> None:
        scores = [_trust(authority=0.7, quality=0.6, is_official=True)]
        original = compute_trust_summary(scores)
        data = original.model_dump()
        restored = TrustSummary.model_validate(data)
        assert restored.total_documents == original.total_documents
        assert restored.average_trust_score == original.average_trust_score

    def test_trust_score_with_none_fetched_at(self) -> None:
        s = TrustScore(overall=0.5)
        data = s.model_dump()
        restored = TrustScore.model_validate(data)
        assert restored.overall == 0.5


# ===================================================================
# Part 11 — Backward compatibility
# ===================================================================


class TestBackwardCompatibility:
    def test_document_metadata_defaults(self) -> None:
        from predictron_engine.evidence.models import DocumentMetadata

        meta = DocumentMetadata()
        assert meta.trust_score is None
        assert meta.provenance == []
        assert meta.trust_level == "unknown"
        assert meta.authority_score == 0.0

    def test_intelligence_summary_defaults(self) -> None:
        from predictron_engine.evidence.models import IntelligenceSummary

        s = IntelligenceSummary()
        assert s.trust_summary is None
        assert s.documents_input == 0

    def test_evidence_bundle_defaults(self) -> None:
        from predictron_engine.evidence.models import EvidenceBundle

        bundle = EvidenceBundle(startup_name="Test")
        assert bundle.trust_summary is None
        assert bundle.documents == []

    def test_evidence_item_defaults(self) -> None:
        from predictron_engine.models.report import EvidenceItem

        item = EvidenceItem(
            domain="industry",
            category="regulatory",
            statement="Test",
            source="test",
        )
        assert item.provenance_record is None

    def test_extracted_features_defaults(self) -> None:
        from predictron_engine.models.extracted_features import ExtractedFeatures

        features = ExtractedFeatures()
        assert features.provenance_document_ids == []

    def test_empty_bundle_has_no_trust(self) -> None:
        from predictron_engine.evidence.models import EvidenceBundle

        bundle = EvidenceBundle.empty("Startup")
        assert bundle.trust_summary is None

    def test_metadata_preserves_existing_fields(self) -> None:
        from predictron_engine.evidence.models import DocumentMetadata, DocumentType

        meta = DocumentMetadata(
            document_type=DocumentType.HOMEPAGE,
            authority_score=0.85,
            quality_score=0.7,
            priority=1,
            canonical_url="https://example.com",
            language="en",
            word_count=500,
            heading_count=5,
            table_count=1,
            list_count=3,
            content_hash="abc123",
            is_duplicate=False,
            duplicate_of=None,
            trust_level="official",
            source_provider="website",
        )
        assert meta.document_type == DocumentType.HOMEPAGE
        assert meta.authority_score == 0.85
        assert meta.source_provider == "website"
        assert meta.trust_score is None


# ===================================================================
# Part 12 — Edge cases
# ===================================================================


class TestEdgeCases:
    def test_zero_authority_zero_quality(self) -> None:
        s = _trust(authority=0.0, quality=0.0)
        assert s.overall > 0.0  # source_type + freshness contribute
        assert s.overall < 0.5

    def test_perfect_scores(self) -> None:
        s = _trust(authority=1.0, quality=1.0, is_official=True)
        assert s.overall >= 0.85

    def test_duplicate_at_perfect_scores(self) -> None:
        s_normal = _trust(authority=1.0, quality=1.0, is_official=True)
        s_dup = _trust(authority=1.0, quality=1.0, is_official=True, is_dup=True)
        assert s_dup.overall < s_normal.overall
        assert s_dup.overall > 0.7  # still high because other factors are perfect

    def test_summary_single_high(self) -> None:
        t = TrustScore(overall=0.7)
        s = compute_trust_summary([t])
        assert s.high_trust_count == 1
        assert s.medium_trust_count == 0

    def test_summary_single_medium(self) -> None:
        t = TrustScore(overall=0.5)
        s = compute_trust_summary([t])
        assert s.medium_trust_count == 1
        assert s.high_trust_count == 0

    def test_summary_single_low(self) -> None:
        t = TrustScore(overall=0.1)
        s = compute_trust_summary([t])
        assert s.low_trust_count == 1

    def test_summary_boundary_high_medium(self) -> None:
        t_high = TrustScore(overall=0.7)
        t_med = TrustScore(overall=0.699)
        s = compute_trust_summary([t_high, t_med])
        assert s.high_trust_count == 1
        assert s.medium_trust_count == 1

    def test_summary_boundary_medium_low(self) -> None:
        t_med = TrustScore(overall=0.4)
        t_low = TrustScore(overall=0.399)
        s = compute_trust_summary([t_med, t_low])
        assert s.medium_trust_count == 1
        assert s.low_trust_count == 1

    def test_provenance_record_with_none_fetched_at(self) -> None:
        r = build_provenance_record(
            document_id="d1",
            url="https://x.com",
            final_url="https://x.com",
            document_type="unknown",
            source_provider="test",
            trust_level="unknown",
            trust_score=0.0,
            fetched_at=None,
            is_official=False,
            is_duplicate=False,
            duplicate_of=None,
            freshness_days=0,
        )
        assert r.fetched_at is None

    def test_many_documents_summary(self) -> None:
        scores = [_trust(authority=float(i) / 10, quality=float(i) / 10) for i in range(10)]
        s = compute_trust_summary(scores)
        assert s.total_documents == 10
        assert 0.0 <= s.average_trust_score <= 1.0


# ===================================================================
# Part 13 — Integration with document_intelligence enrich_documents
# ===================================================================


class TestDocumentIntelligenceIntegration:
    """Verify that enrich_documents produces trust scores on each document."""

    def test_enriched_doc_has_trust_score(self) -> None:
        from predictron_engine.evidence.document_intelligence import enrich_documents
        from predictron_engine.evidence.models import (
            DocumentStatus,
            EvidenceDocument,
            make_document_id,
        )

        url = "https://example.com"
        doc = EvidenceDocument(
            id=make_document_id(url),
            original_url=url,
            url=url,
            page_type="homepage",
            status=DocumentStatus.SUCCESS,
            fetched_at=_FROZEN_FETCHED,
            response_time_ms=100,
            http_status=200,
            title="Example",
            text="Hello world this is a test page with enough content to be meaningful.",
        )
        enriched, summary = enrich_documents(
            [doc],
            official_host="example.com",
            source_provider="website",
        )
        assert len(enriched) == 1
        meta = enriched[0].metadata
        assert meta is not None
        assert meta.trust_score is not None
        assert 0.0 <= meta.trust_score.overall <= 1.0
        assert len(meta.trust_score.factors) == 5

    def test_enriched_doc_has_provenance(self) -> None:
        from predictron_engine.evidence.document_intelligence import enrich_documents
        from predictron_engine.evidence.models import (
            DocumentStatus,
            EvidenceDocument,
            make_document_id,
        )

        url = "https://example.com"
        doc = EvidenceDocument(
            id=make_document_id(url),
            original_url=url,
            url=url,
            page_type="homepage",
            status=DocumentStatus.SUCCESS,
            fetched_at=_FROZEN_FETCHED,
            response_time_ms=100,
            http_status=200,
            title="Example",
            text="Hello world this is a test page with enough content to be meaningful.",
        )
        enriched, _ = enrich_documents([doc], official_host="example.com")
        meta = enriched[0].metadata
        assert meta is not None
        assert len(meta.provenance) == 1
        rec = meta.provenance[0]
        assert rec.document_id == doc.id
        assert "example.com" in rec.url
        assert rec.is_official is True

    def test_intelligence_summary_has_trust_summary(self) -> None:
        from predictron_engine.evidence.document_intelligence import enrich_documents
        from predictron_engine.evidence.models import (
            DocumentStatus,
            EvidenceDocument,
            make_document_id,
        )

        doc = EvidenceDocument(
            id=make_document_id("https://a.com"),
            original_url="https://a.com",
            url="https://a.com",
            page_type="homepage",
            status=DocumentStatus.SUCCESS,
            fetched_at=_FROZEN_FETCHED,
            response_time_ms=100,
            http_status=200,
            title="A",
            text="Some meaningful content here that is long enough.",
        )
        _, summary = enrich_documents([doc])
        assert summary.trust_summary is not None
        assert summary.trust_summary.total_documents == 1

    def test_failed_doc_skips_trust(self) -> None:
        from predictron_engine.evidence.document_intelligence import enrich_documents
        from predictron_engine.evidence.models import (
            DocumentStatus,
            EvidenceDocument,
            make_document_id,
        )

        doc = EvidenceDocument(
            id=make_document_id("https://fail.com"),
            original_url="https://fail.com",
            url="https://fail.com",
            page_type="homepage",
            status=DocumentStatus.FAILED,
            fetched_at=_FROZEN_FETCHED,
            response_time_ms=100,
            http_status=500,
            title="Error",
            text="Error occurred",
        )
        enriched, summary = enrich_documents([doc])
        assert enriched[0].metadata is None
        assert summary.trust_summary is not None
        assert summary.trust_summary.total_documents == 0
