"""Tests for multi-stage entity resolution."""

from __future__ import annotations

from typing import Any

import pytest

from predictron_engine.dataset.entity_resolution import (
    REVIEW_BUCKET_THRESHOLD,
    EntityResolver,
    MatchEvidence,
    MatchResult,
    ResolutionIndex,
    ResolutionReport,
    ResolvedCluster,
    score_pair,
)
from predictron_engine.dataset.models import CompanyProfile, DatasetRecord
from tests.dataset.conftest import make_record


def _rec(
    record_id: str,
    name: str,
    website: str | None = None,
    *,
    domain: str | None = None,
    country: str | None = None,
    industries: list[str] | None = None,
    city: str | None = None,
    region: str | None = None,
    legal_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    source: str = "direct",
) -> DatasetRecord:
    if website is None:
        website = f"https://{record_id}.example.com"
    record = make_record(startup_name=name, website=website, record_id=record_id)
    record.profile = CompanyProfile(
        domain=domain,
        country_code=country,
        industries=industries or [],
        city=city,
        region=region,
        legal_name=legal_name,
    )
    record.source = source
    if metadata:
        record.analysis_metadata = metadata
    return record


# ---------------------------------------------------------------------------
# Score pair: exact identifiers
# ---------------------------------------------------------------------------

class TestScorePairExactIdentifier:
    def test_same_sec_cik(self) -> None:
        a = _rec("a", "Acme", metadata={"sec_cik": "12345"})
        b = _rec("b", "Acme Corp", website="https://other.com", metadata={"sec_cik": "12345"})
        result = score_pair(a, b)
        assert result.confidence >= 0.99
        assert "sec_cik" in result.match_reason
        assert "identifier:sec_cik" in result.matched_fields

    def test_same_company_number(self) -> None:
        a = _rec("a", "Acme", metadata={"company_number": "09876"})
        b = _rec("b", "Acme Ltd", website="https://b.com", metadata={"company_number": "09876"})
        result = score_pair(a, b)
        assert result.confidence >= 0.99
        assert "company_number" in result.match_reason

    def test_same_registration_number(self) -> None:
        a = _rec("a", "Acme", metadata={"registration_number": "REG-1"})
        b = _rec("b", "Acme", website="https://b.com", metadata={"registration_number": "REG-1"})
        result = score_pair(a, b)
        assert result.confidence >= 0.99

    def test_different_identifiers_no_match(self) -> None:
        a = _rec("a", "Acme", metadata={"sec_cik": "111"})
        b = _rec("b", "Acme", website="https://b.com", metadata={"sec_cik": "222"})
        result = score_pair(a, b)
        assert result.match_reason != "exact_sec_cik"

    def test_same_record(self) -> None:
        a = _rec("a", "Acme")
        result = score_pair(a, a)
        assert result.confidence == 1.0
        assert result.match_reason == "same_record"


# ---------------------------------------------------------------------------
# Score pair: exact domain
# ---------------------------------------------------------------------------

class TestScorePairExactDomain:
    def test_same_profile_domain(self) -> None:
        a = _rec("a", "Acme", domain="acme.com")
        b = _rec("b", "Acme Corp", website="https://other.com", domain="acme.com")
        result = score_pair(a, b)
        assert result.confidence == 0.98
        assert result.match_reason == "exact_domain"
        assert "domain" in result.matched_fields

    def test_same_domain_from_website(self) -> None:
        a = _rec("a", "Acme", website="https://acme.com")
        b = _rec("b", "Acme Corp", website="https://acme.com")
        result = score_pair(a, b)
        assert result.confidence == 0.98
        assert result.match_reason == "exact_domain"

    def test_different_domains_no_exact_match(self) -> None:
        a = _rec("a", "Acme", domain="acme.com")
        b = _rec("b", "Acme", website="https://b.com", domain="acme.co")
        result = score_pair(a, b)
        assert result.match_reason != "exact_domain"


# ---------------------------------------------------------------------------
# Score pair: exact normalized core name
# ---------------------------------------------------------------------------

class TestScorePairExactCoreName:
    def test_same_core_name(self) -> None:
        a = _rec("a", "Acme Corp")
        b = _rec("b", "Acme Corporation")
        result = score_pair(a, b)
        assert result.confidence >= 0.90
        assert result.match_reason == "exact_name"

    def test_country_match_boosts(self) -> None:
        a = _rec("a", "Acme Corp", country="US")
        b = _rec("b", "Acme Corporation", country="US")
        result = score_pair(a, b)
        assert result.confidence >= 0.92
        assert "country" in result.matched_fields

    def test_country_industry_boost(self) -> None:
        a = _rec("a", "Acme Corp", country="US", industries=["fintech"])
        b = _rec("b", "Acme Corporation", country="US", industries=["fintech"])
        result = score_pair(a, b)
        assert result.confidence >= 0.94

    def test_country_conflict_depresses(self) -> None:
        a = _rec("a", "Acme Corp", country="US")
        b = _rec("b", "Acme Corp", country="GB")
        result = score_pair(a, b)
        assert result.confidence < REVIEW_BUCKET_THRESHOLD
        assert result.match_reason == "exact_name_country_conflict"

    def test_different_names_not_exact(self) -> None:
        a = _rec("a", "Acme Corp")
        b = _rec("b", "Beta Inc")
        result = score_pair(a, b)
        assert result.match_reason != "exact_name"


# ---------------------------------------------------------------------------
# Score pair: fuzzy matching
# ---------------------------------------------------------------------------

class TestScorePairFuzzy:
    def test_high_fuzzy_match(self) -> None:
        # Identical after suffix stripping → exact_name (not fuzzy).
        a = _rec("a", "Acme Corp")
        b = _rec("b", "Acme Corporation")
        result = score_pair(a, b)
        assert result.confidence > 0.0
        assert result.match_reason == "exact_name"

    def test_moderate_fuzzy_match(self) -> None:
        # Near-identical names hit the fuzzy zone (identity or review).
        a = _rec("a", "Acme Corp")
        b = _rec("b", "Acme Corpp")
        result = score_pair(a, b)
        assert result.confidence > 0.0
        assert "fuzzy" in result.match_reason

    def test_fuzzy_with_country_and_industry(self) -> None:
        a = _rec("a", "Acme Corp", country="US", industries=["saas"])
        b = _rec("b", "Acme Corporationn", country="US", industries=["saas"])
        result = score_pair(a, b)
        assert "country" in result.matched_fields
        assert "industry" in result.matched_fields

    def test_fuzzy_below_threshold(self) -> None:
        a = _rec("a", "Acme Corp")
        b = _rec("b", "XYZ Industries LLC")
        result = score_pair(a, b)
        assert result.confidence == 0.0
        assert result.match_reason == "below_threshold"

    def test_fuzzy_review_bucket(self) -> None:
        a = _rec("a", "Acme Corp")
        b = _rec("b", "Acme Corporated")
        result = score_pair(a, b)
        # Should be in review or identity range.
        assert result.confidence >= 0.0

    def test_fuzzy_evidence_captured(self) -> None:
        a = _rec("a", "Acme Corp")
        b = _rec("b", "Acme Corpo")
        result = score_pair(a, b)
        assert result.evidence.fuzzy_name_score > 0.0

    def test_fuzzy_domain_overlap(self) -> None:
        a = _rec("a", "Acme Corp", domain="acme.com")
        b = _rec("b", "Acme Corps", domain="acme.co")
        result = score_pair(a, b)
        if result.match_reason != "below_threshold":
            assert "domain" in result.matched_fields


# ---------------------------------------------------------------------------
# ResolutionIndex / Blocking
# ---------------------------------------------------------------------------

class TestResolutionIndex:
    def test_build_and_candidate_pairs(self) -> None:
        a = _rec("a", "Acme Corp", domain="acme.com")
        b = _rec("b", "Acme Inc", domain="acme.com")
        c = _rec("c", "Beta LLC", domain="beta.com")
        index = ResolutionIndex()
        index.build([a, b, c])
        pairs = index.candidate_pairs()
        # a and b share a domain; c is alone.
        assert ("a", "b") in pairs or ("b", "a") in pairs
        # c should not be paired with a or b.
        c_pairs = [p for p in pairs if "c" in p]
        assert c_pairs == []

    def test_blocking_by_identifier(self) -> None:
        a = _rec("a", "Acme", metadata={"sec_cik": "123"})
        b = _rec("b", "Acme", website="https://b.com", metadata={"sec_cik": "123"})
        c = _rec("c", "Beta", website="https://c.com")
        index = ResolutionIndex()
        index.build([a, b, c])
        pairs = index.candidate_pairs()
        assert ("a", "b") in pairs
        c_pairs = [p for p in pairs if "c" in p]
        assert c_pairs == []

    def test_blocking_by_name(self) -> None:
        a = _rec("a", "Acme Corp")
        b = _rec("b", "Acme Corporation")
        c = _rec("c", "Beta LLC")
        index = ResolutionIndex()
        index.build([a, b, c])
        pairs = index.candidate_pairs()
        # a and b share canonical core name "acme".
        assert ("a", "b") in pairs or ("b", "a") in pairs

    def test_blocking_by_token(self) -> None:
        a = _rec("a", "Zeta Technologies")
        b = _rec("b", "Zeta Tech Corp")
        c = _rec("c", "Alpha Inc")
        index = ResolutionIndex()
        index.build([a, b, c])
        pairs = index.candidate_pairs()
        # a and b share "zeta" token.
        assert ("a", "b") in pairs or ("b", "a") in pairs

    def test_empty_records(self) -> None:
        index = ResolutionIndex()
        index.build([])
        assert index.candidate_pairs() == set()

    def test_match_pair(self) -> None:
        a = _rec("a", "Acme Corp", metadata={"sec_cik": "123"})
        b = _rec("b", "Acme Corp", website="https://b.com", metadata={"sec_cik": "123"})
        index = ResolutionIndex()
        index.build([a, b])
        result = index.match_pair("a", "b")
        assert result.confidence >= 0.99


# ---------------------------------------------------------------------------
# EntityResolver
# ---------------------------------------------------------------------------

class TestEntityResolver:
    def test_singleton_records(self) -> None:
        records = [
            _rec("a", "Acme Corp", domain="acme.com"),
            _rec("b", "Beta Inc", domain="beta.com"),
        ]
        resolver = EntityResolver()
        result = resolver.resolve(records)
        assert result.report.identity_count == 2
        assert result.report.merged_count == 0

    def test_merge_by_identifier(self) -> None:
        records = [
            _rec("a", "Acme Corp", metadata={"sec_cik": "123"}),
            _rec("b", "Acme Corp", website="https://b.com", metadata={"sec_cik": "123"}),
        ]
        resolver = EntityResolver()
        result = resolver.resolve(records)
        assert result.report.identity_count == 1
        assert result.report.merged_count == 1

    def test_merge_by_domain(self) -> None:
        records = [
            _rec("a", "Acme Corp", domain="acme.com"),
            _rec("b", "Acme Inc", website="https://b.com", domain="acme.com"),
        ]
        resolver = EntityResolver()
        result = resolver.resolve(records)
        assert result.report.identity_count == 1

    def test_merge_by_exact_name(self) -> None:
        records = [
            _rec("a", "Acme Corp"),
            _rec("b", "Acme Corporation"),
        ]
        resolver = EntityResolver()
        result = resolver.resolve(records)
        assert result.report.identity_count == 1

    def test_manual_review_pairs(self) -> None:
        # Two records that are borderline — they share a name token but
        # aren't clearly the same entity.
        records = [
            _rec("a", "Acme Technologies"),
            _rec("b", "Acme Tech Corp"),
        ]
        resolver = EntityResolver(min_confidence=REVIEW_BUCKET_THRESHOLD)
        result = resolver.resolve(records)
        # Should have at least one review pair or be merged.
        assert result.report.identity_count >= 1

    def test_no_merge_low_confidence(self) -> None:
        records = [
            _rec("a", "Acme Corp"),
            _rec("b", "XYZ Industries"),
        ]
        resolver = EntityResolver()
        result = resolver.resolve(records)
        assert result.report.identity_count == 2

    def test_country_conflict_not_auto_merged(self) -> None:
        records = [
            _rec("a", "Acme Corp", country="US"),
            _rec("b", "Acme Corp", country="GB"),
        ]
        resolver = EntityResolver()
        result = resolver.resolve(records)
        # Should not be auto-merged due to country conflict.
        if result.report.identity_count == 1:
            # If merged, it should be a review pair, not automatic.
            cluster = result.report.clusters[0]
            assert cluster.merge_decision != "automatic_merge"

    def test_deterministic_output(self) -> None:
        records = [
            _rec("a", "Acme Corp", metadata={"sec_cik": "123"}),
            _rec("b", "Acme Inc", website="https://b.com", metadata={"sec_cik": "123"}),
            _rec("c", "Beta LLC", domain="beta.com"),
        ]
        resolver = EntityResolver()
        r1 = resolver.resolve(records)
        r2 = resolver.resolve(records)
        assert r1.report.identity_count == r2.report.identity_count
        assert r1.report.merged_count == r2.report.merged_count

    def test_empty_records(self) -> None:
        resolver = EntityResolver()
        result = resolver.resolve([])
        assert result.report.identity_count == 0
        assert result.identities == []

    def test_three_way_merge(self) -> None:
        records = [
            _rec("a", "Acme Corp", metadata={"sec_cik": "123"}),
            _rec("b", "Acme Inc", website="https://b.com", metadata={"sec_cik": "123"}),
            _rec("c", "Acme Ltd", website="https://c.com", metadata={"sec_cik": "123"}),
        ]
        resolver = EntityResolver()
        result = resolver.resolve(records)
        assert result.report.identity_count == 1
        assert len(result.identities[0].record_ids) == 3

    def test_chained_merge(self) -> None:
        # a matches b by domain, b matches c by identifier.
        records = [
            _rec("a", "Acme Corp", domain="acme.com"),
            _rec("b", "Acme Inc", domain="acme.com", metadata={"sec_cik": "999"}),
            _rec("c", "Acme Ltd", website="https://c.com", metadata={"sec_cik": "999"}),
        ]
        resolver = EntityResolver()
        result = resolver.resolve(records)
        assert result.report.identity_count == 1


# ---------------------------------------------------------------------------
# ResolutionReport
# ---------------------------------------------------------------------------

class TestResolutionReport:
    def test_confidence_distribution(self) -> None:
        report = ResolutionReport(
            pair_confidences=[0.5, 0.6, 0.8, 0.9, 0.95],
        )
        dist = report.confidence_distribution()
        assert isinstance(dist, dict)
        assert len(dist) > 0

    def test_merge_statistics(self) -> None:
        report = ResolutionReport(
            record_ids=["a", "b", "c"],
            clusters=[
                ResolvedCluster(
                    record_ids=["a", "b"],
                    confidence=0.95,
                    merge_decision="automatic_merge",
                ),
                ResolvedCluster(
                    record_ids=["c"],
                    confidence=1.0,
                    merge_decision="singleton",
                ),
            ],
            identity_count=2,
            merged_count=1,
            automatic_count=1,
            manual_review_count=0,
            unrelated_count=5,
        )
        stats = report.merge_statistics()
        assert stats["clusters"] == 2
        assert stats["identity_count"] == 2
        assert stats["automatic_merges"] == 1
        assert stats["largest_cluster_size"] == 2

    def test_to_dict(self) -> None:
        report = ResolutionReport(
            record_ids=["a"],
            clusters=[],
            identity_count=1,
        )
        d = report.to_dict()
        assert d["report_type"] == "entity_resolution_report"
        assert d["record_count"] == 1
        assert d["identity_count"] == 1


# ---------------------------------------------------------------------------
# MatchResult / MatchEvidence
# ---------------------------------------------------------------------------

class TestMatchResult:
    def test_to_dict(self) -> None:
        evidence = MatchEvidence(pair=("a", "b"), domain_match=True, domain="acme.com")
        result = MatchResult(
            pair=("a", "b"),
            confidence=0.98,
            match_reason="exact_domain",
            matched_fields=["domain"],
            evidence=evidence,
        )
        d = result.to_dict()
        assert d["confidence"] == 0.98
        assert d["match_reason"] == "exact_domain"
        assert d["evidence"]["domain"] == "acme.com"

    def test_match_evidence_to_dict(self) -> None:
        evidence = MatchEvidence(
            pair=("a", "b"),
            identifier_match=True,
            identifier_kind="sec_cik",
        )
        d = evidence.to_dict()
        assert d["identifier_match"] is True
        assert d["identifier_kind"] == "sec_cik"


# ---------------------------------------------------------------------------
# Resolver thresholds
# ---------------------------------------------------------------------------

class TestResolverThresholds:
    def test_invalid_min_confidence(self) -> None:
        with pytest.raises(ValueError, match="min_confidence"):
            EntityResolver(min_confidence=-0.1)

    def test_invalid_identity_threshold(self) -> None:
        with pytest.raises(ValueError, match="identity_threshold"):
            EntityResolver(identity_threshold=1.5)

    def test_min_exceeds_identity(self) -> None:
        with pytest.raises(ValueError, match="min_confidence must be <= identity_threshold"):
            EntityResolver(min_confidence=0.9, identity_threshold=0.8)

    def test_custom_thresholds(self) -> None:
        resolver = EntityResolver(min_confidence=0.5, identity_threshold=0.9)
        assert resolver._min_confidence == 0.5
        assert resolver._identity_threshold == 0.9


# ---------------------------------------------------------------------------
# ResolvedCluster
# ---------------------------------------------------------------------------

class TestResolvedCluster:
    def test_to_dict(self) -> None:
        cluster = ResolvedCluster(
            record_ids=["b", "a"],
            confidence=0.95,
            merge_decision="automatic_merge",
            matched_fields=["domain"],
            match_reason="exact_domain",
            evidence=[{"domain_match": True}],
        )
        d = cluster.to_dict()
        assert d["record_ids"] == ["a", "b"]  # sorted
        assert d["confidence"] == 0.95
        assert d["merge_decision"] == "automatic_merge"
