"""Tests for deterministic company identity resolution (CIH Phase 1).

Covers CompanyIdentityResolver, compute_company_id, fallback slug
normalization, and the pure report-to-snapshot field mapping.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.services.companies import (
    CompanyIdentityResolver,
    build_snapshot_inputs,
    compute_company_id,
    normalize_fallback_slug,
    utc_now,
)

RESOLVER = CompanyIdentityResolver()


def _identity(name, website=None):
    return RESOLVER.resolve(name, website)


# ── Priority 1: canonical domain ────────────────────────────────────────


def test_priority_domain_wins_over_name():
    i = _identity("Acme Inc", "https://acme.com")
    assert i.match_source == "canonical_domain"
    assert i.canonical_domain == "acme.com"
    assert i.canonical_name == "acme"


def test_domain_extraction_strips_www():
    assert _identity("Acme", "https://www.acme.com").canonical_domain == "acme.com"


def test_domain_extraction_strips_scheme():
    assert _identity("Acme", "http://acme.com").canonical_domain == "acme.com"


def test_domain_extraction_lowercases():
    assert _identity("Acme", "HTTPS://ACME.COM").canonical_domain == "acme.com"


def test_domain_without_scheme_parsed():
    assert _identity("Acme", "acme.com").canonical_domain == "acme.com"


def test_domain_with_path_keeps_host_only():
    assert (
        _identity("Acme", "https://acme.com/about?x=1").canonical_domain == "acme.com"
    )


def test_subdomain_preserved():
    assert _identity("Acme", "https://app.acme.com").canonical_domain == "app.acme.com"


def test_empty_website_yields_no_domain():
    assert _identity("Acme", "").canonical_domain is None


def test_whitespace_website_yields_no_domain():
    assert _identity("Acme", "   ").canonical_domain is None


def test_none_website_yields_no_domain():
    assert _identity("Acme", None).canonical_domain is None


def test_invalid_website_yields_no_domain():
    assert _identity("Acme", "http:///missing-host").canonical_domain is None


def test_domain_variants_resolve_to_same_id():
    a = _identity("Acme Inc", "https://acme.com")
    b = _identity("Acme Incorporated", "www.acme.com")
    assert a.company_id == b.company_id


# ── Priority 2: canonical company name ──────────────────────────────────


def test_priority_name_without_website():
    i = _identity("Acme Inc", None)
    assert i.match_source == "canonical_name"
    assert i.canonical_name == "acme"


def test_name_suffix_stripping_consistent():
    ids = {
        _identity(name).company_id
        for name in ["Acme Inc", "Acme Incorporated", "ACME LLC", "Acme Corp"]
    }
    assert len(ids) == 1


def test_name_core_name_drops_punctuation():
    assert _identity("Stripe, Inc.").canonical_name == "stripe"


def test_name_casefolded():
    assert _identity("STRIPE").canonical_name == "stripe"


def test_name_unicode_normalized():
    assert _identity("Café GmbH").canonical_name == "caf"


def test_name_blank_falls_to_slug():
    i = _identity("   ", None)
    assert i.match_source == "fallback_slug"
    assert i.fallback_slug == "unknown"


def test_name_only_deterministic_id():
    assert _identity("Stripe Inc").company_id == _identity("Stripe Inc").company_id


def test_name_only_distinct_names_distinct_ids():
    assert _identity("Alpha Inc").company_id != _identity("Beta Inc").company_id


# ── Priority 3: fallback slug ───────────────────────────────────────────


def test_fallback_slug_for_blank():
    assert normalize_fallback_slug("") == "unknown"
    assert normalize_fallback_slug(" ") == "unknown"
    assert normalize_fallback_slug("###") == "unknown"


def test_fallback_slug_slugs_the_name():
    assert normalize_fallback_slug("Nice Co. Ltd") == "nice-co-ltd"


def test_fallback_slug_lowercases():
    assert normalize_fallback_slug("STRIPE") == "stripe"


def test_fallback_slug_collapses_separators():
    assert normalize_fallback_slug("Big   Blue    Corp") == "big-blue-corp"


def test_fallback_slug_truncated_to_128():
    long = "x" * 300
    slug = normalize_fallback_slug(long)
    assert len(slug) == 128


def test_fallback_slug_id_stable_for_blank():
    i1 = _identity("   ", None)
    i2 = _identity("", None)
    assert i1.company_id == i2.company_id == compute_company_id("unknown")


# ── compute_company_id determinism ──────────────────────────────────────


def test_compute_company_id_stable():
    assert compute_company_id("acme.com") == compute_company_id("acme.com")


def test_compute_company_id_distinct_material_distinct_id():
    assert compute_company_id("acme.com") != compute_company_id("acme")


def test_compute_company_id_is_sha256_hex():
    import hashlib

    assert compute_company_id("acme.com") == hashlib.sha256(
        b"acme.com"
    ).hexdigest()


def test_compute_company_id_empty_uses_unknown():
    assert compute_company_id("") == compute_company_id("unknown")


def test_compute_company_id_returns_64_hex_chars():
    assert len(compute_company_id("acme.com")) == 64


# ── Resolver result shape ───────────────────────────────────────────────


def test_resolve_preserves_primary_name():
    i = _identity("Acme Inc", "https://acme.com")
    assert i.primary_name == "Acme Inc"


def test_resolve_preserves_website():
    i = _identity("Acme Inc", "https://acme.com")
    assert i.website == "https://acme.com"


def test_resolve_website_none_stays_none():
    assert _identity("Acme Inc").website is None


def test_resolve_canonical_name_key_derived():
    i = _identity("Acme Inc", None)
    assert i.canonical_name_key


def test_resolve_canonical_name_key_maximal_compression():
    from predictron_engine.dataset.company_name import canonical_name_key

    i = _identity("Acme Corp", None)
    assert i.canonical_name_key == canonical_name_key("Acme Corp")


def test_resolve_is_repeatable_and_frozen():
    a = _identity("Acme Inc", "https://acme.com")
    b = _identity("Acme Inc", "https://acme.com")
    assert a == b


def test_resolve_domain_id_differs_from_name_only_id():
    with_domain = _identity("Acme Inc", "https://acme.com")
    without = _identity("Acme Inc", None)
    assert with_domain.company_id != without.company_id


def test_resolver_instance_reuse():
    r = CompanyIdentityResolver()
    assert r.resolve("X", None).company_id == r.resolve("X", None).company_id


# ── build_snapshot_inputs (pure report mapping) ─────────────────────────


def _make_report():
    from predictron_engine.models.extracted_features import ExtractedFeatures
    from predictron_engine.models.report import (
        DecisionCategory,
        InvestmentDecision,
        InvestmentReadiness,
        Report,
        ScoreResult,
    )
    from predictron_engine.models.startup import Startup

    return Report(
        startup=Startup(
            name="Acme Inc",
            website="https://acme.com",
            description="desc",
        ),
        features=ExtractedFeatures(),
        overall_score=77.5,
        overall_confidence=0.83,
        scores=[
            ScoreResult(dimension="market_opportunity", score=80.0),
            ScoreResult(dimension="founder_quality", score=70.0),
        ],
        investment_decision=InvestmentDecision(
            category=DecisionCategory.INVEST,
            conviction="high",
            composite_score=75.2,
        ),
        investment_readiness=InvestmentReadiness(readiness_score=68.0),
    )


def _make_request(website="https://acme.com"):
    from app.schemas.analysis import StartupAnalysisRequest

    return StartupAnalysisRequest(
        startup_name="Acme Inc",
        website=website,
        description="A sufficiently long description for validation.",
    )


def _make_response():
    from app.schemas.analysis import StartupAnalysisResponse

    return StartupAnalysisResponse(
        startup_name="Acme Inc",
        venture_score=77.5,
        market_score=80.0,
        founder_score=70.0,
        traction_score=60.0,
        confidence=0.83,
    )


def test_build_snapshot_inputs_maps_decision():
    inputs = build_snapshot_inputs(_make_request(), _make_report(), _make_response())
    assert inputs.decision == "invest"


def test_build_snapshot_inputs_maps_confidence():
    inputs = build_snapshot_inputs(_make_request(), _make_report(), _make_response())
    assert inputs.confidence == 0.83


def test_build_snapshot_inputs_maps_composite_from_decision():
    inputs = build_snapshot_inputs(_make_request(), _make_report(), _make_response())
    assert inputs.composite_score == 75.2


def test_build_snapshot_inputs_maps_readiness():
    inputs = build_snapshot_inputs(_make_request(), _make_report(), _make_response())
    assert inputs.readiness_score == 68.0


def test_build_snapshot_inputs_maps_dimension_scores():
    inputs = build_snapshot_inputs(_make_request(), _make_report(), _make_response())
    assert inputs.dimension_scores == {
        "market_opportunity": 80.0,
        "founder_quality": 70.0,
    }


def test_build_snapshot_inputs_keeps_name():
    inputs = build_snapshot_inputs(_make_request(), _make_report(), _make_response())
    assert inputs.startup_name == "Acme Inc"


def test_build_snapshot_inputs_keeps_website():
    inputs = build_snapshot_inputs(_make_request(), _make_report(), _make_response())
    assert inputs.website == "https://acme.com/"


def test_build_snapshot_inputs_falls_back_to_report_website():
    inputs = build_snapshot_inputs(_make_request(website=None), _make_report(), _make_response())
    assert inputs.website == "https://acme.com"


def test_build_snapshot_inputs_no_website_anywhere():
    report = _make_report()
    report.startup.website = ""
    inputs = build_snapshot_inputs(_make_request(website=None), report, _make_response())
    assert inputs.website is None


def test_build_snapshot_inputs_without_decision_uses_overall_score():
    report = _make_report()
    report.investment_decision = None
    inputs = build_snapshot_inputs(_make_request(), report, _make_response())
    assert inputs.decision is None
    assert inputs.composite_score == 77.5


def test_build_snapshot_inputs_without_readiness():
    report = _make_report()
    report.investment_readiness = None
    inputs = build_snapshot_inputs(_make_request(), report, _make_response())
    assert inputs.readiness_score is None


def test_build_snapshot_inputs_uses_metadata_timestamp():
    from predictron_engine.models.report import AnalysisMetadata

    report = _make_report()
    report.analysis_metadata = AnalysisMetadata(
        engine_version="0.12.1",
        processing_time_ms=5.0,
        timestamp=datetime(2025, 5, 1, 12, 0, tzinfo=UTC),
    )
    inputs = build_snapshot_inputs(_make_request(), report, _make_response())
    assert inputs.created_at == datetime(2025, 5, 1, 12, 0, tzinfo=UTC)


def test_build_snapshot_inputs_default_timestamp_is_utc():
    inputs = build_snapshot_inputs(_make_request(), _make_report(), _make_response())
    assert inputs.created_at.tzinfo is not None


@pytest.mark.parametrize(
    ("name", "website", "expected_source"),
    [
        ("Acme Inc", "https://acme.com", "canonical_domain"),
        ("Acme Inc", None, "canonical_name"),
        ("", None, "fallback_slug"),
        ("   ", "   ", "fallback_slug"),
        ("?!", None, "fallback_slug"),
    ],
)
def test_match_source_reflects_priority(name, website, expected_source):
    assert _identity(name, website).match_source == expected_source


def test_utc_now_returns_aware_datetime():
    assert utc_now().tzinfo is not None
