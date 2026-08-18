"""Tests for the official website identification and evidence prioritization layer.

Every test is pure-deterministic — identical inputs always produce identical
outputs.  No HTTP calls, no LLM calls, no side effects.
"""

from __future__ import annotations

import pytest

from predictron_engine.evidence.prioritization import (
    EvidenceType,
    OfficialWebsiteResult,
    PrioritizationSettings,
    PrioritizedPage,
    _classify_path,
    _is_homepage,
    _is_third_party,
    _name_in_domain,
    identify_official_website,
    prioritise_pages,
    score_evidence_quality,
    select_pages_for_fetch,
)
from predictron_engine.evidence.ranking import RankedUrl

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ranked(urls: list[str], score: float = 0.9) -> list[RankedUrl]:
    """Build a list of RankedUrl from plain URL strings."""
    return [RankedUrl(url=u, score=score) for u in urls]


HOMEPAGE_URL = "https://sampleco.com"
ABOUT_URL = "https://sampleco.com/about"
PRICING_URL = "https://sampleco.com/pricing"
DOCS_URL = "https://sampleco.com/docs"
BLOG_URL = "https://sampleco.com/blog"
CAREERS_URL = "https://sampleco.com/careers"
PRODUCT_URL = "https://sampleco.com/product"
NEWS_URL = "https://sampleco.com/news"
GITHUB_URL = "https://github.com/sampleco/sampleco"
LINKEDIN_URL = "https://linkedin.com/company/sampleco"
TECHCRUNCH_URL = "https://techcrunch.com/2024/01/sampleco-series-a"
MEDIUM_URL = "https://medium.com/@sampleco"
SUBDOMAIN_URL = "https://docs.sampleco.com"
DEEP_PATH_URL = "https://sampleco.com/some/very/deep/nested/page"


# ===================================================================
# _name_in_domain
# ===================================================================


class TestNameInDomain:
    def test_exact_match(self) -> None:
        assert _name_in_domain("sampleco", "sampleco.com") == 1.0

    def test_partial_match(self) -> None:
        score = _name_in_domain("sample cloud", "samplecloud.com")
        # "samplecloud" is one token, name splits to {"sample","cloud"}
        # no exact token overlap → 0.0
        assert score == pytest.approx(0.0)

    def test_multi_token_match(self) -> None:
        score = _name_in_domain("sample cloud", "sample-cloud.com")
        # host splits to {"sample","cloud"} → full overlap
        assert score == pytest.approx(1.0)

    def test_no_match(self) -> None:
        assert _name_in_domain("widgets", "sampleco.com") == 0.0

    def test_empty_name(self) -> None:
        assert _name_in_domain("", "sampleco.com") == 0.0

    def test_empty_host(self) -> None:
        assert _name_in_domain("sampleco", "") == 0.0

    def test_both_empty(self) -> None:
        assert _name_in_domain("", "") == 0.0

    def test_multi_token_name(self) -> None:
        score = _name_in_domain("ai robot", "ai-robot.com")
        assert score == pytest.approx(1.0)

    def test_tld_stripped(self) -> None:
        score = _name_in_domain("sampleco", "sampleco.io")
        assert score == pytest.approx(1.0)


# ===================================================================
# _is_third_party
# ===================================================================


class TestIsThirdParty:
    def test_github(self) -> None:
        assert _is_third_party("github.com") == EvidenceType.REPOSITORY

    def test_github_subdomain(self) -> None:
        # www.github.com ends with .github.com → third party
        assert _is_third_party("www.github.com") == EvidenceType.REPOSITORY

    def test_crunchbase(self) -> None:
        assert _is_third_party("crunchbase.com") == EvidenceType.DIRECTORY

    def test_linkedin(self) -> None:
        assert _is_third_party("linkedin.com") == EvidenceType.SOCIAL

    def test_medium(self) -> None:
        assert _is_third_party("medium.com") == EvidenceType.BLOG

    def test_techcrunch(self) -> None:
        assert _is_third_party("techcrunch.com") == EvidenceType.NEWS

    def test_unknown_domain(self) -> None:
        assert _is_third_party("sampleco.com") is None

    def test_empty_host(self) -> None:
        assert _is_third_party("") is None


# ===================================================================
# _classify_path
# ===================================================================


class TestClassifyPath:
    def test_about(self) -> None:
        etype, bonus, reason = _classify_path("https://sampleco.com/about")
        assert etype == EvidenceType.ABOUT
        assert bonus > 0

    def test_pricing(self) -> None:
        etype, bonus, reason = _classify_path("https://sampleco.com/pricing")
        assert etype == EvidenceType.PRICING
        assert bonus > 0

    def test_docs(self) -> None:
        etype, bonus, reason = _classify_path("https://sampleco.com/docs/intro")
        assert etype == EvidenceType.DOCUMENTATION
        assert bonus > 0

    def test_blog(self) -> None:
        etype, bonus, reason = _classify_path("https://sampleco.com/blog/post-1")
        assert etype == EvidenceType.BLOG
        assert bonus > 0

    def test_careers(self) -> None:
        etype, bonus, reason = _classify_path("https://sampleco.com/careers")
        assert etype == EvidenceType.CAREERS

    def test_homepage(self) -> None:
        etype, bonus, reason = _classify_path("https://sampleco.com/")
        assert etype == EvidenceType.UNKNOWN
        assert bonus == 0.0

    def test_deep_unknown(self) -> None:
        etype, bonus, reason = _classify_path(DEEP_PATH_URL)
        assert etype == EvidenceType.UNKNOWN
        assert bonus == 0.0

    def test_api_docs(self) -> None:
        etype, bonus, reason = _classify_path("https://sampleco.com/api/v1")
        assert etype == EvidenceType.DOCUMENTATION

    def test_plans_pricing(self) -> None:
        etype, bonus, reason = _classify_path("https://sampleco.com/plans")
        assert etype == EvidenceType.PRICING

    def test_features_product(self) -> None:
        etype, bonus, reason = _classify_path("https://sampleco.com/features")
        assert etype == EvidenceType.PRODUCT


# ===================================================================
# _is_homepage
# ===================================================================


class TestIsHomepage:
    def test_root_slash(self) -> None:
        assert _is_homepage("https://sampleco.com/") is True

    def test_no_slash(self) -> None:
        assert _is_homepage("https://sampleco.com") is True

    def test_subpath(self) -> None:
        assert _is_homepage("https://sampleco.com/about") is False

    def test_trailing_content(self) -> None:
        assert _is_homepage("https://sampleco.com/something") is False

    def test_empty_string(self) -> None:
        assert _is_homepage("") is True


# ===================================================================
# EvidenceType constants
# ===================================================================


class TestEvidenceType:
    def test_all_types_are_strings(self) -> None:
        types = [
            EvidenceType.OFFICIAL_HOMEPAGE,
            EvidenceType.ABOUT,
            EvidenceType.PRODUCT,
            EvidenceType.PRICING,
            EvidenceType.DOCUMENTATION,
            EvidenceType.BLOG,
            EvidenceType.CAREERS,
            EvidenceType.NEWS,
            EvidenceType.DIRECTORY,
            EvidenceType.SOCIAL,
            EvidenceType.REPOSITORY,
            EvidenceType.UNKNOWN,
        ]
        for t in types:
            assert isinstance(t, str)

    def test_unique_values(self) -> None:
        types = {
            EvidenceType.OFFICIAL_HOMEPAGE,
            EvidenceType.ABOUT,
            EvidenceType.PRODUCT,
            EvidenceType.PRICING,
            EvidenceType.DOCUMENTATION,
            EvidenceType.BLOG,
            EvidenceType.CAREERS,
            EvidenceType.NEWS,
            EvidenceType.DIRECTORY,
            EvidenceType.SOCIAL,
            EvidenceType.REPOSITORY,
            EvidenceType.UNKNOWN,
        }
        assert len(types) == 12


# ===================================================================
# PrioritizationSettings
# ===================================================================


class TestPrioritizationSettings:
    def test_defaults(self) -> None:
        s = PrioritizationSettings()
        assert s.min_official_confidence == 0.6
        assert s.max_fetch_pages == 10

    def test_custom(self) -> None:
        s = PrioritizationSettings(min_official_confidence=0.8, max_fetch_pages=5)
        assert s.min_official_confidence == 0.8
        assert s.max_fetch_pages == 5

    def test_invalid_confidence(self) -> None:
        with pytest.raises(Exception):
            PrioritizationSettings(min_official_confidence=1.5)

    def test_invalid_pages(self) -> None:
        with pytest.raises(Exception):
            PrioritizationSettings(max_fetch_pages=0)


# ===================================================================
# OfficialWebsiteResult
# ===================================================================


class TestOfficialWebsiteResult:
    def test_default(self) -> None:
        r = OfficialWebsiteResult(url=None, confidence=0.0, factors=())
        assert r.url is None
        assert r.confidence == 0.0
        assert r.factors == ()

    def test_frozen(self) -> None:
        r = OfficialWebsiteResult(url="https://example.com", confidence=0.9)
        with pytest.raises(Exception):
            r.url = "https://other.com"  # type: ignore[misc]


# ===================================================================
# PrioritizedPage
# ===================================================================


class TestPrioritizedPage:
    def test_ordering_by_priority(self) -> None:
        p1 = PrioritizedPage(
            url="a", quality_score=0.9, evidence_type="x", priority=2,
        )
        p2 = PrioritizedPage(
            url="b", quality_score=0.8, evidence_type="x", priority=1,
        )
        assert p2 < p1

    def test_ordering_by_quality_same_priority(self) -> None:
        p1 = PrioritizedPage(
            url="a", quality_score=0.9, evidence_type="x", priority=1,
        )
        p2 = PrioritizedPage(
            url="b", quality_score=0.7, evidence_type="x", priority=1,
        )
        assert p1 < p2

    def test_ordering_by_url_fallback(self) -> None:
        p1 = PrioritizedPage(
            url="a.com", quality_score=0.9, evidence_type="x", priority=1,
        )
        p2 = PrioritizedPage(
            url="b.com", quality_score=0.9, evidence_type="x", priority=1,
        )
        assert p1 < p2

    def test_sort_produces_correct_order(self) -> None:
        pages = [
            PrioritizedPage(
                url="c.com", quality_score=0.5, evidence_type="x", priority=3,
            ),
            PrioritizedPage(
                url="a.com", quality_score=0.9, evidence_type="x", priority=1,
            ),
            PrioritizedPage(
                url="b.com", quality_score=0.7, evidence_type="x", priority=2,
            ),
        ]
        pages.sort()
        assert [p.url for p in pages] == ["a.com", "b.com", "c.com"]


# ===================================================================
# identify_official_website
# ===================================================================


class TestIdentifyOfficialWebsite:
    def test_empty_ranked(self) -> None:
        result = identify_official_website([])
        assert result.url is None
        assert result.confidence == 0.0

    def test_exact_known_host_match(self) -> None:
        ranked = _ranked([HOMEPAGE_URL, GITHUB_URL])
        result = identify_official_website(
            ranked,
            startup_name="sampleco",
            known_website_host="sampleco.com",
        )
        assert result.url == HOMEPAGE_URL
        assert result.confidence == 1.0
        assert "exact match with known website" in result.factors

    def test_subdomain_of_known_host(self) -> None:
        ranked = _ranked([SUBDOMAIN_URL, GITHUB_URL])
        result = identify_official_website(
            ranked,
            startup_name="sampleco",
            known_website_host="sampleco.com",
        )
        assert result.url == SUBDOMAIN_URL
        assert result.confidence >= 0.85
        assert "subdomain of known website" in result.factors

    def test_name_similarity_without_known(self) -> None:
        ranked = _ranked([HOMEPAGE_URL, GITHUB_URL])
        result = identify_official_website(
            ranked,
            startup_name="sampleco",
            known_website_host=None,
        )
        assert result.url == HOMEPAGE_URL
        assert result.confidence > 0.0

    def test_third_party_penalty(self) -> None:
        ranked = _ranked([GITHUB_URL, HOMEPAGE_URL])
        result = identify_official_website(
            ranked,
            startup_name="sampleco",
            known_website_host="sampleco.com",
        )
        assert result.url == HOMEPAGE_URL

    def test_only_third_party_urls(self) -> None:
        ranked = _ranked([GITHUB_URL, LINKEDIN_URL])
        result = identify_official_website(
            ranked,
            startup_name="sampleco",
            known_website_host=None,
        )
        assert result.url is not None
        assert result.confidence < 0.5

    def test_position_bonus(self) -> None:
        ranked = _ranked([HOMEPAGE_URL])
        result = identify_official_website(
            ranked,
            startup_name="xyz",
            known_website_host=None,
        )
        assert any("rank position 1" in f for f in result.factors)

    def test_deterministic(self) -> None:
        ranked = _ranked([HOMEPAGE_URL, ABOUT_URL, GITHUB_URL])
        r1 = identify_official_website(ranked, startup_name="sampleco")
        r2 = identify_official_website(ranked, startup_name="sampleco")
        assert r1 == r2

    def test_confidence_capped_at_1(self) -> None:
        ranked = _ranked([HOMEPAGE_URL])
        result = identify_official_website(
            ranked,
            startup_name="sampleco",
            known_website_host="sampleco.com",
        )
        assert result.confidence <= 1.0

    def test_confidence_floored_at_0(self) -> None:
        ranked = _ranked([GITHUB_URL])
        result = identify_official_website(
            ranked,
            startup_name="sampleco",
            known_website_host=None,
        )
        assert result.confidence >= 0.0


# ===================================================================
# score_evidence_quality
# ===================================================================


class TestScoreEvidenceQuality:
    def test_default_score(self) -> None:
        quality, etype, reasons = score_evidence_quality(
            "https://example.com/page",
        )
        assert 0.0 <= quality <= 1.0
        assert etype == EvidenceType.UNKNOWN

    def test_official_homepage(self) -> None:
        quality, etype, reasons = score_evidence_quality(
            "https://sampleco.com/",
            official_host="sampleco.com",
        )
        assert etype == EvidenceType.OFFICIAL_HOMEPAGE
        assert quality > 0.7
        assert "official domain match" in reasons

    def test_about_page(self) -> None:
        quality, etype, reasons = score_evidence_quality(
            "https://sampleco.com/about",
        )
        assert etype == EvidenceType.ABOUT
        assert "about/company page" in reasons

    def test_pricing_page(self) -> None:
        quality, etype, reasons = score_evidence_quality(
            "https://sampleco.com/pricing",
        )
        assert etype == EvidenceType.PRICING

    def test_docs_page(self) -> None:
        quality, etype, reasons = score_evidence_quality(
            "https://sampleco.com/docs",
        )
        assert etype == EvidenceType.DOCUMENTATION

    def test_github_page(self) -> None:
        quality, etype, reasons = score_evidence_quality(
            "https://github.com/org/repo",
        )
        assert etype == EvidenceType.REPOSITORY
        assert "GitHub repository" in reasons

    def test_linkedin_page(self) -> None:
        quality, etype, reasons = score_evidence_quality(
            "https://linkedin.com/company/x",
        )
        assert etype == EvidenceType.SOCIAL

    def test_rank_position_bonus(self) -> None:
        q_top, _, _ = score_evidence_quality(
            "https://example.com/page", rank_position=0,
        )
        q_low, _, _ = score_evidence_quality(
            "https://example.com/page", rank_position=10,
        )
        assert q_top > q_low

    def test_brand_title_bonus(self) -> None:
        q_brand, _, reasons_brand = score_evidence_quality(
            "https://example.com",
            title="SampleCo - Official Homepage",
        )
        q_plain, _, reasons_plain = score_evidence_quality(
            "https://example.com",
            title="Some Random Page",
        )
        assert q_brand > q_plain
        assert "branded title signal" in reasons_brand
        assert "branded title signal" not in reasons_plain

    def test_official_subdomain(self) -> None:
        quality, _, reasons = score_evidence_quality(
            "https://docs.sampleco.com",
            official_host="sampleco.com",
        )
        assert "official subdomain" in reasons
        assert quality > 0.5

    def test_quality_capped(self) -> None:
        quality, _, _ = score_evidence_quality(
            "https://sampleco.com/",
            title="Official Homepage",
            official_host="sampleco.com",
            rank_position=0,
        )
        assert quality <= 1.0

    def test_deterministic(self) -> None:
        q1, e1, r1 = score_evidence_quality(
            "https://x.com/about", rank_position=2,
        )
        q2, e2, r2 = score_evidence_quality(
            "https://x.com/about", rank_position=2,
        )
        assert (q1, e1, r1) == (q2, e2, r2)


# ===================================================================
# prioritise_pages
# ===================================================================


class TestPrioritisePages:
    def test_empty_ranked(self) -> None:
        assert prioritise_pages([]) == []

    def test_returns_sorted_list(self) -> None:
        ranked = _ranked([DEEP_PATH_URL, ABOUT_URL, HOMEPAGE_URL])
        pages = prioritise_pages(ranked)
        assert len(pages) == 3
        for i in range(len(pages) - 1):
            assert pages[i] < pages[i + 1] or pages[i].url == pages[i + 1].url

    def test_homepage_gets_priority_1(self) -> None:
        ranked = _ranked([HOMEPAGE_URL])
        pages = prioritise_pages(ranked)
        assert pages[0].priority == 1
        assert pages[0].evidence_type == EvidenceType.OFFICIAL_HOMEPAGE

    def test_about_gets_priority_2(self) -> None:
        ranked = _ranked([ABOUT_URL])
        pages = prioritise_pages(ranked)
        assert pages[0].priority == 2
        assert pages[0].evidence_type == EvidenceType.ABOUT

    def test_pricing_gets_priority_3(self) -> None:
        ranked = _ranked([PRICING_URL])
        pages = prioritise_pages(ranked)
        assert pages[0].priority == 3

    def test_blog_gets_priority_4(self) -> None:
        ranked = _ranked([BLOG_URL])
        pages = prioritise_pages(ranked)
        assert pages[0].priority == 4

    def test_github_gets_priority_5(self) -> None:
        ranked = _ranked([GITHUB_URL])
        pages = prioritise_pages(ranked)
        assert pages[0].priority == 5
        assert pages[0].evidence_type == EvidenceType.REPOSITORY

    def test_unknown_gets_priority_6(self) -> None:
        ranked = _ranked([DEEP_PATH_URL])
        pages = prioritise_pages(ranked)
        assert pages[0].priority == 6

    def test_search_results_integration(self) -> None:
        ranked = _ranked([HOMEPAGE_URL])
        search = {HOMEPAGE_URL: ("SampleCo", "The best company")}
        pages = prioritise_pages(ranked, search_results=search)
        assert len(pages) == 1
        assert pages[0].quality_score > 0.0

    def test_official_host_bonus(self) -> None:
        ranked = _ranked([HOMEPAGE_URL])
        pages_no_host = prioritise_pages(ranked, official_host=None)
        pages_with_host = prioritise_pages(
            ranked, official_host="sampleco.com",
        )
        assert pages_with_host[0].quality_score > pages_no_host[0].quality_score

    def test_custom_settings(self) -> None:
        ranked = _ranked([HOMEPAGE_URL])
        settings = PrioritizationSettings(max_fetch_pages=1)
        pages = prioritise_pages(ranked, settings=settings)
        assert len(pages) == 1

    def test_deterministic(self) -> None:
        ranked = _ranked([HOMEPAGE_URL, ABOUT_URL, GITHUB_URL])
        p1 = prioritise_pages(ranked)
        p2 = prioritise_pages(ranked)
        assert p1 == p2

    def test_all_pages_have_reasons(self) -> None:
        ranked = _ranked([
            HOMEPAGE_URL, ABOUT_URL, GITHUB_URL, TECHCRUNCH_URL,
        ])
        pages = prioritise_pages(ranked)
        for page in pages:
            assert len(page.reasons) > 0

    def test_quality_scores_bounded(self) -> None:
        ranked = _ranked([
            HOMEPAGE_URL, ABOUT_URL, PRICING_URL, DOCS_URL, GITHUB_URL,
        ])
        pages = prioritise_pages(ranked)
        for page in pages:
            assert 0.0 <= page.quality_score <= 1.0


# ===================================================================
# select_pages_for_fetch
# ===================================================================


class TestSelectPagesForFetch:
    def test_default_limit(self) -> None:
        ranked = _ranked([
            f"https://sampleco.com/page{i}" for i in range(20)
        ])
        prioritised = prioritise_pages(ranked)
        selected = select_pages_for_fetch(prioritised)
        assert len(selected) <= 10

    def test_custom_limit(self) -> None:
        ranked = _ranked([
            f"https://sampleco.com/page{i}" for i in range(20)
        ])
        prioritised = prioritise_pages(ranked)
        settings = PrioritizationSettings(max_fetch_pages=3)
        selected = select_pages_for_fetch(prioritised, settings=settings)
        assert len(selected) == 3

    def test_empty_input(self) -> None:
        assert select_pages_for_fetch([]) == []

    def test_fewer_than_limit(self) -> None:
        ranked = _ranked([HOMEPAGE_URL, ABOUT_URL])
        prioritised = prioritise_pages(ranked)
        selected = select_pages_for_fetch(prioritised)
        assert len(selected) == 2

    def test_respects_priority_order(self) -> None:
        ranked = _ranked([
            DEEP_PATH_URL,
            GITHUB_URL,
            HOMEPAGE_URL,
            ABOUT_URL,
            LINKEDIN_URL,
        ])
        prioritised = prioritise_pages(ranked)
        settings = PrioritizationSettings(max_fetch_pages=2)
        selected = select_pages_for_fetch(prioritised, settings=settings)
        assert selected[0].priority <= selected[1].priority

    def test_preserves_page_objects(self) -> None:
        ranked = _ranked([HOMEPAGE_URL])
        prioritised = prioritise_pages(ranked)
        selected = select_pages_for_fetch(prioritised)
        assert selected[0].url == HOMEPAGE_URL
        assert selected[0].quality_score > 0.0

    def test_deterministic(self) -> None:
        ranked = _ranked([
            HOMEPAGE_URL, ABOUT_URL, GITHUB_URL, PRICING_URL,
        ])
        prioritised = prioritise_pages(ranked)
        s1 = select_pages_for_fetch(prioritised)
        s2 = select_pages_for_fetch(prioritised)
        assert s1 == s2
