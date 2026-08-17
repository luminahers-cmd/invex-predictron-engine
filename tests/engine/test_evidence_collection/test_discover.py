"""Tests for deterministic page discovery."""

from __future__ import annotations

from pydantic import HttpUrl

from predictron_engine.evidence.discover import DefaultPageDiscoverer
from predictron_engine.evidence.models import PageType


class TestDefaultPageDiscoverer:
    def setup_method(self) -> None:
        self.discoverer = DefaultPageDiscoverer()
        self.website = HttpUrl("https://example.com")

    def test_homepage_first(self) -> None:
        candidates = self.discoverer.discover(self.website)
        assert candidates[0].page_type == PageType.HOMEPAGE
        assert str(candidates[0].url) == "https://example.com/"

    def test_contains_all_default_paths(self) -> None:
        candidates = self.discoverer.discover(self.website)
        paths = {candidate.path for candidate in candidates}
        assert paths == {
            "", "about", "about-us", "company", "products", "services", "platform", "technology"
        }

    def test_page_types_mapped(self) -> None:
        by_path = {
            candidate.path: candidate.page_type
            for candidate in self.discoverer.discover(self.website)
        }
        assert by_path["about"] == PageType.ABOUT
        assert by_path["about-us"] == PageType.ABOUT
        assert by_path["company"] == PageType.COMPANY
        assert by_path["products"] == PageType.PRODUCTS
        assert by_path["services"] == PageType.SERVICES
        assert by_path["platform"] == PageType.PLATFORM
        assert by_path["technology"] == PageType.TECHNOLOGY

    def test_urls_are_absolute_and_under_site(self) -> None:
        for candidate in self.discoverer.discover(self.website):
            assert str(candidate.url).startswith("https://example.com")

    def test_deduplicates_identical_urls(self) -> None:
        custom = DefaultPageDiscoverer(
            paths=[("", PageType.HOMEPAGE), ("", PageType.UNKNOWN), ("about", PageType.ABOUT)]
        )
        candidates = custom.discover(self.website)
        assert len(candidates) == 2
        assert candidates[0].page_type == PageType.HOMEPAGE

    def test_custom_paths(self) -> None:
        custom = DefaultPageDiscoverer(paths=[("careers", PageType.UNKNOWN)])
        candidates = custom.discover(self.website)
        assert len(candidates) == 1
        assert candidates[0].path == "careers"
        assert candidates[0].page_type == PageType.UNKNOWN
