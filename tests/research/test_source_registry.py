"""Tests for the deterministic SourceRegistry and discovery rules.

Covers catalog integrity, source lookup, category grouping, rule
validation, rule extensibility, and default-catalog determinism.
"""

from __future__ import annotations

import pytest

from predictron_engine.research import (
    SOURCE_CATALOG,
    SOURCE_REGISTRY,
    SourceCategory,
    SourceRegistry,
    all_sources,
    all_topic_ids,
    candidate_sources,
    discovery_rules,
    get_source,
    has_source,
    source_index,
    sources_by_category,
)
from predictron_engine.research.exceptions import (
    SourceRegistryError,
    UnknownSourceError,
)
from predictron_engine.research.models import ResearchSource
from predictron_engine.research.source_registry import SourceDiscoveryRule

ALL_TOPICS = set(all_topic_ids())

EXPECTED_SOURCE_IDS = {
    "official_website",
    "github",
    "crunchbase",
    "linkedin",
    "sec_edgar",
    "product_hunt",
    "wellfound",
    "yc",
    "company_blog",
    "engineering_blog",
    "documentation",
    "api_documentation",
    "developer_docs",
    "press_releases",
    "news",
    "patents",
    "research_papers",
    "g2",
    "capterra",
    "reddit",
    "hacker_news",
    "app_store",
    "google_play",
    "job_listings",
    "case_studies",
    "careers",
}

# Topic -> sources mandated by the sprint discovery-rule examples.
RULE_EXAMPLES = {
    "funding": {"crunchbase", "sec_edgar", "press_releases", "official_website"},
    "technology": {"github", "documentation", "engineering_blog", "patents", "developer_docs"},
    "customers": {"case_studies", "g2", "capterra", "official_website"},
    "hiring": {"careers", "linkedin", "wellfound"},
    "competition": {"official_website", "product_hunt", "news"},
}


def _source(identifier: str = "custom_source", **overrides: object) -> ResearchSource:
    base: dict[str, object] = {
        "identifier": identifier,
        "display_name": "Custom Source",
        "source_category": SourceCategory.GITHUB,
        "trust_score": 0.7,
        "freshness_score": 0.6,
        "coverage_score": 0.6,
        "relative_cost": 0.1,
        "supports_structured_data": True,
        "preferred_topics": ("technology",),
    }
    base.update(overrides)
    return ResearchSource(**base)  # type: ignore[arg-type]


def _rule(
    topic_id: str = "technology",
    identifiers: tuple[str, ...] = ("custom_source",),
) -> SourceDiscoveryRule:
    return SourceDiscoveryRule(
        topic_id=topic_id, source_identifiers=identifiers
    )


class TestCatalogIntegrity:
    """The default catalog is complete, unique, and well-formed."""

    def test_expected_sources_present(self) -> None:
        assert {s.identifier for s in SOURCE_CATALOG} == EXPECTED_SOURCE_IDS

    def test_identifiers_unique(self) -> None:
        ids = [s.identifier for s in SOURCE_CATALOG]
        assert len(ids) == len(set(ids)) == len(EXPECTED_SOURCE_IDS)

    def test_display_names_unique_and_readable(self) -> None:
        names = [s.display_name for s in SOURCE_CATALOG]
        assert all(name.strip() for name in names)
        assert len(names) == len(set(names))

    def test_categories_all_valid_enum_members(self) -> None:
        assert all(
            isinstance(s.source_category, SourceCategory)
            for s in SOURCE_CATALOG
        )

    def test_scores_within_unit_range(self) -> None:
        for source in SOURCE_CATALOG:
            for field in (
                "trust_score",
                "freshness_score",
                "coverage_score",
                "relative_cost",
            ):
                assert 0.0 <= getattr(source, field) <= 1.0

    def test_preferred_topics_reference_real_topics(self) -> None:
        for source in SOURCE_CATALOG:
            assert source.preferred_topics
            assert set(source.preferred_topics) <= ALL_TOPICS

    def test_default_registry_matches_catalog(self) -> None:
        assert SOURCE_REGISTRY.all_sources() == SOURCE_CATALOG
        assert all_sources() == SOURCE_CATALOG


class TestDefaultRules:
    """Every taxonomy topic maps to candidate sources."""

    def test_every_topic_has_a_rule(self) -> None:
        covered = {rule.topic_id for rule in discovery_rules()}
        assert covered == ALL_TOPICS

    def test_every_rule_lists_at_least_two_sources(self) -> None:
        for rule in discovery_rules():
            assert len(rule.source_identifiers) >= 2

    def test_rule_candidates_are_resolvable_sources(self) -> None:
        for rule in discovery_rules():
            for identifier in rule.source_identifiers:
                assert has_source(identifier)

    def test_rule_examples_are_candidates(self) -> None:
        for topic_id, expected in RULE_EXAMPLES.items():
            identifiers = {s.identifier for s in candidate_sources(topic_id)}
            assert expected <= identifiers

    def test_rule_order_is_stable(self) -> None:
        assert discovery_rules() == discovery_rules()
        assert candidate_sources("funding") == candidate_sources("funding")

    def test_unknown_topic_has_no_candidates(self) -> None:
        assert candidate_sources("martian_golf") == ()


class TestRegistryLookup:
    """Source lookup helpers behave deterministically."""

    def test_get_and_has_source(self) -> None:
        assert get_source("crunchbase").display_name == "Crunchbase"
        assert has_source("github")
        assert not has_source("bogus")

    def test_unknown_source_raises(self) -> None:
        for call in (get_source, source_index):
            with pytest.raises(UnknownSourceError):
                call("bogus")

    def test_source_index_unique_and_canonical(self) -> None:
        positions = [source_index(s.identifier) for s in SOURCE_CATALOG]
        assert positions == list(range(len(SOURCE_CATALOG)))

    def test_sources_by_category(self) -> None:
        for source in sources_by_category(SourceCategory.CRUNCHBASE):
            assert source.source_category is SourceCategory.CRUNCHBASE
        assert all(
            source in sources_by_category(source.source_category)
            for source in SOURCE_CATALOG
        )

    def test_sources_by_category_invalid(self) -> None:
        with pytest.raises(ValueError):
            SOURCE_REGISTRY.sources_by_category("crunchbase")  # type: ignore[arg-type]

    def test_partition_covers_catalog(self) -> None:
        grouped = [
            source
            for category in SOURCE_REGISTRY.categories()
            for source in SOURCE_REGISTRY.sources_by_category(category)
        ]
        assert len(grouped) == len(SOURCE_CATALOG)
        assert {s.identifier for s in grouped} == EXPECTED_SOURCE_IDS


class TestRegistryConstruction:
    """Custom registries validate deterministically."""

    def test_custom_registry(self) -> None:
        registry = SourceRegistry(
            sources=(_source(),),
            rules=(_rule(),),
        )
        assert registry.all_sources() == (_source(),)
        assert registry.candidate_sources("technology") == (_source(),)

    def test_rule_for_topic_and_has_rule(self) -> None:
        registry = SOURCE_REGISTRY
        assert registry.has_rule("funding")
        assert registry.rule_for_topic("funding") is not None
        assert not registry.has_rule("bogus")

    def test_duplicate_source_identifier_rejected(self) -> None:
        with pytest.raises(SourceRegistryError):
            SourceRegistry(sources=(_source(), _source()))

    def test_non_source_entry_rejected(self) -> None:
        with pytest.raises(SourceRegistryError):
            SourceRegistry(sources=("github",))  # type: ignore[list-item]

    def test_duplicate_topic_rules_rejected(self) -> None:
        with pytest.raises(SourceRegistryError):
            SourceRegistry(
                sources=(_source("a"), _source("b")),
                rules=(
                    _rule("technology", ("a",)),
                    _rule("technology", ("b",)),
                ),
            )

    def test_rule_referencing_unknown_source_rejected(self) -> None:
        with pytest.raises(UnknownSourceError):
            SourceRegistry(
                sources=(_source(),),
                rules=(_rule(identifiers=("missing_source",)),),
            )

    def test_non_rule_entry_rejected(self) -> None:
        with pytest.raises(SourceRegistryError):
            SourceRegistry(
                sources=(_source(),),
                rules=("not-a-rule",),  # type: ignore[list-item]
            )

    def test_custom_rule_and_source_extend_discovery(self) -> None:
        discovery_registry = SourceRegistry(
            sources=SOURCE_CATALOG + (_source("quantum_lens"),),
            rules=tuple(
                _rule("funding", ("quantum_lens",))
                if rule.topic_id == "funding"
                else rule
                for rule in discovery_rules()
            ),
        )
        assert len(discovery_registry) == len(SOURCE_CATALOG) + 1
        assert {
            s.identifier
            for s in discovery_registry.candidate_sources("funding")
        } == {"quantum_lens"}
        assert "news" in {
            s.identifier for s in discovery_registry.candidate_sources("news")
        }


class TestRuleModel:
    """SourceDiscoveryRule validation."""

    def test_requires_topic_and_sources(self) -> None:
        with pytest.raises(ValueError):
            SourceDiscoveryRule(topic_id="", source_identifiers=("github",))
        with pytest.raises(ValueError):
            SourceDiscoveryRule(topic_id="technology", source_identifiers=())

    def test_identifiers_deduped(self) -> None:
        rule = SourceDiscoveryRule(
            topic_id="technology", source_identifiers=("github", "github")
        )
        assert rule.source_identifiers == ("github",)
