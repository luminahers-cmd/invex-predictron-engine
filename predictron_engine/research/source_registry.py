"""Deterministic source catalog and discovery rules — Phase 8 Sprint 2.

The Source Discovery layer answers *where* research should be performed.
This module owns the metadata that makes that answer deterministic:

* :class:`SourceRegistry` — an immutable catalog of
  :class:`~predictron_engine.research.models.ResearchSource` entries plus
  the topic-to-source :class:`SourceDiscoveryRule` set.
* The default catalog and rules consumed by :class:`SourceDiscovery`.

This is **metadata only**: no scraping, no HTTP requests, no browser
automation, no authentication, and no data collection.  The registry is
pure — the same construction always yields the same catalog.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from predictron_engine.research.exceptions import (
    SourceRegistryError,
    UnknownSourceError,
)
from predictron_engine.research.models import (
    ResearchSource,
    SourceCategory,
)


@dataclass(frozen=True)
class SourceDiscoveryRule:
    """Maps one research topic to its candidate source identifiers.

    The rule declares *which* sources are candidates for a topic; the
    ranker decides the *order*.  Rules are metadata only.
    """

    topic_id: str
    source_identifiers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.topic_id.strip():
            raise ValueError("topic_id must not be empty")
        identifiers = tuple(dict.fromkeys(self.source_identifiers))
        if not identifiers:
            raise ValueError("source_identifiers must not be empty")
        object.__setattr__(self, "topic_id", self.topic_id.strip())
        object.__setattr__(self, "source_identifiers", identifiers)


# ----------------------------------------------------------------------
# Default catalog
# ----------------------------------------------------------------------

# Each spec: (identifier, display_name, source_category, trust_score,
# freshness_score, coverage_score, relative_cost, supports_structured_data,
# preferred_topics, description).
_SourceSpec = tuple[
    str,
    str,
    SourceCategory,
    float,
    float,
    float,
    float,
    bool,
    tuple[str, ...],
    str,
]

_CATALOG_SPECS: tuple[_SourceSpec, ...] = (
    (
        "official_website",
        "Official Website",
        SourceCategory.OFFICIAL_WEBSITE,
        0.55,
        0.60,
        0.70,
        0.10,
        False,
        (
            "founders",
            "team",
            "product",
            "technology",
            "pricing",
            "business_model",
            "news",
        ),
        "The company's own website and product pages.",
    ),
    (
        "company_blog",
        "Company Blog",
        SourceCategory.COMPANY_BLOG,
        0.50,
        0.70,
        0.60,
        0.00,
        False,
        ("product", "business_model", "partnerships", "news"),
        "Official company blog posts and announcements.",
    ),
    (
        "engineering_blog",
        "Engineering Blog",
        SourceCategory.ENGINEERING_BLOG,
        0.70,
        0.70,
        0.55,
        0.00,
        False,
        ("technology",),
        "Engineering team posts describing architecture and process.",
    ),
    (
        "documentation",
        "Documentation",
        SourceCategory.DOCUMENTATION,
        0.60,
        0.75,
        0.75,
        0.00,
        False,
        ("product", "technology", "pricing"),
        "Official product documentation and user guides.",
    ),
    (
        "api_documentation",
        "API Documentation",
        SourceCategory.API_DOCUMENTATION,
        0.75,
        0.75,
        0.65,
        0.00,
        True,
        ("technology", "product"),
        "Structured API references and interface specs.",
    ),
    (
        "developer_docs",
        "Developer Docs",
        SourceCategory.DEVELOPER_DOCS,
        0.70,
        0.70,
        0.65,
        0.00,
        True,
        ("technology",),
        "Deep developer-facing documentation and SDKs.",
    ),
    (
        "github",
        "GitHub",
        SourceCategory.GITHUB,
        0.75,
        0.70,
        0.65,
        0.00,
        True,
        ("technology", "team", "product"),
        "Open-source repositories and developer activity.",
    ),
    (
        "news",
        "News",
        SourceCategory.NEWS,
        0.50,
        0.90,
        0.65,
        0.10,
        False,
        ("news", "market", "funding", "risks", "competition"),
        "Press coverage and editorial articles.",
    ),
    (
        "press_releases",
        "Press Releases",
        SourceCategory.PRESS_RELEASES,
        0.70,
        0.85,
        0.55,
        0.00,
        False,
        (
            "funding",
            "traction",
            "partnerships",
            "news",
            "legal",
            "business_model",
        ),
        "Official press releases and public announcements.",
    ),
    (
        "case_studies",
        "Case Studies",
        SourceCategory.CASE_STUDIES,
        0.75,
        0.50,
        0.65,
        0.00,
        False,
        ("customers", "product"),
        "Published customer success case studies.",
    ),
    (
        "g2",
        "G2",
        SourceCategory.G2,
        0.65,
        0.75,
        0.75,
        0.20,
        True,
        ("customers", "product", "reviews", "pricing", "competition"),
        "B2B software review platform with structured ratings.",
    ),
    (
        "capterra",
        "Capterra",
        SourceCategory.CAPTERRA,
        0.65,
        0.75,
        0.75,
        0.20,
        True,
        ("customers", "product", "reviews", "pricing", "competition"),
        "Software directory with structured customer reviews.",
    ),
    (
        "app_store",
        "App Store",
        SourceCategory.APP_STORE,
        0.70,
        0.80,
        0.70,
        0.00,
        True,
        ("product", "customers", "traction", "reviews"),
        "iOS app store listing, ratings, and reviews.",
    ),
    (
        "google_play",
        "Google Play",
        SourceCategory.GOOGLE_PLAY,
        0.70,
        0.80,
        0.70,
        0.00,
        True,
        ("product", "customers", "traction", "reviews"),
        "Android app listing, ratings, and reviews.",
    ),
    (
        "reddit",
        "Reddit",
        SourceCategory.REDDIT,
        0.35,
        0.80,
        0.60,
        0.00,
        False,
        ("customers", "reviews", "product", "news", "risks"),
        "Community discussions and anecdotal sentiment.",
    ),
    (
        "hacker_news",
        "Hacker News",
        SourceCategory.HACKER_NEWS,
        0.45,
        0.85,
        0.55,
        0.00,
        True,
        ("technology", "traction", "product", "news", "reviews"),
        "Startup and tech community discussions.",
    ),
    (
        "linkedin",
        "LinkedIn",
        SourceCategory.LINKEDIN,
        0.65,
        0.60,
        0.65,
        0.20,
        True,
        ("founders", "team", "hiring"),
        "Professional profiles and company pages.",
    ),
    (
        "wellfound",
        "Wellfound",
        SourceCategory.WELLFOUND,
        0.60,
        0.65,
        0.55,
        0.10,
        True,
        ("hiring", "founders", "team"),
        "Startup jobs and founder/team profiles.",
    ),
    (
        "job_listings",
        "Job Listings",
        SourceCategory.JOB_LISTINGS,
        0.60,
        0.85,
        0.55,
        0.00,
        True,
        ("hiring", "team"),
        "Aggregated company job postings across boards.",
    ),
    (
        "careers",
        "Careers",
        SourceCategory.CAREERS,
        0.55,
        0.80,
        0.55,
        0.00,
        False,
        ("hiring", "team"),
        "The company's own careers and hiring pages.",
    ),
    (
        "crunchbase",
        "Crunchbase",
        SourceCategory.CRUNCHBASE,
        0.85,
        0.70,
        0.80,
        0.30,
        True,
        ("funding", "founders", "market", "traction"),
        "Structured venture, funding, and company database.",
    ),
    (
        "sec_edgar",
        "SEC EDGAR",
        SourceCategory.SEC_EDGAR,
        0.95,
        0.60,
        0.85,
        0.00,
        True,
        ("funding", "legal", "business_model"),
        "Official SEC filings and financial disclosures.",
    ),
    (
        "yc",
        "YC",
        SourceCategory.YC,
        0.85,
        0.55,
        0.60,
        0.00,
        True,
        ("founders", "funding", "market"),
        "Y Combinator profiles and public batch data.",
    ),
    (
        "product_hunt",
        "Product Hunt",
        SourceCategory.PRODUCT_HUNT,
        0.55,
        0.80,
        0.55,
        0.00,
        True,
        ("product", "competition", "traction"),
        "Product launches and community reception.",
    ),
    (
        "patents",
        "Patents",
        SourceCategory.PATENTS,
        0.90,
        0.50,
        0.70,
        0.10,
        True,
        ("technology", "legal"),
        "Patent and intellectual property records.",
    ),
    (
        "research_papers",
        "Research Papers",
        SourceCategory.RESEARCH_PAPERS,
        0.75,
        0.40,
        0.65,
        0.10,
        True,
        ("market", "technology", "competition"),
        "Academic and analyst research literature.",
    ),
)


def _build_source(spec: _SourceSpec) -> ResearchSource:
    """Build a ResearchSource from a typed spec tuple."""
    (
        identifier,
        display_name,
        source_category,
        trust_score,
        freshness_score,
        coverage_score,
        relative_cost,
        supports_structured_data,
        preferred_topics,
        description,
    ) = spec
    return ResearchSource(
        identifier=identifier,
        display_name=display_name,
        source_category=source_category,
        trust_score=trust_score,
        freshness_score=freshness_score,
        coverage_score=coverage_score,
        relative_cost=relative_cost,
        supports_structured_data=supports_structured_data,
        preferred_topics=preferred_topics,
        description=description,
    )


SOURCE_CATALOG: tuple[ResearchSource, ...] = tuple(
    _build_source(spec) for spec in _CATALOG_SPECS
)


# ----------------------------------------------------------------------
# Default discovery rules
# ----------------------------------------------------------------------

# topic_id -> candidate source identifiers, in stable curatorial order.
_DEFAULT_RULE_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("founders", ("linkedin", "official_website", "crunchbase", "yc", "wellfound", "news")),
    ("team", ("linkedin", "official_website", "github", "careers", "wellfound", "job_listings")),
    (
        "product",
        (
            "official_website",
            "documentation",
            "g2",
            "capterra",
            "product_hunt",
            "app_store",
            "google_play",
            "case_studies",
        ),
    ),
    (
        "technology",
        (
            "github",
            "documentation",
            "engineering_blog",
            "api_documentation",
            "patents",
            "developer_docs",
        ),
    ),
    ("market", ("research_papers", "news", "press_releases", "crunchbase", "yc")),
    (
        "competition",
        ("official_website", "product_hunt", "news", "g2", "capterra", "research_papers"),
    ),
    (
        "customers",
        (
            "case_studies",
            "g2",
            "capterra",
            "official_website",
            "app_store",
            "google_play",
            "reddit",
        ),
    ),
    ("pricing", ("official_website", "documentation", "g2", "capterra", "reddit", "product_hunt")),
    (
        "traction",
        (
            "press_releases",
            "news",
            "crunchbase",
            "app_store",
            "google_play",
            "product_hunt",
            "hacker_news",
        ),
    ),
    (
        "business_model",
        ("official_website", "news", "press_releases", "company_blog", "sec_edgar"),
    ),
    (
        "funding",
        ("crunchbase", "sec_edgar", "press_releases", "official_website", "news", "yc"),
    ),
    ("hiring", ("careers", "linkedin", "wellfound", "job_listings")),
    ("partnerships", ("press_releases", "company_blog", "official_website", "news")),
    ("legal", ("sec_edgar", "news", "patents", "press_releases")),
    ("reviews", ("g2", "capterra", "app_store", "google_play", "reddit", "hacker_news")),
    ("news", ("news", "press_releases", "company_blog", "reddit", "hacker_news")),
    ("risks", ("news", "sec_edgar", "patents", "press_releases", "reddit", "hacker_news")),
)

DEFAULT_DISCOVERY_RULES: tuple[SourceDiscoveryRule, ...] = tuple(
    SourceDiscoveryRule(topic_id=topic_id, source_identifiers=identifiers)
    for topic_id, identifiers in _DEFAULT_RULE_SPECS
)


class SourceRegistry:
    """Immutable catalog of research sources and discovery rules.

    Parameters
    ----------
    sources:
        Optional catalog overriding the built-in one.  Every source must
        have a unique ``identifier``.
    rules:
        Optional discovery rules overriding the built-in set.  Every rule
        must reference identifiers present in ``sources`` and no topic may
        be claimed by more than one rule.
    """

    def __init__(
        self,
        sources: Sequence[ResearchSource] | None = None,
        *,
        rules: Sequence[SourceDiscoveryRule] | None = None,
    ) -> None:
        catalog = (
            tuple(sources) if sources is not None else SOURCE_CATALOG
        )
        self._catalog = catalog
        seen: set[str] = set()
        for source in catalog:
            if not isinstance(source, ResearchSource):
                raise SourceRegistryError(
                    "catalog entries must be ResearchSource instances"
                )
            if source.identifier in seen:
                raise SourceRegistryError(
                    f"duplicate source identifier: {source.identifier!r}"
                )
            seen.add(source.identifier)
        self._sources: Mapping[str, ResearchSource] = MappingProxyType(
            {source.identifier: source for source in catalog}
        )
        self._index: Mapping[str, int] = MappingProxyType(
            {source.identifier: position for position, source in enumerate(catalog)}
        )
        by_category: dict[SourceCategory, list[ResearchSource]] = {}
        for source in catalog:
            by_category.setdefault(source.source_category, []).append(source)
        self._by_category: Mapping[SourceCategory, tuple[ResearchSource, ...]] = (
            MappingProxyType(
                {category: tuple(entries) for category, entries in by_category.items()}
            )
        )
        rule_list = (
            tuple(rules) if rules is not None else DEFAULT_DISCOVERY_RULES
        )
        self._rule_list = rule_list
        rules_by_topic: dict[str, SourceDiscoveryRule] = {}
        for rule in rule_list:
            if not isinstance(rule, SourceDiscoveryRule):
                raise SourceRegistryError(
                    "rules must be SourceDiscoveryRule instances"
                )
            for identifier in rule.source_identifiers:
                if identifier not in self._sources:
                    raise UnknownSourceError(
                        f"rule for topic {rule.topic_id!r} references "
                        f"unknown source: {identifier!r}"
                    )
            if rule.topic_id in rules_by_topic:
                raise SourceRegistryError(
                    f"duplicate discovery rule for topic: {rule.topic_id!r}"
                )
            rules_by_topic[rule.topic_id] = rule
        self._rules_by_topic: Mapping[str, SourceDiscoveryRule] = (
            MappingProxyType(rules_by_topic)
        )

    def __len__(self) -> int:
        """Return the number of catalogued sources."""
        return len(self._catalog)

    def all_sources(self) -> tuple[ResearchSource, ...]:
        """Return the catalog in canonical order."""
        return self._catalog

    def get_source(self, identifier: str) -> ResearchSource:
        """Return the source for ``identifier`` or raise :class:`UnknownSourceError`."""
        source = self._sources.get(identifier)
        if source is None:
            raise UnknownSourceError(f"unknown research source: {identifier!r}")
        return source

    def has_source(self, identifier: str) -> bool:
        """Return whether ``identifier`` exists in the catalog."""
        return identifier in self._sources

    def source_index(self, identifier: str) -> int:
        """Return the canonical catalog position of ``identifier``.

        The index is a stable tie-breaker for ranking; it carries no
        semantic meaning beyond catalog position.
        """
        index = self._index.get(identifier)
        if index is None:
            raise UnknownSourceError(f"unknown research source: {identifier!r}")
        return index

    def sources_by_category(
        self, category: SourceCategory
    ) -> tuple[ResearchSource, ...]:
        """Return all sources in ``category``, in canonical catalog order."""
        if not isinstance(category, SourceCategory):
            raise ValueError("category must be a SourceCategory")
        return self._by_category.get(category, ())

    def categories(self) -> tuple[SourceCategory, ...]:
        """Return every category present, in canonical first-appearance order."""
        seen: list[SourceCategory] = []
        for source in self._catalog:
            if source.source_category not in seen:
                seen.append(source.source_category)
        return tuple(seen)

    def discovery_rules(self) -> tuple[SourceDiscoveryRule, ...]:
        """Return the discovery rules in registration order."""
        return self._rule_list

    def rule_for_topic(self, topic_id: str) -> SourceDiscoveryRule | None:
        """Return the rule for ``topic_id`` or ``None`` when absent."""
        return self._rules_by_topic.get(topic_id)

    def has_rule(self, topic_id: str) -> bool:
        """Return whether a discovery rule exists for ``topic_id``."""
        return topic_id in self._rules_by_topic

    def topics_covered(self) -> tuple[str, ...]:
        """Return the topics with rules, in rule order."""
        return tuple(rule.topic_id for rule in self._rule_list)

    def candidate_sources(self, topic_id: str) -> tuple[ResearchSource, ...]:
        """Return candidate sources for ``topic_id`` in rule order.

        When no rule exists for the topic the result is empty; the
        discovery engine applies its deterministic fallback set then.
        """
        rule = self._rules_by_topic.get(topic_id)
        if rule is None:
            return ()
        resolved: list[ResearchSource] = []
        for identifier in rule.source_identifiers:
            resolved.append(self._sources[identifier])
        return tuple(resolved)


# Default registry shared by the discovery engine.
SOURCE_REGISTRY = SourceRegistry()


def get_source(identifier: str) -> ResearchSource:
    """Return the default-catalog source for ``identifier``."""
    return SOURCE_REGISTRY.get_source(identifier)


def has_source(identifier: str) -> bool:
    """Return whether ``identifier`` exists in the default catalog."""
    return SOURCE_REGISTRY.has_source(identifier)


def all_sources() -> tuple[ResearchSource, ...]:
    """Return the default catalog in canonical order."""
    return SOURCE_REGISTRY.all_sources()


def sources_by_category(
    category: SourceCategory,
) -> tuple[ResearchSource, ...]:
    """Return default-catalog sources in ``category``."""
    return SOURCE_REGISTRY.sources_by_category(category)


def source_index(identifier: str) -> int:
    """Return the canonical position of ``identifier`` in the default catalog."""
    return SOURCE_REGISTRY.source_index(identifier)


def categories() -> tuple[SourceCategory, ...]:
    """Return every category present in the default catalog."""
    return SOURCE_REGISTRY.categories()


def candidate_sources(topic_id: str) -> tuple[ResearchSource, ...]:
    """Return default-rule candidate sources for ``topic_id``."""
    return SOURCE_REGISTRY.candidate_sources(topic_id)


def rule_for_topic(topic_id: str) -> SourceDiscoveryRule | None:
    """Return the default rule for ``topic_id`` or ``None``."""
    return SOURCE_REGISTRY.rule_for_topic(topic_id)


def discovery_rules() -> tuple[SourceDiscoveryRule, ...]:
    """Return the default discovery rules in registration order."""
    return SOURCE_REGISTRY.discovery_rules()
