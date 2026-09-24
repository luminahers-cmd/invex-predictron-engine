"""Deterministic research taxonomy — Phase 8 Sprint 1.

Defines the canonical set of research topics the planner reasons about.
Each topic carries its metadata (importance, prediction impact, freshness
requirements, preferred source *categories*), optional dependencies, and
a stable identifier.

Source categories are metadata only — they describe where evidence for a
topic would be found, they never perform discovery or scraping.  Later
Phase 8 sprints (Source Discovery, Collection) consume these categories.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from predictron_engine.research.exceptions import UnknownTopicError
from predictron_engine.research.models import ResearchTopic

# Controlled vocabulary of source *categories*.  These are immutable
# metadata labels; no discovery or collection happens here.
SOURCE_CATEGORIES: tuple[str, ...] = (
    "company_website",
    "product_docs",
    "news_articles",
    "press_releases",
    "linkedin",
    "github",
    "job_postings",
    "funding_databases",
    "reviews_platforms",
    "app_stores",
    "analyst_reports",
    "market_reports",
    "social_media",
    "public_registry",
    "patent_database",
    "web_search",
)

SOURCE_CATEGORY_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "company_website": "The company's own website and product pages.",
        "product_docs": "Documentation, manuals, and product guides.",
        "news_articles": "Press coverage and editorial articles.",
        "press_releases": "Official press releases and announcements.",
        "linkedin": "LinkedIn profiles and company pages.",
        "github": "Open-source repositories and developer activity.",
        "job_postings": "Job listings and hiring signals.",
        "funding_databases": "Venture databases such as Crunchbase.",
        "reviews_platforms": "Customer review platforms.",
        "app_stores": "Mobile app store listings and ratings.",
        "analyst_reports": "Equity research and independent analyst notes.",
        "market_reports": "Third-party market sizing and industry reports.",
        "social_media": "Social media accounts and community activity.",
        "public_registry": "Government and public company registries.",
        "patent_database": "Patent and intellectual property records.",
        "web_search": "General web search across all categories.",
    }
)

# Each spec: (id, name, description, importance, prediction_impact,
# freshness_requirement_days, freshness_sensitivity, source_categories,
# dependencies).
TopicSpec = tuple[
    str,
    str,
    str,
    float,
    float,
    int,
    float,
    tuple[str, ...],
    tuple[str, ...],
]

_TOPIC_SPECS: tuple[TopicSpec, ...] = (
    (
        "founders",
        "Founders",
        "Assess the founders' background, track record, and incentive alignment.",
        0.90,
        0.90,
        365,
        0.50,
        ("company_website", "linkedin", "news_articles", "web_search"),
        (),
    ),
    (
        "team",
        "Team",
        "Assess team composition, experience, and hiring depth.",
        0.80,
        0.85,
        180,
        0.50,
        ("company_website", "linkedin", "github", "web_search"),
        ("founders",),
    ),
    (
        "product",
        "Product",
        "Assess what the company builds and how it solves the problem.",
        0.90,
        0.95,
        180,
        0.55,
        (
            "company_website",
            "product_docs",
            "reviews_platforms",
            "app_stores",
            "web_search",
        ),
        (),
    ),
    (
        "technology",
        "Technology",
        "Assess technical architecture, moats, and feasibility.",
        0.75,
        0.80,
        180,
        0.55,
        (
            "company_website",
            "product_docs",
            "github",
            "patent_database",
            "web_search",
        ),
        ("product",),
    ),
    (
        "market",
        "Market",
        "Assess market size, growth, and structure.",
        0.85,
        0.90,
        180,
        0.60,
        ("market_reports", "analyst_reports", "news_articles", "web_search"),
        ("product",),
    ),
    (
        "competition",
        "Competition",
        "Assess the competitive landscape and positioning.",
        0.80,
        0.85,
        180,
        0.60,
        ("market_reports", "news_articles", "analyst_reports", "web_search"),
        ("technology", "market"),
    ),
    (
        "customers",
        "Customers",
        "Assess who buys, usage patterns, and satisfaction.",
        0.80,
        0.85,
        90,
        0.65,
        (
            "reviews_platforms",
            "app_stores",
            "analyst_reports",
            "web_search",
        ),
        ("product",),
    ),
    (
        "pricing",
        "Pricing",
        "Assess pricing model, tiers, and willingness to pay.",
        0.70,
        0.80,
        90,
        0.65,
        ("company_website", "product_docs", "analyst_reports", "web_search"),
        ("market", "customers"),
    ),
    (
        "traction",
        "Traction",
        "Assess adoption, growth, revenue, and usage signals.",
        0.80,
        0.85,
        90,
        0.75,
        (
            "press_releases",
            "news_articles",
            "funding_databases",
            "social_media",
            "web_search",
        ),
        ("customers",),
    ),
    (
        "business_model",
        "Business Model",
        "Assess how the company makes money and unit economics.",
        0.75,
        0.80,
        180,
        0.50,
        ("company_website", "product_docs", "analyst_reports", "web_search"),
        ("product", "market"),
    ),
    (
        "funding",
        "Funding",
        "Assess capital raised, investors, and valuation history.",
        0.80,
        0.90,
        90,
        0.65,
        ("funding_databases", "press_releases", "news_articles", "web_search"),
        ("traction", "market"),
    ),
    (
        "hiring",
        "Hiring",
        "Assess headcount growth and hiring signals.",
        0.55,
        0.60,
        90,
        0.70,
        ("job_postings", "linkedin", "social_media", "web_search"),
        ("team", "funding"),
    ),
    (
        "partnerships",
        "Partnerships",
        "Assess distribution, channel, and strategic partnerships.",
        0.60,
        0.65,
        180,
        0.55,
        (
            "press_releases",
            "news_articles",
            "company_website",
            "web_search",
        ),
        ("business_model",),
    ),
    (
        "legal",
        "Legal",
        "Assess incorporation, IP, regulatory status, and litigation.",
        0.50,
        0.60,
        365,
        0.30,
        ("public_registry", "news_articles", "web_search"),
        (),
    ),
    (
        "reviews",
        "Reviews",
        "Assess qualitative sentiment from reviews and feedback.",
        0.45,
        0.50,
        90,
        0.70,
        (
            "reviews_platforms",
            "app_stores",
            "social_media",
            "web_search",
        ),
        ("product", "customers"),
    ),
    (
        "news",
        "News",
        "Capture recent announcements, coverage, and public events.",
        0.40,
        0.50,
        30,
        0.95,
        ("news_articles", "press_releases", "social_media", "web_search"),
        (),
    ),
    (
        "risks",
        "Risks",
        "Assess concentrated risks and threats to the thesis.",
        0.70,
        0.80,
        180,
        0.45,
        (
            "news_articles",
            "public_registry",
            "analyst_reports",
            "web_search",
        ),
        ("market", "legal", "competition"),
    ),
)


def _build_topic(spec: TopicSpec) -> ResearchTopic:
    """Build a ResearchTopic from a typed spec tuple."""
    (
        topic_id,
        name,
        description,
        importance,
        prediction_impact,
        freshness_requirement_days,
        freshness_sensitivity,
        source_categories,
        dependencies,
    ) = spec
    return ResearchTopic(
        topic_id=topic_id,
        name=name,
        description=description,
        importance=importance,
        prediction_impact=prediction_impact,
        freshness_requirement_days=freshness_requirement_days,
        freshness_sensitivity=freshness_sensitivity,
        source_categories=source_categories,
        dependencies=dependencies,
    )


RESEARCH_TOPICS: tuple[ResearchTopic, ...] = tuple(
    _build_topic(spec) for spec in _TOPIC_SPECS
)

TOPIC_REGISTRY: Mapping[str, ResearchTopic] = MappingProxyType(
    {topic.topic_id: topic for topic in RESEARCH_TOPICS}
)


def get_topic(topic_id: str) -> ResearchTopic:
    """Return the topic for ``topic_id`` or raise :class:`UnknownTopicError`."""
    topic = TOPIC_REGISTRY.get(topic_id)
    if topic is None:
        raise UnknownTopicError(f"unknown research topic: {topic_id!r}")
    return topic


def has_topic(topic_id: str) -> bool:
    """Return whether ``topic_id`` exists in the taxonomy."""
    return topic_id in TOPIC_REGISTRY


def all_topic_ids() -> tuple[str, ...]:
    """Return all topic identifiers in deterministic taxonomy order."""
    return tuple(TOPIC_REGISTRY)


def topic_index(topic_id: str) -> int:
    """Return the deterministic taxonomy position of ``topic_id``.

    The index is a stable tie-breaker for ordering rules: it has no
    semantic meaning beyond "position in the taxonomy".
    """
    for position, known in enumerate(RESEARCH_TOPICS):
        if known.topic_id == topic_id:
            return position
    raise UnknownTopicError(f"unknown research topic: {topic_id!r}")


def dependency_edges() -> tuple[tuple[str, str], ...]:
    """Return ``(dependency, dependent)`` edges, sorted for determinism.

    An edge ``(a, b)`` means topic ``a`` must be researched before
    topic ``b``.
    """
    edges: list[tuple[str, str]] = []
    for topic in RESEARCH_TOPICS:
        for dependency in topic.dependencies:
            edges.append((dependency, topic.topic_id))
    return tuple(sorted(edges))
