"""Deterministic placeholder collectors — Phase 8 Sprint 3.

This module implements :class:`PlaceholderCollector`, a shared base that
turns fixed class-level declarations into deterministic
:class:`~predictron_engine.research.models.Evidence` objects, and the 15
default collectors every Research Plan can dispatch to:

* :class:`FounderCollector`   — founders
* :class:`ProductCollector`   — product
* :class:`TechnologyCollector` — technology
* :class:`MarketCollector`    — market
* :class:`CompetitionCollector` — competition
* :class:`FundingCollector`   — funding
* :class:`PricingCollector`   — pricing
* :class:`TractionCollector`  — traction
* :class:`CustomerCollector`  — customers
* :class:`NewsCollector`      — news
* :class:`LegalCollector`     — legal
* :class:`RiskCollector`      — risks
* :class:`ReviewsCollector`   — reviews
* :class:`PartnershipCollector` — partnerships
* :class:`HiringCollector`    — hiring

No collector performs I/O: there is no HTTP, no scraping, no browser
automation, no search, no LLM, and no API.  Each collector returns only
deterministic placeholder evidence objects — the same task always
produces the same evidence.
"""

from __future__ import annotations

from typing import ClassVar, NamedTuple

from predictron_engine.research.collector import (
    EvidenceCollector,
    enforce_supported,
)
from predictron_engine.research.exceptions import InvalidCollectionInputError
from predictron_engine.research.models import (
    Evidence,
    EvidenceReference,
    ResearchTask,
)

# Clearly-synthetic origin for placeholder evidence references.  No
# collector ever resolves or contacts this host.
PLACEHOLDER_EVIDENCE_ORIGIN = "https://placeholder.predictron.local"


class ClaimSpec(NamedTuple):
    """One deterministic placeholder claim a collector can emit.

    ``claim`` is a template with ``{company}`` and optionally ``{topic}``
    interpolation slots.  ``confidence`` is a fixed constant in
    ``[0, 1]`` describing the placeholder's assumed reliability.
    """

    category: str
    claim: str
    confidence: float


class PlaceholderCollector(EvidenceCollector):
    """Base for deterministic, stateless placeholder collectors.

    Concrete collectors only declare fixed class attributes:

    ``collector_id``
        Stable snake_case identifier.
    ``display_name``
        Human-readable name.
    ``description``
        What evidence the collector produces.
    ``supported_topics``
        The topic identifiers this collector can handle.
    ``source_identifier``
        The recommended source this collector stands in for.
    ``source_category``
        The source category of that source (metadata only).
    ``claim_specs``
        The deterministic placeholder claims to emit per task.
    """

    source_identifier: ClassVar[str]
    source_category: ClassVar[str]
    claim_specs: ClassVar[tuple[ClaimSpec, ...]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Fail loudly when a concrete collector is misconfigured."""
        super().__init_subclass__(**kwargs)
        required = (
            "collector_id",
            "display_name",
            "supported_topics",
            "source_identifier",
            "source_category",
            "claim_specs",
        )
        for attribute in required:
            if not getattr(cls, attribute, ()):
                raise TypeError(
                    f"{cls.__name__} must declare a non-empty "
                    f"'{attribute}'"
                )
        for spec in cls.claim_specs:
            if not isinstance(spec, ClaimSpec):
                raise TypeError(
                    f"{cls.__name__}.claim_specs must be ClaimSpec entries"
                )

    def collect(
        self,
        task: ResearchTask,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple[Evidence, ...]:
        """Produce deterministic placeholder evidence for ``task``.

        Raises
        ------
        InvalidCollectionInputError:
            When ``task`` is not a :class:`ResearchTask` or
            ``company_name`` is blank.
        UnsupportedTopic:
            When ``task.topic_id`` is not supported by this collector.
        """
        if not isinstance(task, ResearchTask):
            raise InvalidCollectionInputError(
                "collect expects a ResearchTask instance"
            )
        enforce_supported(self, task.topic_id)
        company = company_name.strip()
        if not company:
            raise InvalidCollectionInputError(
                "company_name must not be empty"
            )
        cls = type(self)
        return tuple(
            _build_evidence(self, task, company, spec)
            for spec in cls.claim_specs
        )


def _build_evidence(
    collector: PlaceholderCollector,
    task: ResearchTask,
    company: str,
    spec: ClaimSpec,
) -> Evidence:
    """Build one deterministic evidence item from a claim spec."""
    cls = type(collector)
    claim = spec.claim.format(company=company, topic=task.topic_id)
    reference = EvidenceReference(
        source_category=cls.source_category,
        source_identifier=cls.source_identifier,
        url=(
            f"{PLACEHOLDER_EVIDENCE_ORIGIN}/{task.topic_id}/"
            f"{cls.source_identifier}"
        ),
        description=f"Placeholder reference for {task.title}.",
    )
    return Evidence(
        task_id=task.task_id,
        topic_id=task.topic_id,
        collector_id=cls.collector_id,
        category=spec.category,
        claim=claim,
        confidence=spec.confidence,
        reference=reference,
    )


class FounderCollector(PlaceholderCollector):
    """Placeholder evidence about founders and incentive alignment."""

    collector_id = "founder_collector"
    display_name = "Founders"
    description = "Evidence about founder background, track record, and incentives."
    supported_topics = ("founders",)
    source_identifier = "linkedin"
    source_category = "linkedin"
    claim_specs = (
        ClaimSpec(
            "founder_identity",
            "Identified the founding team members for {company}.",
            0.90,
        ),
        ClaimSpec(
            "founder_background",
            "Recorded founder background and prior track record for {company}.",
            0.85,
        ),
        ClaimSpec(
            "founder_incentives",
            "Recorded founder incentive-alignment signals for {company}.",
            0.80,
        ),
    )


class ProductCollector(PlaceholderCollector):
    """Placeholder evidence about the product and its market fit."""

    collector_id = "product_collector"
    display_name = "Product"
    description = "Evidence about what the company builds and how it solves the problem."
    supported_topics = ("product",)
    source_identifier = "official_website"
    source_category = "official_website"
    claim_specs = (
        ClaimSpec(
            "product_offerings",
            "Recorded the products and offerings of {company}.",
            0.90,
        ),
        ClaimSpec(
            "product_positioning",
            "Recorded how {company} positions its product in the market.",
            0.85,
        ),
        ClaimSpec(
            "product_roadmap",
            "Recorded public product roadmap signals for {company}.",
            0.75,
        ),
    )


class TechnologyCollector(PlaceholderCollector):
    """Placeholder evidence about the technical stack and moat."""

    collector_id = "technology_collector"
    display_name = "Technology"
    description = "Evidence about architecture, technical moats, and feasibility."
    supported_topics = ("technology",)
    source_identifier = "github"
    source_category = "github"
    claim_specs = (
        ClaimSpec(
            "technology_stack",
            "Recorded the technology stack used by {company}.",
            0.90,
        ),
        ClaimSpec(
            "architecture_moat",
            "Recorded architecture and technical moat signals for {company}.",
            0.80,
        ),
        ClaimSpec(
            "engineering_process",
            "Recorded engineering process and delivery signals for {company}.",
            0.75,
        ),
    )


class MarketCollector(PlaceholderCollector):
    """Placeholder evidence about the market the company operates in."""

    collector_id = "market_collector"
    display_name = "Market"
    description = "Evidence about market size, growth, and structure."
    supported_topics = ("market",)
    source_identifier = "research_papers"
    source_category = "research_papers"
    claim_specs = (
        ClaimSpec(
            "market_size",
            "Recorded the addressable market size for {company}.",
            0.85,
        ),
        ClaimSpec(
            "market_growth",
            "Recorded market growth and tailwind signals for {company}.",
            0.85,
        ),
        ClaimSpec(
            "market_structure",
            "Recorded market structure and segment signals for {company}.",
            0.80,
        ),
    )


class CompetitionCollector(PlaceholderCollector):
    """Placeholder evidence about the competitive landscape."""

    collector_id = "competition_collector"
    display_name = "Competition"
    description = "Evidence about competitive landscape and positioning."
    supported_topics = ("competition",)
    source_identifier = "news"
    source_category = "news"
    claim_specs = (
        ClaimSpec(
            "competitor_map",
            "Mapped the competitive landscape for {company}.",
            0.85,
        ),
        ClaimSpec(
            "competitive_positioning",
            "Recorded how {company} differentiates from competitors.",
            0.80,
        ),
        ClaimSpec(
            "moat_analysis",
            "Recorded competitive moat signals for {company}.",
            0.75,
        ),
    )


class FundingCollector(PlaceholderCollector):
    """Placeholder evidence about capital raised and investors."""

    collector_id = "funding_collector"
    display_name = "Funding"
    description = "Evidence about capital raised, investors, and valuation history."
    supported_topics = ("funding",)
    source_identifier = "crunchbase"
    source_category = "crunchbase"
    claim_specs = (
        ClaimSpec(
            "funding_rounds",
            "Recorded the funding rounds of {company}.",
            0.90,
        ),
        ClaimSpec(
            "investor_network",
            "Recorded the investor network backing {company}.",
            0.85,
        ),
        ClaimSpec(
            "valuation_history",
            "Recorded valuation and cap-table signals for {company}.",
            0.75,
        ),
    )


class PricingCollector(PlaceholderCollector):
    """Placeholder evidence about pricing model and tiers."""

    collector_id = "pricing_collector"
    display_name = "Pricing"
    description = "Evidence about pricing model, tiers, and willingness to pay."
    supported_topics = ("pricing",)
    source_identifier = "g2"
    source_category = "g2"
    claim_specs = (
        ClaimSpec(
            "pricing_tiers",
            "Recorded the pricing tiers and model of {company}.",
            0.90,
        ),
        ClaimSpec(
            "use_case_pricing",
            "Recorded how {company} prices across use cases.",
            0.80,
        ),
        ClaimSpec(
            "willingness_to_pay",
            "Recorded willingness-to-pay signals for {company}.",
            0.70,
        ),
    )


class TractionCollector(PlaceholderCollector):
    """Placeholder evidence about adoption, growth, and revenue."""

    collector_id = "traction_collector"
    display_name = "Traction"
    description = "Evidence about adoption, growth, revenue, and usage signals."
    supported_topics = ("traction",)
    source_identifier = "press_releases"
    source_category = "press_releases"
    claim_specs = (
        ClaimSpec(
            "adoption_signals",
            "Recorded adoption and usage signals for {company}.",
            0.85,
        ),
        ClaimSpec(
            "revenue_signals",
            "Recorded revenue and growth signals for {company}.",
            0.80,
        ),
        ClaimSpec(
            "growth_metrics",
            "Recorded public growth metrics for {company}.",
            0.75,
        ),
    )


class CustomerCollector(PlaceholderCollector):
    """Placeholder evidence about customers, usage, and satisfaction."""

    collector_id = "customer_collector"
    display_name = "Customers"
    description = "Evidence about who buys, usage patterns, and satisfaction."
    supported_topics = ("customers",)
    source_identifier = "case_studies"
    source_category = "case_studies"
    claim_specs = (
        ClaimSpec(
            "customer_segments",
            "Recorded the customer segments of {company}.",
            0.85,
        ),
        ClaimSpec(
            "satisfaction_signals",
            "Recorded customer satisfaction signals for {company}.",
            0.80,
        ),
        ClaimSpec(
            "usage_patterns",
            "Recorded usage and adoption patterns for {company}.",
            0.75,
        ),
    )


class NewsCollector(PlaceholderCollector):
    """Placeholder evidence about announcements and coverage."""

    collector_id = "news_collector"
    display_name = "News"
    description = "Evidence about recent announcements, coverage, and public events."
    supported_topics = ("news",)
    source_identifier = "news"
    source_category = "news"
    claim_specs = (
        ClaimSpec(
            "recent_coverage",
            "Recorded recent press coverage of {company}.",
            0.90,
        ),
        ClaimSpec(
            "product_announcements",
            "Recorded product announcements from {company}.",
            0.85,
        ),
        ClaimSpec(
            "public_events",
            "Recorded public events involving {company}.",
            0.80,
        ),
    )


class LegalCollector(PlaceholderCollector):
    """Placeholder evidence about incorporation, IP, and regulation."""

    collector_id = "legal_collector"
    display_name = "Legal"
    description = "Evidence about incorporation, IP, regulatory status, and litigation."
    supported_topics = ("legal",)
    source_identifier = "sec_edgar"
    source_category = "sec_edgar"
    claim_specs = (
        ClaimSpec(
            "incorporation_status",
            "Recorded the incorporation and entity status of {company}.",
            0.90,
        ),
        ClaimSpec(
            "intellectual_property",
            "Recorded the intellectual property portfolio of {company}.",
            0.85,
        ),
        ClaimSpec(
            "regulatory_status",
            "Recorded the regulatory status of {company}.",
            0.80,
        ),
    )


class RiskCollector(PlaceholderCollector):
    """Placeholder evidence about concentrated risks to the thesis."""

    collector_id = "risk_collector"
    display_name = "Risks"
    description = "Evidence about concentrated risks and threats to the thesis."
    supported_topics = ("risks",)
    source_identifier = "news"
    source_category = "news"
    claim_specs = (
        ClaimSpec(
            "concentration_risks",
            "Recorded concentration risks facing {company}.",
            0.80,
        ),
        ClaimSpec(
            "threat_landscape",
            "Recorded threat-landscape signals for {company}.",
            0.80,
        ),
        ClaimSpec(
            "thesis_risks",
            "Recorded thesis-critical risks for {company}.",
            0.85,
        ),
    )


class ReviewsCollector(PlaceholderCollector):
    """Placeholder evidence about qualitative sentiment from reviews."""

    collector_id = "reviews_collector"
    display_name = "Reviews"
    description = "Evidence about qualitative sentiment from reviews and feedback."
    supported_topics = ("reviews",)
    source_identifier = "g2"
    source_category = "g2"
    claim_specs = (
        ClaimSpec(
            "average_ratings",
            "Recorded aggregate review ratings for {company}.",
            0.85,
        ),
        ClaimSpec(
            "sentiment_patterns",
            "Recorded review sentiment patterns for {company}.",
            0.80,
        ),
        ClaimSpec(
            "review_volume",
            "Recorded review volume and recency signals for {company}.",
            0.75,
        ),
    )


class PartnershipCollector(PlaceholderCollector):
    """Placeholder evidence about distribution and strategic partners."""

    collector_id = "partnership_collector"
    display_name = "Partnerships"
    description = "Evidence about distribution, channel, and strategic partnerships."
    supported_topics = ("partnerships",)
    source_identifier = "press_releases"
    source_category = "press_releases"
    claim_specs = (
        ClaimSpec(
            "channel_partners",
            "Recorded the channel partners of {company}.",
            0.85,
        ),
        ClaimSpec(
            "strategic_alliances",
            "Recorded strategic alliances involving {company}.",
            0.80,
        ),
        ClaimSpec(
            "distribution_network",
            "Recorded the distribution network of {company}.",
            0.75,
        ),
    )


class HiringCollector(PlaceholderCollector):
    """Placeholder evidence about headcount growth and hiring signals."""

    collector_id = "hiring_collector"
    display_name = "Hiring"
    description = "Evidence about headcount growth and hiring signals."
    supported_topics = ("hiring",)
    source_identifier = "job_listings"
    source_category = "job_listings"
    claim_specs = (
        ClaimSpec(
            "hiring_signals",
            "Recorded hiring signals for {company}.",
            0.85,
        ),
        ClaimSpec(
            "role_demand",
            "Recorded role and seniority demand signals for {company}.",
            0.80,
        ),
        ClaimSpec(
            "headcount_trends",
            "Recorded headcount trend signals for {company}.",
            0.75,
        ),
    )


DEFAULT_COLLECTORS: tuple[EvidenceCollector, ...] = (
    FounderCollector(),
    ProductCollector(),
    TechnologyCollector(),
    MarketCollector(),
    CompetitionCollector(),
    FundingCollector(),
    PricingCollector(),
    TractionCollector(),
    CustomerCollector(),
    NewsCollector(),
    LegalCollector(),
    RiskCollector(),
    ReviewsCollector(),
    PartnershipCollector(),
    HiringCollector(),
)
