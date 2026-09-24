"""Tests for the EvidenceCollector interface and default placeholder collectors.

Covers the abstract contract, shared placeholder base behaviour,
determinism, unsupported-topic handling, invalid inputs, metadata, and
the 15 default collectors.
"""

from __future__ import annotations

import json

import pytest

from predictron_engine.research import (
    DEFAULT_COLLECTORS,
    PLACEHOLDER_EVIDENCE_ORIGIN,
    ClaimSpec,
    CompetitionCollector,
    CustomerCollector,
    Evidence,
    EvidenceCollector,
    EvidenceReference,
    FounderCollector,
    FundingCollector,
    HiringCollector,
    LegalCollector,
    MarketCollector,
    NewsCollector,
    PartnershipCollector,
    PlaceholderCollector,
    PricingCollector,
    ProductCollector,
    ResearchPriority,
    ResearchTask,
    ReviewsCollector,
    RiskCollector,
    TechnologyCollector,
    TractionCollector,
    UnsupportedTopic,
    enforce_supported,
)
from predictron_engine.research.exceptions import InvalidCollectionInputError

# (collector class, canonical topic)
COLLECTOR_TOPICS: tuple[tuple[type[PlaceholderCollector], str], ...] = (
    (FounderCollector, "founders"),
    (ProductCollector, "product"),
    (TechnologyCollector, "technology"),
    (MarketCollector, "market"),
    (CompetitionCollector, "competition"),
    (FundingCollector, "funding"),
    (PricingCollector, "pricing"),
    (TractionCollector, "traction"),
    (CustomerCollector, "customers"),
    (NewsCollector, "news"),
    (LegalCollector, "legal"),
    (RiskCollector, "risks"),
    (ReviewsCollector, "reviews"),
    (PartnershipCollector, "partnerships"),
    (HiringCollector, "hiring"),
)


def _task(topic_id: str = "funding", task_id: str = "research_funding") -> ResearchTask:
    return ResearchTask(
        task_id=task_id,
        topic_id=topic_id,
        title=f"{topic_id} research",
        description=f"Research the {topic_id} topic.",
        priority=ResearchPriority.HIGH,
        priority_score=60.0,
        importance=0.8,
        prediction_impact=0.9,
        freshness_requirement_days=90,
        dependency_ids=(),
        source_categories=(),
        priority_rank=1,
        execution_order=1,
    )


class StaticCollector(PlaceholderCollector):
    """A minimal deterministic collector used across these tests."""

    collector_id = "static_collector"
    display_name = "Static"
    description = "A static test collector."
    supported_topics = ("funding", "team")
    source_identifier = "crunchbase"
    source_category = "crunchbase"
    claim_specs = (
        ClaimSpec("funding_rounds", "Recorded that {company} raised capital.", 0.9),
    )


class TestInterface:
    """EvidenceCollector enforces the abstract contract."""

    def test_cannot_instantiate_abstract_collector(self) -> None:
        with pytest.raises(TypeError):
            EvidenceCollector()  # type: ignore[abstract]

    def test_supports_and_metadata_from_declaration(self) -> None:
        collector = StaticCollector()
        assert collector.supports("funding")
        assert collector.supports("team")
        assert not collector.supports("product")
        metadata = collector.metadata()
        assert metadata.collector_id == "static_collector"
        assert metadata.display_name == "Static"
        assert metadata.supported_topics == ("funding", "team")
        assert metadata.deterministic is True

    def test_metadata_serializable(self) -> None:
        metadata = StaticCollector().metadata()
        raw = json.dumps(metadata.to_dict())
        assert "static_collector" in raw
        assert metadata.to_dict()["supported_topics"] == ["funding", "team"]

    def test_missing_configuration_fails_loudly(self) -> None:
        with pytest.raises(TypeError):

            class _Broken(PlaceholderCollector):
                pass

    def test_enforce_supported_raises_unsupported_topic(self) -> None:
        collector = StaticCollector()
        with pytest.raises(UnsupportedTopic):
            enforce_supported(collector, "product")
        enforce_supported(collector, ("funding", "team"))
        with pytest.raises(UnsupportedTopic):
            enforce_supported(collector, ("funding", "product"))


class TestPlaceholderBehaviour:
    """The shared base turns declarations into evidence."""

    def test_collect_returns_deterministic_evidence(self) -> None:
        collector = StaticCollector()
        task = _task()
        first = collector.collect(task, company_name="Acme")
        second = collector.collect(task, company_name="Acme")
        assert first == second
        assert len(first) == 1
        item = first[0]
        assert isinstance(item, Evidence)
        assert item.collector_id == "static_collector"
        assert item.task_id == "research_funding"
        assert item.topic_id == "funding"
        assert item.category == "funding_rounds"
        assert item.claim == "Recorded that Acme raised capital."
        assert item.confidence == 0.9

    def test_evidence_is_content_addressed_and_serializable(self) -> None:
        item = StaticCollector().collect(_task(), company_name="Acme")[0]
        assert item.evidence_id.startswith("ev_")
        restored = Evidence.from_dict(json.loads(json.dumps(item.to_dict())))
        assert restored == item

    def test_reference_points_at_placeholder_origin(self) -> None:
        task = _task()
        item = StaticCollector().collect(task, company_name="Acme")[0]
        reference = item.reference
        assert isinstance(reference, EvidenceReference)
        assert reference.source_category == "crunchbase"
        assert reference.source_identifier == "crunchbase"
        assert reference.url.startswith(PLACEHOLDER_EVIDENCE_ORIGIN)
        assert task.topic_id in reference.url

    def test_company_and_topic_interpolate_into_claim(self) -> None:
        claim = StaticCollector().collect(_task(), company_name="  Acme  ")[
            0
        ].claim
        assert claim == "Recorded that Acme raised capital."

    def test_non_task_rejected(self) -> None:
        collector = StaticCollector()
        with pytest.raises(InvalidCollectionInputError):
            collector.collect("funding", company_name="Acme")  # type: ignore[arg-type]

    def test_blank_company_rejected(self) -> None:
        with pytest.raises(InvalidCollectionInputError):
            StaticCollector().collect(_task(), company_name="   ")

    def test_unsupported_topic_rejected(self) -> None:
        with pytest.raises(UnsupportedTopic):
            StaticCollector().collect(_task(topic_id="product"), company_name="Acme")

    def test_requires_every_claim_spec_to_be_typed(self) -> None:
        with pytest.raises(TypeError):

            class _BadSpecs(PlaceholderCollector):
                collector_id = "bad_specs"
                display_name = "Bad"
                supported_topics = ("funding",)
                source_identifier = "crunchbase"
                source_category = "crunchbase"
                claim_specs = (("funding_rounds", "Claim.", 0.9),)  # type: ignore[assignment]


class TestDefaultCollectors:
    """The 15 default collectors are complete and deterministic."""

    def test_default_collectors_count(self) -> None:
        assert len(DEFAULT_COLLECTORS) == 15

    def test_every_named_collector_registered(self) -> None:
        registered = {collector.metadata().collector_id for collector in DEFAULT_COLLECTORS}
        expected = {collector_cls.collector_id for collector_cls, _ in COLLECTOR_TOPICS}
        assert registered == expected

    def test_each_collector_covers_only_its_topic(self) -> None:
        for collector in DEFAULT_COLLECTORS:
            metadata = collector.metadata()
            assert len(metadata.supported_topics) == 1

    def test_collector_specificity_matches_taxonomy(self) -> None:
        for collector_cls, topic in COLLECTOR_TOPICS:
            collector = collector_cls()
            assert collector.supports(topic)
            metadata = collector.metadata()
            assert metadata.supported_topics == (topic,)
            assert metadata.display_name
            assert metadata.collector_id.endswith("_collector")

    def test_each_collector_emits_three_evidence_items(self) -> None:
        collector = FundingCollector()
        task = _task()
        items = collector.collect(task, company_name="Acme")
        assert len(items) == 3
        assert len({item.category for item in items}) == 3

    def test_evidence_fields_populated_for_default(self) -> None:
        items = FounderCollector().collect(_task(topic_id="founders"), company_name="Acme")
        assert all(item.topic_id == "founders" for item in items)
        assert all(item.collector_id == "founder_collector" for item in items)
        assert all(0.0 <= item.confidence <= 1.0 for item in items)
        assert all(item.claim for item in items)

    def test_default_collectors_are_statically_identical(self) -> None:
        first = FundingCollector().collect(_task(), company_name="Acme")
        second = FundingCollector().collect(_task(), company_name="Acme")
        assert first == second
        assert [item.evidence_id for item in first] == [
            item.evidence_id for item in second
        ]

    def test_different_companies_change_claims_but_not_identity(self) -> None:
        acme = FundingCollector().collect(_task(), company_name="Acme")
        globex = FundingCollector().collect(_task(), company_name="Globex")
        assert [item.claim for item in acme] != [item.claim for item in globex]
        assert [item.evidence_id for item in acme] != [
            item.evidence_id for item in globex
        ]

    def test_default_topic_coverage_matches_expected_set(self) -> None:
        covered = sorted(
            topic
            for collector in DEFAULT_COLLECTORS
            for topic in collector.metadata().supported_topics
        )
        assert covered == sorted(topic for _, topic in COLLECTOR_TOPICS)
