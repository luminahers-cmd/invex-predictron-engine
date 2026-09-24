"""Tests for the Source Discovery immutable models.

Covers validation, immutability, and to_dict/from_dict serialization for
the Source Discovery value objects (ResearchSource, SourceScore,
SourceRank, SourceRecommendation, SourceDiscoveryPlan).
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from predictron_engine.research import (
    ResearchSource,
    SourceCategory,
    SourceDiscoveryPlan,
    SourceRank,
    SourceRecommendation,
    SourceScore,
)
from predictron_engine.research.models import ResearchPriority, ResearchTask

VALID_CATEGORY_VALUES = {
    "official_website",
    "company_blog",
    "engineering_blog",
    "documentation",
    "api_documentation",
    "developer_docs",
    "github",
    "news",
    "press_releases",
    "case_studies",
    "g2",
    "capterra",
    "app_store",
    "google_play",
    "reddit",
    "hacker_news",
    "linkedin",
    "wellfound",
    "job_listings",
    "careers",
    "crunchbase",
    "sec_edgar",
    "yc",
    "product_hunt",
    "patents",
    "research_papers",
}


def _source(**overrides: object) -> ResearchSource:
    base: dict[str, object] = {
        "identifier": "crunchbase",
        "display_name": "Crunchbase",
        "source_category": SourceCategory.CRUNCHBASE,
        "trust_score": 0.8,
        "freshness_score": 0.7,
        "coverage_score": 0.8,
        "relative_cost": 0.3,
        "supports_structured_data": True,
        "preferred_topics": ("funding", "founders"),
    }
    base.update(overrides)
    return ResearchSource(**base)  # type: ignore[arg-type]


def _task(
    *,
    topic_id: str = "funding",
    task_id: str = "research_funding",
) -> ResearchTask:
    return ResearchTask(
        task_id=task_id,
        topic_id=topic_id,
        title="Funding research",
        description="Assess capital raised.",
        priority=ResearchPriority.HIGH,
        priority_score=60.0,
        importance=0.8,
        prediction_impact=0.9,
        freshness_requirement_days=90,
        dependency_ids=(),
        source_categories=("funding_databases",),
        priority_rank=2,
        execution_order=1,
    )


def _score(
    tracking: bool = False, total: float = 0.8
) -> SourceScore:
    return SourceScore(
        trust=0.8,
        coverage=0.8,
        freshness=0.7,
        cost=0.7,
        structured_data=1.0,
        topic_fit=1.0 if tracking else 0.0,
        total=total,
    )


def _rank(identifier: str = "crunchbase", rank: int = 1) -> SourceRank:
    return SourceRank(
        rank=rank,
        source=_source(identifier=identifier),
        score=_score(True),
    )


class TestSourceCategory:
    """The category enum has the expected controlled vocabulary."""

    def test_members_cover_the_candidate_catalog(self) -> None:
        assert {c.value for c in SourceCategory} == VALID_CATEGORY_VALUES
        assert all(isinstance(c.value, str) for c in SourceCategory)

    def test_str_enum_lookup_by_value(self) -> None:
        assert SourceCategory("crunchbase") is SourceCategory.CRUNCHBASE
        with pytest.raises(ValueError):
            SourceCategory("bogus")


class TestResearchSource:
    """Validation and serialization of :class:`ResearchSource`."""

    def test_minimal_source(self) -> None:
        source = _source()
        assert source.identifier == "crunchbase"
        assert source.display_name == "Crunchbase"
        assert source.source_category is SourceCategory.CRUNCHBASE
        assert source.supports_structured_data is True
        assert source.preferred_topics == ("funding", "founders")

    def test_identifier_required(self) -> None:
        for value in ("", "   "):
            with pytest.raises(ValueError):
                _source(identifier=value)

    def test_display_name_required(self) -> None:
        for value in ("", "  "):
            with pytest.raises(ValueError):
                _source(display_name=value)

    def test_category_must_be_typed(self) -> None:
        with pytest.raises(ValueError):
            _source(source_category="crunchbase")  # type: ignore[arg-type]

    def test_scores_must_be_within_unit_range(self) -> None:
        for field in (
            "trust_score",
            "freshness_score",
            "coverage_score",
            "relative_cost",
        ):
            for value in (-0.1, 1.01):
                with pytest.raises(ValueError):
                    _source(**{field: value})

    def test_preferred_topics_non_empty_and_deduped(self) -> None:
        with pytest.raises(ValueError):
            _source(preferred_topics=())
        source = _source(preferred_topics=("funding", "founders", "funding"))
        assert source.preferred_topics == ("funding", "founders")

    def test_serialization_roundtrip(self) -> None:
        source = _source(description="Venture database.")
        assert ResearchSource.from_dict(source.to_dict()) == source
        raw = json.dumps(source.to_dict())
        assert ResearchSource.from_dict(json.loads(raw)) == source
        assert source.to_dict()["source_category"] == "crunchbase"

    def test_invalid_dict_raises(self) -> None:
        data = _source().to_dict()
        data.pop("identifier")
        with pytest.raises(ValueError):
            ResearchSource.from_dict(data)
        data = _source().to_dict()
        data["supports_structured_data"] = "yes"
        with pytest.raises(ValueError):
            ResearchSource.from_dict(data)

    def test_immutability(self) -> None:
        source = _source()
        with pytest.raises(FrozenInstanceError):
            source.trust_score = 0.99  # type: ignore[misc]


class TestSourceScore:
    """Component scores serialize and remain immutable."""

    def test_json_roundtrip_harmonizes_with_dict_roundtrip(self) -> None:
        score = _score(True, total=0.8)
        assert SourceScore.from_dict(score.to_dict()) == score
        assert SourceScore.from_dict(json.loads(json.dumps(score.to_dict()))) == score

    def test_immutability(self) -> None:
        score = _score()
        with pytest.raises(FrozenInstanceError):
            score.total = 1.0  # type: ignore[misc]


class TestSourceRank:
    """A ranked source carries rank, source, and score."""

    def test_construction_and_roundtrip(self) -> None:
        entry = _rank()
        assert entry.rank == 1
        restored = SourceRank.from_dict(entry.to_dict())
        assert restored == entry
        assert restored.source.identifier == "crunchbase"

    def test_rank_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            _rank(rank=0)


class TestSourceRecommendation:
    """Recommendation helpers and serialization."""

    def test_empty_recommendation(self) -> None:
        recommendation = SourceRecommendation(
            task_id="research_funding", topic_id="funding"
        )
        assert recommendation.sources == ()
        assert recommendation.top_source() is None
        assert not recommendation.has_recommendations()
        assert recommendation.source_identifiers() == ()

    def test_helpers_with_sources(self) -> None:
        recommendation = SourceRecommendation(
            task_id="research_funding",
            topic_id="funding",
            sources=(_rank("crunchbase", 1), _rank("sec_edgar", 2)),
        )
        assert recommendation.top_source().identifier == "crunchbase"
        assert recommendation.source_identifiers() == (
            "crunchbase",
            "sec_edgar",
        )
        assert recommendation.score_for("sec_edgar") is not None
        assert recommendation.score_for("nope") is None
        assert recommendation.has_recommendations()

    def test_task_and_topic_required(self) -> None:
        with pytest.raises(ValueError):
            SourceRecommendation(task_id="", topic_id="funding")
        with pytest.raises(ValueError):
            SourceRecommendation(task_id="t", topic_id="  ")

    def test_serialization_roundtrip(self) -> None:
        recommendation = SourceRecommendation(
            task_id="research_funding",
            topic_id="funding",
            sources=(_rank("crunchbase", 1), _rank("sec_edgar", 2)),
        )
        restored = SourceRecommendation.from_dict(recommendation.to_dict())
        assert restored == recommendation
        assert restored.source_identifiers() == ("crunchbase", "sec_edgar")


class TestSourceDiscoveryPlan:
    """The discovery plan preserves tasks and serializes losslessly."""

    def _plan(self, tasks: tuple[ResearchTask, ...] = (_task(),)):
        recommendations = tuple(
            SourceRecommendation(
                task_id=task.task_id, topic_id=task.topic_id, sources=(_rank(),)
            )
            for task in tasks
        )
        return SourceDiscoveryPlan(
            schema_version="1.0.0",
            plan_id="plan_abc",
            company_name="Acme",
            source_tasks=tasks,
            recommendations=recommendations,
        )

    def test_empty_plan(self) -> None:
        plan = SourceDiscoveryPlan(
            schema_version="1.0.0",
            plan_id="plan_abc",
            company_name="Acme",
            source_tasks=(),
            recommendations=(),
        )
        assert plan.topics_covered == ()
        assert plan.recommendation_for("funding") is None
        assert plan.sources_for("funding") == ()

    def test_topics_covered_and_lookup(self) -> None:
        plan = self._plan((_task(), _task(topic_id="legal", task_id="research_legal")))
        assert plan.topics_covered == ("funding", "legal")
        assert plan.recommendation_for("legal").topic_id == "legal"
        assert plan.recommendation_for("nope") is None
        assert [s.identifier for s in plan.sources_for("funding")] == [
            "crunchbase"
        ]
        assert plan.source_identifiers_for("funding") == ("crunchbase",)

    def test_recommendation_must_reference_known_task(self) -> None:
        with pytest.raises(ValueError):
            SourceDiscoveryPlan(
                schema_version="1.0.0",
                plan_id="plan",
                company_name="Acme",
                source_tasks=(_task(),),
                recommendations=(
                    SourceRecommendation(
                        task_id="research_market", topic_id="market"
                    ),
                ),
            )
        with pytest.raises(ValueError):
            SourceDiscoveryPlan(
                schema_version="1.0.0",
                plan_id="plan",
                company_name="Acme",
                source_tasks=(_task(),),
                recommendations=(
                    SourceRecommendation(
                        task_id="wrong_task", topic_id="funding"
                    ),
                ),
            )

    def test_requires_fields(self) -> None:
        with pytest.raises(ValueError):
            SourceDiscoveryPlan(
                schema_version="",
                plan_id="p",
                company_name="Acme",
                source_tasks=(),
                recommendations=(),
            )
        with pytest.raises(ValueError):
            SourceDiscoveryPlan(
                schema_version="1.0.0",
                plan_id="",
                company_name="Acme",
                source_tasks=(),
                recommendations=(),
            )

    def test_serialization_roundtrip(self) -> None:
        plan = self._plan(
            (_task(), _task(topic_id="legal", task_id="research_legal"))
        )
        assert SourceDiscoveryPlan.from_dict(plan.to_dict()) == plan
        raw = json.dumps(plan.to_dict())
        assert SourceDiscoveryPlan.from_dict(json.loads(raw)) == plan
        assert plan.to_dict()["topics_covered"] == ["funding", "legal"]
