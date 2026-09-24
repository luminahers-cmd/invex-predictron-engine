"""Tests for the Research Planner immutable models.

Covers validation, immutability, and to_dict/from_dict serialization for
the planning value objects.
"""

from __future__ import annotations

import pytest

from predictron_engine.research import (
    EvidenceStatus,
    InvalidPlannerInputError,
    KnowledgeGap,
    PlannerInput,
    ResearchPriority,
    ResearchTopic,
)
from predictron_engine.research.models import ResearchTask


def _topic(**overrides: object) -> ResearchTopic:
    base: dict[str, object] = {
        "topic_id": "market",
        "name": "Market",
        "description": "Assess market size and structure.",
        "importance": 0.85,
        "prediction_impact": 0.90,
        "freshness_requirement_days": 180,
        "freshness_sensitivity": 0.60,
        "source_categories": ("market_reports", "web_search"),
    }
    base.update(overrides)
    return ResearchTopic(**base)  # type: ignore[arg-type]


class TestPlannerInput:
    """Validation and serialization of :class:`PlannerInput`."""

    def test_minimal_planner_input(self) -> None:
        value = PlannerInput(company_name="Acme")
        assert value.company_name == "Acme"
        assert value.website == ""
        assert value.description == ""
        assert value.known_topics == ()
        assert value.prediction_horizon_days == 365

    def test_company_name_required(self) -> None:
        for name in ("", "   "):
            with pytest.raises(InvalidPlannerInputError):
                PlannerInput(company_name=name)

    def test_website_validation(self) -> None:
        for website in ("ftp://x.com", "not-a-url", "http://"):
            with pytest.raises(InvalidPlannerInputError):
                PlannerInput(company_name="Acme", website=website)
        for website in ("https://acme.io", "http://acme.io/path"):
            value = PlannerInput(company_name="Acme", website=website)
            assert value.website == website

    def test_prediction_horizon_must_be_positive(self) -> None:
        for days in (0, -5):
            with pytest.raises(InvalidPlannerInputError):
                PlannerInput(company_name="Acme", prediction_horizon_days=days)

    def test_known_topics_deduped_and_consistent(self) -> None:
        value = PlannerInput(
            company_name="Acme",
            known_topics=("market", "founders", "market"),
        )
        assert value.known_topics == ("market", "founders")
        with pytest.raises(InvalidPlannerInputError):
            PlannerInput(
                company_name="Acme",
                known_topics=("market",),
                partial_topics=("market",),
            )

    def test_serialization_roundtrip(self) -> None:
        value = PlannerInput(
            company_name="Acme",
            website="https://acme.io",
            description="Doing things",
            known_topics=("market", "pricing"),
            partial_topics=("news",),
        )
        assert PlannerInput.from_dict(value.to_dict()) == value


class TestEnums:
    """Enum values serialize to stable strings."""

    def test_priority_order_and_evidence_values(self) -> None:
        assert {p.value for p in ResearchPriority} == {
            "critical",
            "high",
            "medium",
            "low",
        }
        assert (
            ResearchPriority.CRITICAL.order_value()
            > ResearchPriority.HIGH.order_value()
            > ResearchPriority.MEDIUM.order_value()
            > ResearchPriority.LOW.order_value()
        )
        assert {s.value for s in EvidenceStatus} == {
            "none",
            "partial",
            "complete",
        }


class TestResearchTopic:
    """Validation and serialization of :class:`ResearchTopic`."""

    def test_unit_range_enforced(self) -> None:
        for field, value in [
            ("importance", 1.1),
            ("prediction_impact", -0.1),
            ("freshness_sensitivity", 1.5),
        ]:
            with pytest.raises(ValueError):
                _topic(**{field: value})

    def test_structure_and_roundtrip(self) -> None:
        with pytest.raises(ValueError):
            _topic(dependencies=("market",))  # self-dependency
        with pytest.raises(ValueError):
            _topic(source_categories=())
        topic = _topic(dependencies=("product",))
        assert ResearchTopic.from_dict(topic.to_dict()) == topic


class TestGapAndTaskSerialization:
    """KnowledgeGap and ResearchTask round-trip through dictionaries."""

    def test_knowledge_gap_and_task_roundtrip(self) -> None:
        gap = KnowledgeGap(
            topic_id="founders",
            evidence_status=EvidenceStatus.PARTIAL,
            reason="Partial founders evidence.",
        )
        restored_gap = KnowledgeGap.from_dict(gap.to_dict())
        assert restored_gap == gap
        assert gap.to_dict()["evidence_status"] == "partial"

        task = ResearchTask(
            task_id="research_founders",
            topic_id="founders",
            title="Founders research",
            description="Assess the founders.",
            priority=ResearchPriority.HIGH,
            priority_score=62.0,
            importance=0.9,
            prediction_impact=0.9,
            freshness_requirement_days=365,
            dependency_ids=(),
            source_categories=("linkedin",),
            priority_rank=3,
            execution_order=1,
        )
        restored_task = ResearchTask.from_dict(task.to_dict())
        assert restored_task == task
        assert restored_task.priority is ResearchPriority.HIGH
