"""Tests for the CollectorRegistry and registry convenience API.

Covers registration, unregistration, deterministic lookup, topic
resolution, duplicate detection, invalid inputs, covered-topic
enumeration, and the module-level helpers bound to the shared registry.
"""

from __future__ import annotations

import pytest

from predictron_engine.research import (
    COLLECTOR_REGISTRY,
    DEFAULT_COLLECTORS,
    CollectorRegistry,
    EvidenceCollector,
    ResearchPriority,
    ResearchTask,
    collectors_for,
    covered_topics,
    default_collector_registry,
    has_collector,
    list_collectors,
    register_collector,
    resolve_collector,
    unregister_collector,
)
from predictron_engine.research.exceptions import (
    CollectorNotFound,
    DuplicateCollector,
)


def _task(topic_id: str, task_id: str) -> ResearchTask:
    return ResearchTask(
        task_id=task_id,
        topic_id=topic_id,
        title=f"{topic_id} research",
        description="Research.",
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


class _SingleCollector(EvidenceCollector):
    """Test collector supporting exactly one topic."""

    collector_id = "single_collector"
    display_name = "Single"
    supported_topics = ("funding",)

    def collect(
        self,
        task: ResearchTask,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple:
        del task, company_name, website
        return ()


class _OverlapCollector(EvidenceCollector):
    """Test collector overlapping with :class:`_SingleCollector`."""

    collector_id = "overlap_collector"
    display_name = "Overlap"
    supported_topics = ("funding", "pricing")

    def collect(
        self,
        task: ResearchTask,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple:
        del task, company_name, website
        return ()


class _OtherCollector(EvidenceCollector):
    """Test collector for a distinct topic."""

    collector_id = "other_collector"
    display_name = "Other"
    supported_topics = ("pricing",)

    def collect(
        self,
        task: ResearchTask,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple:
        del task, company_name, website
        return ()


class _ModuleTeamCollector(EvidenceCollector):
    """Team collector with a unique id for shared-registry module tests."""

    collector_id = "module_team_collector"
    display_name = "Module Team"
    supported_topics = ("team",)

    def collect(
        self,
        task: ResearchTask,
        *,
        company_name: str,
        website: str = "",
    ) -> tuple:
        del task, company_name, website
        return ()


class TestConstruction:
    """Registries start empty or preloaded."""

    def test_empty_registry(self) -> None:
        registry = CollectorRegistry()
        assert len(registry) == 0
        assert registry.list_collectors() == ()

    def test_preloaded_registry_preserves_order(self) -> None:
        registry = CollectorRegistry((_SingleCollector(), _OtherCollector()))
        assert [c.metadata().collector_id for c in registry.list_collectors()] == [
            "single_collector",
            "other_collector",
        ]

    def test_non_collector_entry_rejected(self) -> None:
        with pytest.raises(TypeError):
            CollectorRegistry(("funding",))  # type: ignore[arg-type]

    def test_default_registry_preloads_all_defaults(self) -> None:
        registry = default_collector_registry()
        assert len(registry) == len(DEFAULT_COLLECTORS)
        assert registry.list_collectors() == DEFAULT_COLLECTORS

    def test_shared_registry_preloads_all_defaults(self) -> None:
        assert len(COLLECTOR_REGISTRY) == len(DEFAULT_COLLECTORS)


class TestRegister:
    """Registration enforces uniqueness and ordering."""

    def test_register_returns_collector(self) -> None:
        registry = CollectorRegistry()
        collector = _SingleCollector()
        assert registry.register(collector) is collector

    def test_duplicate_collector_rejected(self) -> None:
        registry = CollectorRegistry()
        registry.register(_SingleCollector())
        with pytest.raises(DuplicateCollector):
            registry.register(_SingleCollector())

    def test_duplicate_across_instances_rejected(self) -> None:
        registry = CollectorRegistry((_SingleCollector(), _OverlapCollector()))

        class _Clone(EvidenceCollector):
            collector_id = "single_collector"
            display_name = "Clone"
            supported_topics = ("team",)

            def collect(
                self,
                task: ResearchTask,
                *,
                company_name: str,
                website: str = "",
            ) -> tuple:
                del task, company_name, website
                return ()

        with pytest.raises(DuplicateCollector):
            registry.register(_Clone())

    def test_non_collector_register_rejected(self) -> None:
        with pytest.raises(TypeError):
            CollectorRegistry().register("funding")  # type: ignore[arg-type]

    def test_register_preserves_order(self) -> None:
        registry = CollectorRegistry()
        registry.register(_OtherCollector())
        registry.register(_SingleCollector())
        assert [c.metadata().collector_id for c in registry.list_collectors()] == [
            "other_collector",
            "single_collector",
        ]


class TestUnregister:
    """Unregistration removes and returns collectors."""

    def test_unregister_returns_removed_collector(self) -> None:
        registry = CollectorRegistry((_SingleCollector(),))
        removed = registry.unregister("single_collector")
        assert removed is not None
        assert removed.metadata().collector_id == "single_collector"
        assert len(registry) == 0

    def test_unregister_missing_is_idempotent(self) -> None:
        registry = CollectorRegistry()
        assert registry.unregister("missing") is None

    def test_unregister_makes_collector_resolvable_again(self) -> None:
        registry = CollectorRegistry((_SingleCollector(),))
        registry.unregister("single_collector")
        with pytest.raises(CollectorNotFound):
            registry.resolve("funding")


class TestLookup:
    """Lookups are deterministic and registration-ordered."""

    def test_get_and_has_collector(self) -> None:
        registry = CollectorRegistry((_SingleCollector(), _OtherCollector()))
        assert registry.has_collector("single_collector")
        assert not registry.has_collector("missing")
        assert registry.get("other_collector") is not None
        assert registry.get("missing") is None
        assert "single_collector" in registry
        assert "missing" not in registry

    def test_list_collectors_returns_snapshot(self) -> None:
        registry = CollectorRegistry((_SingleCollector(),))
        listed = registry.list_collectors()
        registry.unregister("single_collector")
        assert len(listed) == 1
        assert listed[0].metadata().collector_id == "single_collector"

    def test_collectors_for_returns_matches_in_order(self) -> None:
        registry = CollectorRegistry((_SingleCollector(), _OverlapCollector()))
        matches = registry.collectors_for("funding")
        assert [c.metadata().collector_id for c in matches] == [
            "single_collector",
            "overlap_collector",
        ]
        assert registry.collectors_for("product") == ()

    def test_resolve_prefers_first_registered(self) -> None:
        registry = CollectorRegistry((_SingleCollector(), _OverlapCollector()))
        assert (
            registry.resolve("funding").metadata().collector_id
            == "single_collector"
        )
        reversed_registry = CollectorRegistry(
            (_OverlapCollector(), _SingleCollector())
        )
        assert (
            reversed_registry.resolve("funding").metadata().collector_id
            == "overlap_collector"
        )

    def test_resolve_unknown_topic_raises(self) -> None:
        registry = CollectorRegistry((_SingleCollector(),))
        with pytest.raises(CollectorNotFound):
            registry.resolve("product")
        with pytest.raises(CollectorNotFound):
            registry.resolve("team")

    def test_resolution_is_deterministic(self) -> None:
        registry = CollectorRegistry((_SingleCollector(),))
        assert registry.resolve("funding") is registry.resolve("funding")


class TestCoveredTopics:
    """Supported topics enumerate sorted and unique."""

    def test_covered_topics_union_sorted(self) -> None:
        registry = CollectorRegistry((_SingleCollector(), _OtherCollector()))
        assert registry.topics_covered() == ("funding", "pricing")

    def test_empty_registry_covers_nothing(self) -> None:
        assert CollectorRegistry().topics_covered() == ()


class TestModuleFunctions:
    """Module helpers operate on the shared registry with cleanup."""

    def test_register_resolve_unregister_cycle(self) -> None:
        collector = _ModuleTeamCollector()
        try:
            register_collector(collector)
            assert has_collector("module_team_collector")
            assert (
                resolve_collector("team").metadata().collector_id
                == "module_team_collector"
            )
        finally:
            assert unregister_collector("module_team_collector") is not None

    def test_module_list_and_collectors_for(self) -> None:
        before = len(list_collectors())
        collector = _ModuleTeamCollector()
        try:
            register_collector(collector)
            assert len(list_collectors()) == before + 1
            assert any(
                c.metadata().collector_id == "module_team_collector"
                for c in collectors_for("team")
            )
        finally:
            unregister_collector("module_team_collector")

    def test_module_covered_topics_matches_registry(self) -> None:
        assert covered_topics() == COLLECTOR_REGISTRY.topics_covered()
        assert "founders" in covered_topics()


class TestIntegration:
    """Registry + dispatcher-style resolution stays type-safe."""

    def test_registered_collector_executes_supporting_task(self) -> None:
        registry = CollectorRegistry((_SingleCollector(),))
        collector = registry.resolve("funding")
        assert collector.supports(_task("funding", "research_funding").topic_id)
