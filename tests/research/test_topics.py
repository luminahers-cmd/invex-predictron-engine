"""Tests for the deterministic research taxonomy."""

from __future__ import annotations

import pytest

from predictron_engine.research import (
    RESEARCH_TOPICS,
    SOURCE_CATEGORIES,
    TOPIC_REGISTRY,
    UnknownTopicError,
    all_topic_ids,
    dependency_edges,
    get_topic,
    has_topic,
    topic_index,
)

EXPECTED_TOPIC_IDS = (
    "founders",
    "team",
    "product",
    "technology",
    "market",
    "competition",
    "customers",
    "pricing",
    "traction",
    "business_model",
    "funding",
    "hiring",
    "partnerships",
    "legal",
    "reviews",
    "news",
    "risks",
)

# (dependent, required_prerequisite) pairs mandated by the sprint spec.
REQUIRED_DEPENDENCIES = (
    ("team", "founders"),
    ("market", "product"),
    ("customers", "product"),
    ("traction", "customers"),
    ("competition", "technology"),
)


class TestTaxonomy:
    """The taxonomy is complete, unique, and self-consistent."""

    def test_all_expected_topics_present(self) -> None:
        assert {topic.topic_id for topic in RESEARCH_TOPICS} == set(
            EXPECTED_TOPIC_IDS
        )

    def test_topic_ids_unique(self) -> None:
        ids = [topic.topic_id for topic in RESEARCH_TOPICS]
        assert len(ids) == len(set(ids))

    def test_names_human_readable_unique(self) -> None:
        names = [topic.name for topic in RESEARCH_TOPICS]
        assert all(name.strip() for name in names)
        assert len(names) == len(set(names))

    def test_taxonomy_order_and_registry_consistent(self) -> None:
        assert all_topic_ids() == EXPECTED_TOPIC_IDS
        assert set(TOPIC_REGISTRY) == set(all_topic_ids())
        for topic in RESEARCH_TOPICS:
            assert TOPIC_REGISTRY[topic.topic_id] == topic

    def test_every_topic_metadata_valid(self) -> None:
        for topic in RESEARCH_TOPICS:
            assert 0.0 <= topic.importance <= 1.0
            assert 0.0 <= topic.prediction_impact <= 1.0
            assert 0.0 <= topic.freshness_sensitivity <= 1.0
            assert topic.freshness_requirement_days >= 1
            assert topic.source_categories
            assert set(topic.source_categories) <= set(SOURCE_CATEGORIES)
        assert len(SOURCE_CATEGORIES) == len(set(SOURCE_CATEGORIES))


class TestDependencies:
    """Dependency edges required by the spec exist and form a DAG."""

    def test_required_dependency_pairs(self) -> None:
        for dependent, prerequisite in REQUIRED_DEPENDENCIES:
            assert prerequisite in get_topic(dependent).dependencies

    def test_dependencies_are_valid_references(self) -> None:
        for topic in RESEARCH_TOPICS:
            assert topic.topic_id not in topic.dependencies
            for dependency in topic.dependencies:
                assert has_topic(dependency)

    def test_dependency_graph_is_acyclic(self) -> None:
        # Kahn topological sort: a cycle would emit fewer than all topics.
        order_index = {
            topic.topic_id: i for i, topic in enumerate(RESEARCH_TOPICS)
        }
        pending = {
            topic.topic_id: set(topic.dependencies) for topic in RESEARCH_TOPICS
        }
        ready = {tid for tid, deps in pending.items() if not deps}
        emitted: list[str] = []
        while ready:
            tid = min(ready, key=lambda t: order_index[t])
            ready.remove(tid)
            emitted.append(tid)
            for candidate in pending:
                if tid in pending[candidate]:
                    pending[candidate].remove(tid)
                    if not pending[candidate]:
                        ready.add(candidate)
        assert len(emitted) == len(RESEARCH_TOPICS)

    def test_dependency_edges_deterministic(self) -> None:
        edges = dependency_edges()
        for edge in [
            ("founders", "team"),
            ("product", "market"),
            ("product", "customers"),
            ("customers", "traction"),
            ("technology", "competition"),
        ]:
            assert edge in edges


class TestTopicLookup:
    """Registry lookup helpers behave deterministically."""

    def test_get_topic_and_has_topic(self) -> None:
        assert get_topic("founders").name == "Founders"
        assert has_topic("market")
        assert not has_topic("nope")

    def test_unknown_topic_raises(self) -> None:
        for lookup in (get_topic, topic_index):
            with pytest.raises(UnknownTopicError):
                lookup("unicorns")

    def test_topic_index_matches_taxonomy_position(self) -> None:
        for position, topic in enumerate(RESEARCH_TOPICS):
            assert topic_index(topic.topic_id) == position
