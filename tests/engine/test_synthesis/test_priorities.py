"""Tests for deterministic recommendation prioritization (Sprint 6C)."""

from predictron_engine.synthesis.priorities import (
    _MAX_PRIORITIZED,
    prioritize_recommendations,
)
from tests.engine.test_synthesis.conftest import (
    make_citation,
    make_observation,
    make_recommendation,
)


class TestDeduplication:
    def test_identical_recommendations_merge_to_one(self):
        recs = [
            make_recommendation("Request the pitch deck.", priority="high"),
            make_recommendation("Request the pitch deck.", priority="high"),
        ]
        result = prioritize_recommendations(recs)
        assert len(result) == 1
        assert result[0].rank == 1

    def test_dedup_is_case_and_whitespace_insensitive(self):
        recs = [
            make_recommendation("Request  the pitch deck", category="Due_Diligence"),
            make_recommendation("Request the pitch deck.", category="due_diligence"),
        ]
        assert len(prioritize_recommendations(recs)) == 1

    def test_different_actions_are_not_merged(self):
        recs = [
            make_recommendation("Request the pitch deck."),
            make_recommendation("Request founder profiles."),
        ]
        assert len(prioritize_recommendations(recs)) == 2

    def test_different_categories_are_not_merged(self):
        recs = [
            make_recommendation("Request data.", category="due_diligence"),
            make_recommendation("Request data.", category="risk"),
        ]
        assert len(prioritize_recommendations(recs)) == 2

    def test_merge_keeps_highest_priority(self):
        recs = [
            make_recommendation("Validate traction metrics.", priority="low"),
            make_recommendation("Validate traction metrics.", priority="high"),
        ]
        result = prioritize_recommendations(recs)
        assert len(result) == 1
        assert result[0].priority == "high"

    def test_merge_takes_max_confidence_without_recomputing(self):
        recs = [
            make_recommendation("Validate traction metrics.", confidence=0.4),
            make_recommendation("Validate traction metrics.", confidence=0.9),
        ]
        result = prioritize_recommendations(recs)
        assert result[0].confidence == 0.9


class TestPreservation:
    def test_citations_unioned_and_preserved(self):
        citation_a = make_citation(claim="Claim A")
        citation_b = make_citation(claim="Claim B")
        duplicate_a = make_citation(claim="Claim A")
        recs = [
            make_recommendation(
                "Check references.", citations=[citation_a, duplicate_a]
            ),
            make_recommendation("Check references.", citations=[citation_b]),
        ]
        result = prioritize_recommendations(recs)
        claims = [c.claim for c in result[0].citations]
        assert claims == ["Claim A", "Claim B"]

    def test_supporting_observations_and_provenance_preserved(self):
        obs_one = make_observation(provenance_document_ids=["doc-1"])
        obs_two = make_observation(statement="Second", provenance_document_ids=["doc-2"])
        recs = [
            make_recommendation("Check unit economics.", supporting_observations=[obs_one]),
            make_recommendation("Check unit economics.", supporting_observations=[obs_two]),
        ]
        result = prioritize_recommendations(recs)
        merged = result[0]
        assert [obs.statement for obs in merged.supporting_observations] == [
            "Observation statement",
            "Second",
        ]
        doc_ids = [
            doc_id
            for obs in merged.supporting_observations
            for doc_id in obs.provenance_document_ids
        ]
        assert set(doc_ids) == {"doc-1", "doc-2"}

    def test_expected_confidence_fields_survive_merge(self):
        recs = [
            make_recommendation("Hire a CFO.", expected_confidence=0.6),
            make_recommendation("Hire a CFO.", expected_confidence=0.7),
        ]
        result = prioritize_recommendations(recs)
        assert result[0].expected_confidence == 0.7

    def test_original_recommendations_are_not_mutated(self):
        original = make_recommendation("Request the pitch deck.")
        prioritize_recommendations([original])
        assert original.rank is None


class TestOrdering:
    def test_high_sorts_above_medium_above_low(self):
        recs = [
            make_recommendation("Low task.", priority="low"),
            make_recommendation("Medium task.", priority="medium"),
            make_recommendation("High task.", priority="high"),
        ]
        result = prioritize_recommendations(recs)
        actions = [rec.action for rec in result]
        assert actions == ["High task.", "Medium task.", "Low task."]

    def test_ordering_is_deterministic_regardless_of_input_order(self):
        rec_set = [
            make_recommendation("Alpha action.", priority="high", confidence=0.9),
            make_recommendation("Beta action.", priority="medium", confidence=0.5),
            make_recommendation("Gamma action.", priority="low", confidence=0.2),
        ]
        forward = [r.action for r in prioritize_recommendations(rec_set)]
        reversed_input = list(reversed(rec_set))
        backward = [r.action for r in prioritize_recommendations(reversed_input)]
        assert forward == backward

    def test_expected_confidence_breaks_ties(self):
        recs = [
            make_recommendation("First tied action.", priority="high", confidence=0.8),
            make_recommendation(
                "Second tied action.",
                priority="high",
                confidence=0.8,
                expected_confidence=0.95,
            ),
        ]
        result = prioritize_recommendations(recs)
        assert result[0].action == "Second tied action."

    def test_ranks_are_contiguous_starting_at_one(self):
        recs = [
            make_recommendation(f"Action {index}.") for index in range(5)
        ]
        ranks = [rec.rank for rec in prioritize_recommendations(recs)]
        assert ranks == [1, 2, 3, 4, 5]


class TestCapping:
    def test_default_cap_of_eight(self):
        recs = [
            make_recommendation(f"Unique action number {index}.") for index in range(12)
        ]
        result = prioritize_recommendations(recs)
        assert len(result) == _MAX_PRIORITIZED

    def test_custom_max_count(self):
        recs = [
            make_recommendation(f"Unique action number {index}.") for index in range(10)
        ]
        assert len(prioritize_recommendations(recs, max_count=3)) == 3

    def test_empty_input_returns_empty(self):
        assert prioritize_recommendations([]) == []
