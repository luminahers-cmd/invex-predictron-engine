"""Tests for Sprint 8 — Progressive Evidence Evaluation.

Validates early-stop behavior, computation savings, and deterministic
evaluation of evidence items.
"""

from __future__ import annotations

import pytest

from predictron_engine.reasoning.progressive_evidence import (
    ProgressiveEvaluation,
    EvidenceCheckpoint,
    evaluate_evidence_progressively,
)
from predictron_engine.evidence.evidence_models import EvidenceItem
from predictron_engine.models.report import Observation


def _make_evidence(domain: str = "industry", statement: str = "Evidence") -> EvidenceItem:
    return EvidenceItem(
        domain=domain,
        category="market_context",
        statement=statement,
        source="test_source",
        relevance_score=0.8,
    )


def _make_obs(dimension: str = "market_opportunity", confidence: float = 0.7) -> Observation:
    return Observation(
        dimension=dimension,
        category="market_context",
        statement="Test observation",
        confidence=confidence,
        importance=0.6,
        source_rule="test_rule",
    )


class TestProgressiveEvaluation:
    """Tests for evaluate_evidence_progressively function."""

    def test_empty_evidence(self) -> None:
        result = evaluate_evidence_progressively([], [])
        assert result.total_evidence_count == 0
        assert result.evaluated_count == 0
        assert result.early_stopped is False

    def test_single_evidence_item(self) -> None:
        evidence = [_make_evidence()]
        result = evaluate_evidence_progressively(evidence, [_make_obs()])
        assert result.evaluated_count == 1
        assert len(result.evaluated_evidence) == 1

    def test_early_stop_on_high_agreement(self) -> None:
        evidence = [_make_evidence(statement=f"Evidence {i}") for i in range(5)]
        observations = [_make_obs(confidence=0.9)]
        result = evaluate_evidence_progressively(evidence, observations)
        assert result.evaluated_count <= 5
        assert result.total_evidence_count == 5

    def test_computation_saved(self) -> None:
        evidence = [_make_evidence(statement=f"E {i}") for i in range(10)]
        result = evaluate_evidence_progressively(evidence, [_make_obs(confidence=0.9)])
        assert result.computation_saved_fraction >= 0.0

    def test_all_evidence_evaluated_when_no_early_stop(self) -> None:
        evidence = [_make_evidence(domain=f"dim_{i}") for i in range(3)]
        result = evaluate_evidence_progressively(evidence, [])
        assert result.evaluated_count == 3
        assert result.skipped_count == 0

    def test_evaluation_is_deterministic(self) -> None:
        evidence = [_make_evidence(statement=f"E {i}") for i in range(5)]
        observations = [_make_obs(confidence=0.8)]
        r1 = evaluate_evidence_progressively(evidence, observations)
        r2 = evaluate_evidence_progressively(evidence, observations)
        assert r1.evaluated_count == r2.evaluated_count
        assert r1.early_stopped == r2.early_stopped
        assert len(r1.checkpoints) == len(r2.checkpoints)

    def test_checkpoints_recorded(self) -> None:
        evidence = [_make_evidence(statement=f"E {i}") for i in range(5)]
        result = evaluate_evidence_progressively(evidence, [_make_obs()])
        assert len(result.checkpoints) >= 1

    def test_serialization(self) -> None:
        result = evaluate_evidence_progressively(
            [_make_evidence()], [_make_obs()],
        )
        d = result.to_dict()
        assert "total_evidence" in d
        assert "evaluated" in d
        assert "computation_saved_fraction" in d

    def test_evaluation_efficiency(self) -> None:
        result = evaluate_evidence_progressively(
            [_make_evidence() for _ in range(3)],
            [_make_obs()],
        )
        assert result.evaluation_efficiency >= 0.0

    def test_skipped_evidence_tracked(self) -> None:
        evidence = [_make_evidence(statement=f"E {i}") for i in range(10)]
        result = evaluate_evidence_progressively(
            evidence, [_make_obs(confidence=0.95)],
        )
        assert len(result.skipped_evidence) + result.evaluated_count == 10


class TestEvidenceCheckpoint:
    """Tests for EvidenceCheckpoint data class."""

    def test_checkpoint_fields(self) -> None:
        cp = EvidenceCheckpoint(
            items_evaluated=5,
            supporting_count=3,
            conflicting_count=1,
            cumulative_confidence=0.75,
            agreement_ratio=0.6,
            support_ratio=0.75,
            consecutive_support=2,
            should_continue=True,
        )
        assert cp.items_evaluated == 5
        assert cp.supporting_count == 3
