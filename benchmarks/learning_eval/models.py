"""Benchmark run and report data contracts for the learning-eval suite.

Every ``LearningAssertion`` records one deterministic check performed by the
suite; every run aggregates them into a scrollable result that is stored
append-only in ``benchmarks/learning_eval/history``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from predictron_engine.learning import (
    LearningConfidence,
    LearningDigest,
    LearningPattern,
    LearningSnapshot,
)

__all__ = [
    "AssertionResult",
    "LearningEvalCaseResult",
    "LearningEvalReport",
    "LearningEvalRun",
    "MetricBounds",
]


@dataclass(frozen=True)
class AssertionResult:
    """Outcome of one deterministic assertion against the engine output."""

    name: str
    passed: bool
    detail: str = ""

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass(frozen=True)
class LearningEvalCaseResult:
    """Assertions collected for one eval-case fixture."""

    case_id: str
    assertions: list[AssertionResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(a.passed for a in self.assertions)

    @property
    def total(self) -> int:
        return len(self.assertions)


@dataclass(frozen=True)
class LearningEvalRun:
    """One immutable execution of the learning-eval suite."""

    run_id: str
    engine_version: str
    created_at: str
    cases: list[LearningEvalCaseResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.cases)

    @property
    def total_assertions(self) -> int:
        return sum(c.total for c in self.cases)

    @property
    def total_passed(self) -> int:
        return sum(
            a.passed for c in self.cases for a in c.assertions
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "engine_version": self.engine_version,
            "created_at": self.created_at,
            "passed": self.passed,
            "total_assertions": self.total_assertions,
            "total_passed": self.total_passed,
            "cases": [
                {
                    "case_id": c.case_id,
                    "passed": c.passed,
                    "assertions": [
                        {
                            "name": a.name,
                            "passed": a.passed,
                            "detail": a.detail,
                        }
                        for a in c.assertions
                    ],
                }
                for c in self.cases
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearningEvalRun:
        """Rehydrate a run from a stored dict without trusting its integrity."""
        cases = []
        for case in data.get("cases", []):
            assertions = [
                AssertionResult(
                    name=item["name"],
                    passed=bool(item["passed"]),
                    detail=str(item.get("detail", "")),
                )
                for item in case.get("assertions", [])
            ]
            cases.append(
                LearningEvalCaseResult(
                    case_id=str(case.get("case_id", "")),
                    assertions=assertions,
                )
            )
        return cls(
            run_id=str(data["run_id"]),
            engine_version=str(data.get("engine_version", "")),
            created_at=str(data.get("created_at", "")),
            cases=cases,
        )


@dataclass(frozen=True)
class MetricBounds:
    """Upper bounds a fixture expects the engine to stay within."""

    max_ece: float = 0.0
    max_bias: float = 0.0


@dataclass(frozen=True)
class LearningEvalReport:
    """Human/JSON consumable summary of a suite run."""

    run: LearningEvalRun
    snapshots: dict[str, LearningSnapshot]
    patterns: dict[str, dict[str, LearningPattern]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run.to_dict(),
            "snapshots": {
                case_id: _snapshot_summary(snapshot)
                for case_id, snapshot in self.snapshots.items()
            },
            "patterns": _patterns_summary(self.patterns),
        }


def _snapshot_summary(snapshot: LearningSnapshot) -> dict[str, Any]:
    digest = snapshot.digest
    confidence = snapshot.confidence
    return {
        "snapshot_id": snapshot.snapshot_id,
        "content_hash": snapshot.content_hash,
        "schema_version": snapshot.schema_version,
        "verify": snapshot.verify(),
        "evaluation_count": digest.evaluation_count,
        "sample_count": digest.sample_count,
        "scoreable": digest.scoreable,
        "counts": dict(snapshot.counts),
        "metrics": dict(snapshot.metrics),
        "digest": _digest_summary(digest),
        "confidence": _confidence_summary(confidence),
        "distributions": {
            dimension: dict(bucket_counts)
            for dimension, bucket_counts in snapshot.distributions.items()
        },
    }


def _digest_summary(digest: LearningDigest) -> dict[str, Any]:
    return {
        "evaluation_count": digest.evaluation_count,
        "sample_count": digest.sample_count,
        "scoreable": digest.scoreable,
        "true_positive": digest.true_positive,
        "true_negative": digest.true_negative,
        "false_positive": digest.false_positive,
        "false_negative": digest.false_negative,
        "accuracy": digest.accuracy,
        "precision": digest.precision,
        "recall": digest.recall,
        "false_positive_rate": digest.false_positive_rate,
        "false_negative_rate": digest.false_negative_rate,
        "verdict_counts": dict(digest.verdict_counts),
    }


def _confidence_summary(confidence: LearningConfidence) -> dict[str, Any]:
    return {
        "count": confidence.count,
        "mean": confidence.mean,
        "std_dev": confidence.std_dev,
        "bias": confidence.bias,
        "calibrated": confidence.calibrated,
    }


def _patterns_summary(
    patterns: dict[str, dict[str, LearningPattern]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        dimension: {
            value: {
                "samples": verdict.samples,
                "scoreable": verdict.scoreable,
                "true_positive": verdict.true_positive,
                "true_negative": verdict.true_negative,
                "false_positive": verdict.false_positive,
                "false_negative": verdict.false_negative,
                "accuracy": verdict.accuracy,
                "precision": verdict.precision,
                "confidence_bias": verdict.confidence_bias,
                "recommendation": verdict.recommendation,
            }
            for value, verdict in buckets.items()
        }
        for dimension, buckets in patterns.items()
    }


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with microsecond precision."""
    return datetime.now(UTC).isoformat()
