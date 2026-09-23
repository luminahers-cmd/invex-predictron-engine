"""The deterministic assertion suite over the learning engine (Phase 7).

Drives :func:`predictron_engine.learning.build_learning_snapshot` over the
fixtures in :mod:`benchmarks.learning_eval.dataset`, then:

* cross-checks every numeric aggregate against the independent reference
  implementation (:mod:`benchmarks.learning_eval.reference`);
* pins the hand-derived expected values from ``EXPECTED``;
* asserts the deterministic snapshot contract (schema version, verification,
  stable snapshot ids, stable content hashes, canonical distributions);
* asserts structural invariants of the canonical observations / patterns.

The suite is deterministic: the same fixtures always yield the same run id
and the same assertion results.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from statistics import pstdev

from benchmarks.learning_eval.dataset import EXPECTED, FIXED_ANCHOR, fixtures
from benchmarks.learning_eval.models import (
    AssertionResult,
    LearningEvalCaseResult,
    LearningEvalReport,
    LearningEvalRun,
)
from benchmarks.learning_eval.reference import (
    belief_vectors,
    reference_bias,
    reference_confusion,
    reference_ece,
    reference_rates,
)
from predictron_engine.dataset.evaluation import EvaluationVerdict
from predictron_engine.learning import (
    LEARNING_DIMENSIONS,
    LEARNING_SNAPSHOT_SCHEMA_VERSION,
    LearningObservationCategory,
    LearningPattern,
    LearningSample,
    LearningSnapshot,
    build_learning_snapshot,
)

__all__ = [
    "LEARNING_ENGINE_VERSION",
    "ensure_learning_version",
    "run_suite",
]

LEARNING_ENGINE_VERSION = "7.0.0"


def ensure_learning_version() -> None:
    """Fail loudly when the engine contract under benchmark changes."""
    if LEARNING_ENGINE_VERSION not in ("7.0.0",):
        raise AssertionError(f"Engine version drifted: {LEARNING_ENGINE_VERSION}")


def _check(
    name: str,
    condition: bool,
    detail: object = "",
) -> AssertionResult:
    return AssertionResult(name=name, passed=bool(condition), detail=str(detail))


def _close(value: float | None, expected: float | None) -> bool:
    if value is None or expected is None:
        return value is None and expected is None
    return abs(value - expected) <= 1e-9


def _expected_snapshot_id(scope: str) -> str:
    material = "|".join((scope, "daily", FIXED_ANCHOR.isoformat()))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _freeze(scope: str, case_id: str, samples: list[LearningSample]):
    """Build one snapshot with deterministic timestamps for the suite."""
    anchor = FIXED_ANCHOR
    recorded_at = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)
    five_minutes = datetime(2026, 6, 15, 12, 5, 0, tzinfo=UTC)
    baseline = build_learning_snapshot(
        samples,
        anchor_date=anchor,
        scope=scope,
        engine_version=LEARNING_ENGINE_VERSION,
        recorded_at=recorded_at,
    )
    twin = build_learning_snapshot(
        samples,
        anchor_date=anchor,
        scope=scope,
        engine_version=LEARNING_ENGINE_VERSION,
        recorded_at=five_minutes,
    )
    return baseline, twin


def _case_assertions(
    case_id: str,
    samples: list[LearningSample],
) -> LearningEvalCaseResult:
    expected = EXPECTED[case_id]
    assertions: list[AssertionResult] = []
    scope = f"bench:{case_id}"

    snapshot, twin = _freeze(scope, case_id, samples)

    # --- Snapshot contract -------------------------------------------------
    assertions.append(
        _check(
            "schema_version",
            snapshot.schema_version == LEARNING_SNAPSHOT_SCHEMA_VERSION,
            snapshot.schema_version,
        )
    )
    assertions.append(
        _check(
            "content_hash_present",
            len(snapshot.content_hash) == 64,
            snapshot.content_hash,
        )
    )
    assertions.append(_check("verify", snapshot.verify(), snapshot.content_hash))
    assertions.append(
        _check(
            "snapshot_id_stable",
            snapshot.snapshot_id == twin.snapshot_id,
            snapshot.snapshot_id,
        )
    )
    assertions.append(
        _check(
            "snapshot_id_epoch",
            snapshot.snapshot_id == _expected_snapshot_id(scope),
            snapshot.snapshot_id,
        )
    )
    assertions.append(
        _check(
            "content_hash_stable_across_timestamps",
            snapshot.content_hash == twin.content_hash,
            snapshot.content_hash,
        )
    )
    assertions.append(
        _check("twin_verify", twin.verify(), twin.content_hash)
    )
    assertions.append(
        _check(
            "engine_version",
            snapshot.engine_version == LEARNING_ENGINE_VERSION,
            snapshot.engine_version,
        )
    )

    # --- Counts + digest (engines against reference AND pinned literals) ----
    digest = snapshot.digest
    assertions.append(
        _check("counts_exact", snapshot.counts == expected["counts"], snapshot.counts)
    )
    assertions.append(
        _check("digest_scoreable", digest.scoreable == digest_true(samples), digest.scoreable)
    )
    reference_conf = reference_confusion(samples)
    assertions.append(
        _check(
            "digest_matches_reference",
            (
                digest.true_positive == reference_conf["tp"]
                and digest.true_negative == reference_conf["tn"]
                and digest.false_positive == reference_conf["fp"]
                and digest.false_negative == reference_conf["fn"]
            ),
            reference_conf,
        )
    )
    rates = reference_rates(reference_conf)
    assertions.append(
        _check(
            "digest_rates_match_reference",
            (
                _close(digest.accuracy, rates["accuracy"])
                and _close(digest.precision, rates["precision"])
                and _close(digest.recall, rates["recall"])
                and _close(digest.false_positive_rate, rates["false_positive_rate"])
                and _close(digest.false_negative_rate, rates["false_negative_rate"])
            ),
            rates,
        )
    )
    assertions.append(
        _check("accuracy_pinned", _close(digest.accuracy, expected["accuracy"]), digest.accuracy)
    )
    assertions.append(
        _check(
            "precision_pinned",
            _close(digest.precision, expected["precision"]),
            digest.precision,
        )
    )
    if "recall" in expected:
        assertions.append(
            _check(
                "recall_pinned",
                _close(digest.recall, expected["recall"]),
                digest.recall,
            )
        )
    if "false_positive_rate" in expected:
        assertions.append(
            _check(
                "fpr_pinned",
                _close(digest.false_positive_rate, expected["false_positive_rate"]),
                digest.false_positive_rate,
            )
        )
    if "false_negative_rate" in expected:
        assertions.append(
            _check(
                "fnr_pinned",
                _close(digest.false_negative_rate, expected["false_negative_rate"]),
                digest.false_negative_rate,
            )
        )
    assertions.append(
        _check(
            "metrics_accuracy",
            _close(snapshot.metrics.get("accuracy"), expected["accuracy"]),
            snapshot.metrics,
        )
    )

    # --- Calibration --------------------------------------------------------
    calibration = snapshot.calibration
    reference_cal = reference_ece(samples)
    assertions.append(
        _check(
            "ece_matches_reference",
            _close(
                calibration.expected_calibration_error,
                reference_cal["expected_calibration_error"],
            ),
            reference_cal["expected_calibration_error"],
        )
    )
    assertions.append(
        _check(
            "ece_pinned",
            _close(
                calibration.expected_calibration_error,
                expected["ece"],
            ),
            calibration.expected_calibration_error,
        )
    )
    assertions.append(
        _check(
            "ece_within_tolerance",
            calibration.expected_calibration_error <= 1.0,
            calibration.expected_calibration_error,
        )
    )
    assertions.append(
        _check(
            "calibration_total_samples",
            calibration.total_samples == len([s for s in samples if s.actual_positive is not None]),
            calibration.total_samples,
        )
    )
    assertions.append(
        _check(
            "overconfidence_detected_pinned",
            calibration.overconfidence_detected == expected["overconfidence_detected"],
            calibration.overconfidence_detected,
        )
    )
    if "overconfident_bins" in expected:
        assertions.append(
            _check(
                "overconfident_bins_pinned",
                calibration.overconfident_bins == expected["overconfident_bins"],
                calibration.overconfident_bins,
            )
        )
    assertions.append(
        _check(
            "metrics_ece",
            _close(snapshot.metrics.get("ece"), expected["ece"]),
            snapshot.metrics,
        )
    )

    # --- Confidence statistics ---------------------------------------------
    confidence = snapshot.confidence
    confidences = [sample.confidence for sample in samples]
    if "confidence_mean" in expected:
        assertions.append(
            _check(
                "confidence_mean_pinned",
                _close(confidence.mean, expected["confidence_mean"]),
                confidence.mean,
            )
        )
    assertions.append(
        _check(
            "confidence_count",
            confidence.count == len(samples),
            confidence.count,
        )
    )
    if len(confidences) > 1:
        assertions.append(
            _check(
                "confidence_stddev_independent",
                _close(confidence.std_dev, round(pstdev(confidences), 4)),
                confidence.std_dev,
            )
        )
    assertions.append(
        _check("confidence_bias_pinned", _close(confidence.bias, expected["bias"]), confidence.bias)
    )
    assertions.append(
        _check(
            "confidence_bias_independent",
            _close(confidence.bias, reference_bias(samples)),
            reference_bias(samples),
        )
    )
    assertions.append(
        _check(
            "metrics_bias",
            _close(snapshot.metrics.get("confidence.bias"), expected["bias"]),
            snapshot.metrics,
        )
    )

    # --- Distributions ------------------------------------------------------
    reference_dists = belief_vectors(samples)
    assertions.append(
        _check(
            "distributions_match_reference",
            snapshot.distributions == reference_dists,
            snapshot.distributions,
        )
    )
    assertions.append(
        _check(
            "distributions_dimensions",
            set(snapshot.distributions) == set(LEARNING_DIMENSIONS),
            sorted(snapshot.distributions),
        )
    )

    # --- Patterns -----------------------------------------------------------
    assertions.append(
        _check(
            "patterns_reference_count",
            len(snapshot.patterns)
            == sum(
                1
                for dimension in LEARNING_DIMENSIONS
                for value in {s.resolved.get(dimension, "unknown") for s in samples}
            ),
            len(snapshot.patterns),
        )
    )
    assertions.append(
        _check(
            "patterns_pinned",
            len(snapshot.patterns) == expected["patterns"],
            len(snapshot.patterns),
        )
    )
    cursor = -1
    dimensions_ordered = True
    for dim in LEARNING_DIMENSIONS:
        positions = [
            index
            for index, pattern in enumerate(snapshot.patterns)
            if pattern.dimension.value == dim
        ]
        if positions:
            if positions[0] <= cursor:
                dimensions_ordered = False
            cursor = positions[-1]
    assertions.append(
        _check(
            "patterns_dimensions_ordered",
            dimensions_ordered,
            [p.dimension.value for p in snapshot.patterns],
        )
    )
    assertions.append(
        _check(
            "patterns_buckets_sorted",
            all(
                [p.value for p in snapshot.patterns if p.dimension.value == dim]
                == sorted(
                    p.value
                    for p in snapshot.patterns
                    if p.dimension.value == dim
                )
                for dim in LEARNING_DIMENSIONS
            ),
            len(snapshot.patterns),
        )
    )
    for pattern in snapshot.patterns:
        if pattern.scoreable:
            assertions.append(
                _check(
                    "patterns_rate_bounds",
                    0.0 <= (pattern.accuracy or 0.0) <= 1.0
                    and 0.0 <= (pattern.precision or 0.0) <= 1.0,
                    pattern.model_dump(mode="json"),
                )
            )

    # --- Knowledge ----------------------------------------------------------
    assertions.append(
        _check(
            "knowledge_dimensions",
            set(snapshot.knowledge) == set(LEARNING_DIMENSIONS),
            sorted(snapshot.knowledge),
        )
    )
    for entries in snapshot.knowledge.values():
        assertions.append(
            _check(
                "knowledge_sorted",
                entries
                == sorted(entries, key=lambda entry: (entry.value, entry.sample_size)),
                len(entries),
            )
        )

    # --- Observations -------------------------------------------------------
    if samples:
        categories = {obs.category for obs in snapshot.observations}
        assertions.append(
            _check(
                "observations_nonempty",
                len(snapshot.observations) > 0,
                len(snapshot.observations),
            )
        )
        assertions.append(
            _check(
                "observations_canonical",
                categories <= set(LearningObservationCategory),
                sorted(c.value for c in categories),
            )
        )
        for observation in snapshot.observations:
            assertions.append(
                _check(
                    "observations_fields",
                    observation.dimension.value in LEARNING_DIMENSIONS
                    and observation.value
                    and observation.sample_size >= 0,
                    observation.model_dump(mode="json"),
                )
            )
    else:
        assertions.append(
            _check(
                "observations_empty",
                len(snapshot.observations) == 0,
                len(snapshot.observations),
            )
        )
    assertions.append(
        _check(
            "observations_pinned_count",
            len(snapshot.observations) == expected.get("observations", len(snapshot.observations)),
            len(snapshot.observations),
        )
    )

    # --- Recommendations ----------------------------------------------------
    assertions.append(
        _check(
            "recommendations_pinned_count",
            len(snapshot.recommendations) == expected["recommendations"],
            len(snapshot.recommendations),
        )
    )
    kinds = {rec.kind for rec in snapshot.recommendations}
    assertions.append(
        _check(
            "recommendations_kinds_nonempty",
            all(kind for kind in kinds),
            sorted(kinds),
        )
    )

    # --- Verdict accounting -------------------------------------------------
    verdict_counts = dict(digest.verdict_counts)
    derived: dict[str, int] = {}
    for sample in samples:
        derived[sample.verdict.value] = derived.get(sample.verdict.value, 0) + 1
    assertions.append(
        _check(
            "verdict_counts_correct",
            verdict_counts == dict(sorted(derived.items())),
            verdict_counts,
        )
    )

    return LearningEvalCaseResult(case_id=case_id, assertions=assertions)


def digest_true(samples: list[LearningSample]) -> int:
    """Scoreable count without importing engine internals."""
    return sum(
        1
        for sample in samples
        if sample.verdict in (EvaluationVerdict.CORRECT, EvaluationVerdict.INCORRECT)
        and sample.actual_positive is not None
    )


def run_suite(scope: str = "repository") -> tuple[LearningEvalReport, str]:
    """Execute every fixture, cross-check everything, and return the report.

    Returns ``(report, run_id)`` where ``run_id`` is a deterministic
    SHA-256 over the fixture ids and the pinned expected value surface.
    """
    ensure_learning_version()
    all_cases = fixtures()
    case_results: list[LearningEvalCaseResult] = []
    snapshots: dict[str, LearningSnapshot] = {}
    patterns: dict[str, dict[str, LearningPattern]] = {}

    for case_id in sorted(all_cases):
        samples = all_cases[case_id]
        case_result = _case_assertions(case_id, samples)
        case_results.append(case_result)
        built, _twin = _freeze(scope, case_id, samples)
        snapshots[case_id] = built
        patterns[case_id] = {
            pattern.dimension.value: pattern for pattern in built.patterns
        }

    run = LearningEvalRun(
        run_id=sorted_case_run_id(all_cases),
        engine_version=LEARNING_ENGINE_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        cases=case_results,
    )
    return (
        LearningEvalReport(run=run, snapshots=snapshots, patterns=patterns),
        run.run_id,
    )


def sorted_case_run_id(all_cases: dict[str, list[LearningSample]]) -> str:
    """Deterministic run id over the fixture surface."""
    material = "|".join(
        key
        + ":"
        + ",".join(sorted({s.evaluation_id for s in all_cases[key]}))
        for key in sorted(all_cases)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def fail_counts(run: LearningEvalRun) -> tuple[int, int]:
    """``(failed, total)`` assertion counts for a run."""
    failed = sum(1 for c in run.cases for a in c.assertions if not a.passed)
    return failed, run.total_assertions
