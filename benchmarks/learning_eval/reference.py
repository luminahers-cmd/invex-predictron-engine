"""Independent reference computations for the learning-eval suite.

Every formula here is re-implemented from the *documented math* in
:mod:`predictron_engine.learning.models` — not copied from the engine — so
that the suite cross-checks the engine against an independent derivation of
the same contract.

All arithmetic follows the engine's documented rounding contract: every
published metric is rounded to four decimal places; ECE is computed over
``CALIBRATION_BIN_COUNT = 5`` equal-width bins indexed by
``int(confidence * 5)`` capped at ``4``.
"""

from __future__ import annotations

import math
from statistics import fmean
from typing import Any

from predictron_engine.dataset.evaluation import EvaluationVerdict
from predictron_engine.learning.models import (
    CALIBRATION_BIN_COUNT,
    LEARNING_DIMENSIONS,
    LearningSample,
)

__all__ = [
    "belief_vectors",
    "reference_bias",
    "reference_confusion",
    "reference_ece",
    "reference_rates",
]


def _round4(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def reference_confusion(
    samples: list[LearningSample],
) -> dict[str, int]:
    """Independent ``(scoreable, tp, tn, fp, fn)`` over scoreable samples."""
    result = {"scoreable": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for sample in samples:
        if sample.verdict not in (
            EvaluationVerdict.CORRECT,
            EvaluationVerdict.INCORRECT,
        ):
            continue
        if sample.actual_positive is None:
            continue
        correct = sample.verdict == EvaluationVerdict.CORRECT
        if sample.actual_positive:
            result["tp" if correct else "fn"] += 1
        else:
            result["tn" if correct else "fp"] += 1
        result["scoreable"] += 1
    return result


def reference_rates(
    confusion: dict[str, int],
) -> dict[str, float | None]:
    """Independent accuracy / precision / fp-rate / fn-rate derivation."""
    scoreable = confusion["scoreable"]
    tp = confusion["tp"]
    tn = confusion["tn"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    accuracy: float | None = None
    if scoreable:
        accuracy = (tp + tn) / scoreable
    precision: float | None = None
    if tp + fp:
        precision = tp / (tp + fp)
    fp_rate: float | None = None
    if fp + tn:
        fp_rate = fp / (fp + tn)
    fn_rate: float | None = None
    if fn + tp:
        fn_rate = fn / (fn + tp)
    return {
        "accuracy": _round4(accuracy),
        "precision": _round4(precision),
        "recall": _round4(tp / (tp + fn) if tp + fn else None),
        "false_positive_rate": _round4(fp_rate),
        "false_negative_rate": _round4(fn_rate),
    }


def reference_ece(
    samples: list[LearningSample],
) -> dict[str, Any]:
    """Independent equal-width ECE plus per-bin overconfidence scan."""
    labeled = [s for s in samples if s.actual_positive is not None]
    result: dict[str, Any] = {
        "expected_calibration_error": 0.0,
        "overconfidence_detected": False,
        "overconfident_bins": 0,
        "total_samples": len(labeled),
        "bins": [],
    }
    if not labeled:
        return result

    total = len(labeled)
    widths: list[list[LearningSample]] = [[] for _ in range(CALIBRATION_BIN_COUNT)]
    for sample in labeled:
        index = min(
            CALIBRATION_BIN_COUNT - 1,
            int(sample.confidence * CALIBRATION_BIN_COUNT),
        )
        widths[index].append(sample)

    ece_total = 0.0
    overconfident = 0
    bins: list[dict[str, object]] = []
    for index, cohort in enumerate(widths):
        if not cohort:
            continue
        conf_mean = fmean(sample.confidence for sample in cohort)
        acc = fmean(1.0 if sample.actual_positive else 0.0 for sample in cohort)
        gap = abs(conf_mean - acc)
        ece_total += (len(cohort) / total) * gap
        if conf_mean - acc >= 0.05:
            overconfident += 1
        bins.append(
            {
                "bin": index,
                "samples": len(cohort),
                "confidence_mean": round(conf_mean, 4),
                "accuracy": round(acc, 4),
                "gap": round(gap, 4),
            }
        )
    result["expected_calibration_error"] = round(ece_total, 4)
    result["overconfidence_detected"] = overconfident > 0
    result["overconfident_bins"] = overconfident
    result["bins"] = bins
    return result


def reference_bias(samples: list[LearningSample]) -> float:
    """Independent ``mean(predicted) - rate(actual positive)`` derivation."""
    confidences = [sample.confidence for sample in samples]
    labeled = [s for s in samples if s.actual_positive is not None]
    mean_conf = fmean(confidences) if confidences else None
    base_rate = (
        fmean(1.0 if s.actual_positive else 0.0 for s in labeled)
        if labeled
        else None
    )
    if mean_conf is None:
        return 0.0
    return round((mean_conf or 0.0) - (base_rate or 0.0), 4)


def belief_vectors(
    samples: list[LearningSample],
) -> dict[str, dict[str, int]]:
    """Independent per-dimension bucket distribution derivation."""
    distributions: dict[str, dict[str, int]] = {
        dimension: {} for dimension in LEARNING_DIMENSIONS
    }
    for sample in samples:
        for dimension in LEARNING_DIMENSIONS:
            value = sample.resolved.get(dimension, "unknown")
            distributions[dimension][value] = (
                distributions[dimension].get(value, 0) + 1
            )
    for buckets in distributions.values():
        for key in list(buckets):
            if buckets[key] == 0:
                del buckets[key]
    return {
        dimension: dict(sorted(buckets.items()))
        for dimension, buckets in distributions.items()
    }


def expectation(value: float | None, tolerance: float = 1e-9) -> bool:
    return value is not None and math.isclose(value, 0.0, rel_tol=0, abs_tol=tolerance)
