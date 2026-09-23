"""Deterministic evaluated-sample fixtures for the learning-eval suite.

No randomness anywhere: every fixture is hand-built from a fixed attribute
template so the *expected* values of the case are known exactly and can be
pinned in ``EXPECTED``.  Each fixture exercises a distinct contract of the
learning engine:

* ``empty``                — the degenerate population (zero samples).
* ``perfect``              — perfectly calibrated, perfectly accurate.
* ``mixed_bias``           — high accuracy with confident positives.
* ``imperfect_balanced``   — finite fp/fn with high ECE and fp/fn rates.
* ``overconfident``        — overconfident bins, high fn rate, mid accuracy.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from predictron_engine.dataset.evaluation import EvaluationVerdict
from predictron_engine.learning.models import LearningSample

__all__ = [
    "EXPECTED",
    "FIXED_ANCHOR",
    "fixtures",
]

FIXED_ANCHOR = date(2026, 6, 15)

# Canonical attribute template every sample resolves to (all six dimensions).
def _template() -> dict[str, str]:
    return {
        "sector": "ai",
        "stage": "seed",
        "country": "us",
        "technology": "ml",
        "business_model": "saas",
        "founder": "team",
    }


def _sample(
    evaluation_id: str,
    confidence: float,
    verdict: EvaluationVerdict,
    actual_positive: bool | None,
) -> LearningSample:
    return LearningSample(
        evaluation_id=evaluation_id,
        confidence=confidence,
        verdict=verdict,
        actual_positive=actual_positive,
        resolved=_template(),
    )


def _empty() -> list[LearningSample]:
    return []


def _perfect() -> list[LearningSample]:
    samples = []
    for index in range(5):
        samples.append(
            _sample(
                f"perfect-pos-{index}",
                1.0,
                EvaluationVerdict.CORRECT,
                True,
            )
        )
        samples.append(
            _sample(
                f"perfect-neg-{index}",
                0.0,
                EvaluationVerdict.CORRECT,
                False,
            )
        )
    return samples


def _mixed_bias() -> list[LearningSample]:
    samples = []
    for index in range(5):
        samples.append(
            _sample(f"mixed-pos-{index}", 0.9, EvaluationVerdict.CORRECT, True)
        )
        samples.append(
            _sample(f"mixed-neg-{index}", 0.6, EvaluationVerdict.CORRECT, False)
        )
    return samples


def _imperfect_balanced() -> list[LearningSample]:
    samples = []
    for index in range(2):
        samples.append(
            _sample(f"im-tp-{index}", 0.7, EvaluationVerdict.CORRECT, True)
        )
    for index in range(2):
        samples.append(
            _sample(f"im-tn-{index}", 0.3, EvaluationVerdict.CORRECT, False)
        )
    for index in range(3):
        samples.append(
            _sample(f"im-fp-{index}", 0.8, EvaluationVerdict.INCORRECT, False)
        )
    for index in range(3):
        samples.append(
            _sample(f"im-fn-{index}", 0.4, EvaluationVerdict.INCORRECT, True)
        )
    return samples


def _overconfident() -> list[LearningSample]:
    samples = []
    for index in range(5):
        samples.append(
            _sample(f"oc-fn-{index}", 0.9, EvaluationVerdict.INCORRECT, True)
        )
    for index in range(5):
        samples.append(
            _sample(f"oc-tn-{index}", 0.2, EvaluationVerdict.CORRECT, False)
        )
    return samples


def fixtures() -> dict[str, list[LearningSample]]:
    """All deterministic fixtures keyed by their stable case id."""
    return {
        "empty": _empty(),
        "perfect": _perfect(),
        "mixed_bias": _mixed_bias(),
        "imperfect_balanced": _imperfect_balanced(),
        "overconfident": _overconfident(),
    }


# ---------------------------------------------------------------------------
# Hand-derived expected values (pinned literals, not recomputed here).
# ---------------------------------------------------------------------------

EXPECTED: dict[str, dict[str, Any]] = {
    "empty": {
        "counts": {"evaluations": 0, "samples": 0, "scoreable": 0},
        "accuracy": None,
        "precision": None,
        "ece": 0.0,
        "bias": 0.0,
        "overconfidence_detected": False,
        "patterns": 0,
        "observations": 0,
        "recommendations": 0,
    },
    "perfect": {
        "counts": {"evaluations": 10, "samples": 10, "scoreable": 10},
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
        "ece": 0.0,
        "bias": 0.0,
        "confidence_mean": 0.5,
        "overconfidence_detected": False,
        "patterns": 6,  # six dimensions, one shared value each
        "observations": 6,  # one STRONG_ACCURACY observation per dimension
        "recommendations": 1,  # maintain_confidence_weighting (acc 1.0 >= 0.6)
    },
    "mixed_bias": {
        "counts": {"evaluations": 10, "samples": 10, "scoreable": 10},
        "accuracy": 1.0,
        "precision": 1.0,
        "ece": 0.35,
        "bias": 0.25,
        "confidence_mean": 0.75,
        "overconfidence_detected": True,
        "overconfident_bins": 1,
        "patterns": 6,
        "observations": 12,  # strong_accuracy + overconfidence, per dimension
        "recommendations": 2,  # recalibrate_confidence (ece > 0.1) + maintain (acc >= 0.6)
    },
    "imperfect_balanced": {
        "counts": {"evaluations": 10, "samples": 10, "scoreable": 10},
        "accuracy": 0.4,
        "precision": 0.4,
        "recall": 0.4,
        "false_positive_rate": 0.6,
        "false_negative_rate": 0.6,
        "ece": 0.54,
        "bias": 0.06,
        "confidence_mean": 0.56,
        "overconfidence_detected": True,
        "overconfident_bins": 2,  # bin1 (0.3 vs 0.0) and bin4 (0.8 vs 0.0)
        "patterns": 6,
        "observations": 12,  # high_fp + overconfidence, per dimension
        "recommendations": 1,  # recalibrate_confidence only
    },
    "overconfident": {
        "counts": {"evaluations": 10, "samples": 10, "scoreable": 10},
        "accuracy": 0.5,
        "precision": None,  # no true positives at all
        "recall": 0.0,
        "false_positive_rate": 0.0,
        "false_negative_rate": 1.0,
        "ece": 0.15,
        "bias": 0.05,
        "confidence_mean": 0.55,
        "overconfidence_detected": True,
        "overconfident_bins": 1,
        "patterns": 6,
        "observations": 12,  # high_fn + overconfidence, per dimension
        "recommendations": 1,  # recalibrate_confidence only
    },
}
