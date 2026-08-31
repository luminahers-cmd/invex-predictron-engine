"""Canonical score-aggregation primitives (Sprint P9, H2/H3).

All aggregation formulas in the pipeline — the report *overall score*, the
*investment readiness* composite, and the *decision* composite — are derived
through these shared primitives so that they cannot drift independently.

Prior to this sprint each consumer carried its own copy of an averaging
formula and its own set of magic missing-data defaults (notably ``50.0``,
which treated *missing* evidence as a *neutral/average* score).  That made a
data-poor startup look "middling" rather than "insufficiently evidenced".

Design rules enforced here:

* **One canonical mean.**  ``mean_score`` is the single averaging function
  shared by report aggregation and the decision's score factor.  An empty
  input yields ``0.0`` (no evidence) — it never fabricates a neutral 50.
* **Missing is not average.**  ``weighted_composite`` aggregates only over
  dimensions that actually *have evidence* (``present_keys``).  A dimension
  with no evidence is excluded from both the numerator and the denominator,
  so it cannot drag the composite toward the middle.
* **Deterministic.**  All helpers are pure and total — no randomness, no
  external state, no arbitrary constants beyond the documented scale
  bounds (0-100) and the shared dimension weights.
* **Monotonic.**  Adding or strengthening positive evidence can only raise
  the composite (within the [0, 100] clamp); explicit negative evidence
  continues to pull it down.  Excluding unevidenced dimensions cannot flip
  the sign of an evidenced dimension's contribution.

The single source of truth for dimension weights lives here
(:data:`DIMENSION_WEIGHTS`), so the readiness scoring, gap ranking, and any
other weight consumers reference the same constants.
"""

from __future__ import annotations

from predictron_engine.models.report import ScoreResult

# ---------------------------------------------------------------------------
# Dimension weights — canonical source of truth.
#
# Sum: 0.20 + 0.18 + 0.15 + 0.15 + 0.15 + 0.10 + 0.07 = 1.00
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS: dict[str, float] = {
    "market_opportunity": 0.20,
    "founder_quality": 0.18,
    "product_strength": 0.15,
    "business_model_viability": 0.15,
    "traction_signals": 0.15,
    "competitive_position": 0.10,
    "team_execution": 0.07,
}

DEFAULT_DIMENSION_WEIGHT: float = 0.05
MAX_DIMENSION_WEIGHT: float = max(DIMENSION_WEIGHTS.values())

__all__ = [
    "DIMENSION_WEIGHTS",
    "DEFAULT_DIMENSION_WEIGHT",
    "MAX_DIMENSION_WEIGHT",
    "mean_score",
    "weighted_composite",
]


def mean_score(scores: list[ScoreResult]) -> float:
    """Canonical unweighted mean of dimension scores (0-100).

    Empty input returns ``0.0`` — the absence of scores is reported as
    absent, never invented as a neutral mid-range value.
    """
    if not scores:
        return 0.0
    return sum(s.score for s in scores) / len(scores)


def weighted_composite(
    values: dict[str, float],
    weights: dict[str, float] | None = None,
    *,
    present_keys: set[str] | None = None,
    default_weight: float = DEFAULT_DIMENSION_WEIGHT,
) -> float:
    """Canonical weighted aggregation over *present* dimensions (0-100).

    Args:
        values: Per-dimension numeric contributions (0-100 scale).
        weights: Per-dimension weights; defaults to
            :data:`DIMENSION_WEIGHTS`.  Unknown dimensions receive
            ``default_weight``.
        present_keys: Dimensions that actually carry evidence.  When
            provided, only these keys participate in the average, and the
            weights are renormalized over them so missing dimensions do not
            drag the composite toward a neutral value.
        default_weight: Weight used for a key absent from ``weights``.

    Returns:
        The 0-100 weighted composite, or ``0.0`` when no present dimensions
        are available to aggregate.
    """
    weights = weights if weights is not None else DIMENSION_WEIGHTS

    if present_keys is None:
        present_keys = set(values.keys())

    keys = [k for k in present_keys if k in values]
    if not keys:
        return 0.0

    total_weight = sum(weights.get(k, default_weight) for k in keys)
    if total_weight <= 0.0:
        return 0.0

    composite = 0.0
    for k in keys:
        weight = weights.get(k, default_weight)
        composite += values[k] * (weight / total_weight)
    return composite
