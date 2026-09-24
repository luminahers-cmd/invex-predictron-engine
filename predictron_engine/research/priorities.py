"""Deterministic priority engine — Phase 8 Sprint 1.

Computes a priority score for every research task.  The score combines:

* **Topic importance** — base importance from the taxonomy.
* **Missing evidence** — how much evidence is absent for the topic.
* **Prediction impact** — how strongly the topic moves the prediction.
* **Freshness requirement** — how fast evidence decays, scaled by the
  prediction horizon and the amount of missing evidence.
* **Dependency order** — topics that unlock other research rank higher.
* **Source availability** — whether a source is already available.

All weights are fixed module constants and every input is pure, so the
same input always produces the same plan.  No randomness.
"""

from __future__ import annotations

from predictron_engine.research.models import (
    EvidenceStatus,
    ResearchPriority,
    ResearchTopic,
)

IMPORTANCE_WEIGHT = 0.30
MISSING_EVIDENCE_WEIGHT = 0.25
PREDICTION_IMPACT_WEIGHT = 0.20
FRESHNESS_WEIGHT = 0.10
DEPENDENCY_WEIGHT = 0.10
SOURCE_AVAILABILITY_WEIGHT = 0.05

REFERENCE_HORIZON_DAYS = 365
HORIZON_SCALE_MIN = 0.5
HORIZON_SCALE_MAX = 1.5

EVIDENCE_MISSING_FACTOR: dict[EvidenceStatus, float] = {
    EvidenceStatus.NONE: 1.0,
    EvidenceStatus.PARTIAL: 0.5,
    EvidenceStatus.COMPLETE: 0.0,
}

CRITICAL_SCORE = 75.0
HIGH_SCORE = 55.0
MEDIUM_SCORE = 35.0

# Topics whose evidence is typically reachable through a company website.
_WEBSITE_ADDRESSABLE_TOPICS: frozenset[str] = frozenset(
    {
        "founders",
        "team",
        "product",
        "technology",
        "pricing",
        "business_model",
        "news",
    }
)


def evidence_missing_factor(status: EvidenceStatus) -> float:
    """Return how much evidence is missing for ``status``, in ``[0, 1]``."""
    return EVIDENCE_MISSING_FACTOR[status]


def horizon_freshness_scale(prediction_horizon_days: int) -> float:
    """Scale freshness weight by prediction horizon, clamped deterministically."""
    raw = prediction_horizon_days / REFERENCE_HORIZON_DAYS
    return max(HORIZON_SCALE_MIN, min(HORIZON_SCALE_MAX, raw))


def source_availability_factor(
    topic_id: str,
    website_present: bool,
) -> float:
    """Return the source-availability factor for a topic, in ``[0, 1]``."""
    if not website_present:
        return 0.0
    return 1.0 if topic_id in _WEBSITE_ADDRESSABLE_TOPICS else 0.5


def compute_priority_score(
    topic: ResearchTopic,
    status: EvidenceStatus,
    *,
    dependency_factor: float,
    prediction_horizon_days: int,
    website_present: bool,
) -> float:
    """Compute the deterministic priority score for a topic, in ``[0, 100]``.

    Parameters
    ----------
    topic:
        The :class:`ResearchTopic` being scored.
    status:
        Current evidence status for the topic.
    dependency_factor:
        ``1.0`` when the topic unlocks other planned research, else ``0.0``.
    prediction_horizon_days:
        Prediction window; scales the freshness component.
    website_present:
        Whether a company website is already available.
    """
    missing = evidence_missing_factor(status)
    freshness = (
        topic.freshness_sensitivity
        * missing
        * horizon_freshness_scale(prediction_horizon_days)
    )
    source = source_availability_factor(topic.topic_id, website_present)
    raw = (
        topic.importance * IMPORTANCE_WEIGHT
        + missing * MISSING_EVIDENCE_WEIGHT
        + topic.prediction_impact * PREDICTION_IMPACT_WEIGHT
        + freshness * FRESHNESS_WEIGHT
        + dependency_factor * DEPENDENCY_WEIGHT
        + source * SOURCE_AVAILABILITY_WEIGHT
    )
    return round(raw * 100.0, 4)


def priority_from_score(score: float) -> ResearchPriority:
    """Map a priority score to a deterministic :class:`ResearchPriority`."""
    if score >= CRITICAL_SCORE:
        return ResearchPriority.CRITICAL
    if score >= HIGH_SCORE:
        return ResearchPriority.HIGH
    if score >= MEDIUM_SCORE:
        return ResearchPriority.MEDIUM
    return ResearchPriority.LOW
