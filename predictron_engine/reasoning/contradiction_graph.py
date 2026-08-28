"""Contradiction Graph — explicit modeling of supporting vs. conflicting evidence (Sprint 8).

Explicitly models the tension between supporting and conflicting evidence,
scores contradiction intensity, and surfaces the dominant conflict driving
uncertainty.

Key principles:
  - Every contradiction is traceable to specific observations
  - Contradiction intensity is deterministic (0.0-1.0)
  - The dominant conflict is surfaced as the highest-intensity contradiction
  - Supporting evidence is modeled as reinforcement, not just absence of conflict
  - No external dependencies

Public API:
  - :func:`build_contradiction_graph` — build the graph from observations
  - :class:`ContradictionGraph` — container for all contradiction analysis
  - :class:`ContradictionEdge` — one supporting or conflicting relationship
  - :class:`DominantConflict` — the primary driver of uncertainty
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.models.report import Observation


@dataclass(frozen=True)
class ContradictionEdge:
    """One edge in the contradiction graph.

    Connects two observations that either support or conflict with each
    other. The edge carries the relationship type, intensity, and the
    dimension it belongs to.
    """

    observation_a_idx: int
    observation_b_idx: int
    dimension: str
    relationship: str  # "supporting" | "conflicting" | "neutral"
    intensity: float  # 0.0-1.0 (how strong the relationship is)
    reason: str

    @property
    def is_conflict(self) -> bool:
        return self.relationship == "conflicting"

    @property
    def is_support(self) -> bool:
        return self.relationship == "supporting"


@dataclass(frozen=True)
class DominantConflict:
    """The primary conflict driving uncertainty in the analysis.

    Identifies the pair of observations with the highest contradiction
    intensity, along with an explanation of why they conflict.
    """

    observation_a_statement: str
    observation_b_statement: str
    dimension: str
    intensity: float
    explanation: str


@dataclass
class ContradictionGraph:
    """Full contradiction analysis of a set of observations.

    Contains all edges, per-dimension summary statistics, and the
    dominant conflict. All fields are computed deterministically.
    """

    edges: tuple[ContradictionEdge, ...] = ()
    supporting_count: int = 0
    conflicting_count: int = 0
    neutral_count: int = 0
    total_intensity: float = 0.0
    contradiction_intensity: float = 0.0
    dominant_conflict: DominantConflict | None = None
    per_dimension: dict[str, DimensionContradictionSummary] = field(
        default_factory=dict,
    )

    @property
    def has_contradictions(self) -> bool:
        return self.conflicting_count > 0

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def to_dict(self) -> dict[str, object]:
        """Serialize for reporting and benchmarking."""
        return {
            "total_edges": self.edge_count,
            "supporting_edges": self.supporting_count,
            "conflicting_edges": self.conflicting_count,
            "neutral_edges": self.neutral_count,
            "contradiction_intensity": self.contradiction_intensity,
            "dominant_conflict": (
                {
                    "statement_a": self.dominant_conflict.observation_a_statement,
                    "statement_b": self.dominant_conflict.observation_b_statement,
                    "dimension": self.dominant_conflict.dimension,
                    "intensity": self.dominant_conflict.intensity,
                    "explanation": self.dominant_conflict.explanation,
                }
                if self.dominant_conflict
                else None
            ),
            "per_dimension": {
                k: v.to_dict() for k, v in self.per_dimension.items()
            },
        }


@dataclass(frozen=True)
class DimensionContradictionSummary:
    """Contradiction summary for a single analysis dimension."""

    dimension: str
    supporting_count: int = 0
    conflicting_count: int = 0
    intensity: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "supporting": self.supporting_count,
            "conflicting": self.conflicting_count,
            "intensity": self.intensity,
        }


# ---------------------------------------------------------------------------
# Deterministic relationship detection
# ---------------------------------------------------------------------------


def _confidence_distance(a: Observation, b: Observation) -> float:
    """Deterministic distance between two observation confidences."""
    return abs(a.confidence - b.confidence)


def _importance_weight(a: Observation, b: Observation) -> float:
    """Combined importance weight for a pair of observations."""
    return (a.importance + b.importance) / 2.0


def _classify_relationship(
    a: Observation,
    b: Observation,
) -> str:
    """Classify the relationship between two observations.

    Two observations conflict when:
      - They belong to the same dimension
      - They have different categories (e.g. "strength" vs "risk")
      - Or one has high confidence and the other has low confidence
        with opposing importance

    Two observations support when:
      - Same dimension, same category, or similar confidence/importance

    Otherwise: neutral.
    """
    if a.dimension != b.dimension:
        return "neutral"

    same_category = a.category == b.category
    confidence_aligned = abs(a.confidence - b.confidence) < 0.25
    importance_aligned = abs(a.importance - b.importance) < 0.25

    # Conflict indicators
    category_conflict = not same_category and _categories_tension(a.category, b.category)
    confidence_conflict = not confidence_aligned and (
        (a.confidence > 0.7 and b.confidence < 0.3)
        or (b.confidence > 0.7 and a.confidence < 0.3)
    )
    importance_conflict = not importance_aligned and (
        (a.importance > 0.7 and b.importance < 0.3)
        or (b.importance > 0.7 and a.importance < 0.3)
    )

    conflict_signals = sum([category_conflict, confidence_conflict, importance_conflict])
    if conflict_signals >= 2:
        return "conflicting"

    # Support indicators
    if same_category and confidence_aligned:
        return "supporting"
    if confidence_aligned and importance_aligned:
        return "supporting"

    return "neutral"


def _categories_tension(cat_a: str, cat_b: str) -> bool:
    """Determine if two categories inherently create tension."""
    conflicting_pairs = {
        ("strength", "risk"),
        ("strength", "signal_conflict"),
        ("opportunity", "risk"),
        ("growth", "decline"),
        ("positive", "negative"),
        ("advantage", "disadvantage"),
    }
    pair = (cat_a.lower(), cat_b.lower())
    reverse = (cat_b.lower(), cat_a.lower())
    return pair in conflicting_pairs or reverse in conflicting_pairs


def _compute_edge_intensity(
    a: Observation,
    b: Observation,
    relationship: str,
) -> float:
    """Compute deterministic intensity for a graph edge.

    Intensity is higher when:
      - Both observations have high importance
      - Confidence divergence is large (for conflicts)
      - Confidence convergence is large (for supports)
    """
    importance = _importance_weight(a, b)
    confidence_diff = _confidence_distance(a, b)

    if relationship == "conflicting":
        intensity = importance * 0.6 + confidence_diff * 0.4
    elif relationship == "supporting":
        alignment = 1.0 - confidence_diff
        intensity = importance * 0.6 + alignment * 0.4
    else:
        intensity = importance * 0.3

    return round(max(0.0, min(1.0, intensity)), 4)


def _build_conflict_reason(a: Observation, b: Observation) -> str:
    """Build a deterministic explanation for a conflict."""
    if a.confidence != b.confidence:
        higher = a if a.confidence > b.confidence else b
        lower = b if a.confidence > b.confidence else a
        return (
            f"Observation '{higher.statement[:60]}' has higher confidence "
            f"({higher.confidence:.2f}) than '{lower.statement[:60]}' "
            f"({lower.confidence:.2f}) in dimension '{a.dimension}'"
        )
    if a.category != b.category:
        return (
            f"Conflicting categories in '{a.dimension}': "
            f"'{a.category}' vs '{b.category}'"
        )
    return (
        f"Observations in '{a.dimension}' with different profiles "
        f"create tension"
    )


def _build_support_reason(a: Observation, b: Observation) -> str:
    """Build a deterministic explanation for a support relationship."""
    if a.category == b.category:
        return (
            f"Mutual support in '{a.dimension}' ({a.category}): "
            f"consistent confidence ({a.confidence:.2f}, {b.confidence:.2f})"
        )
    return (
        f"Aligned observations in '{a.dimension}': "
        f"complementary categories ({a.category}, {b.category})"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_contradiction_graph(
    observations: list[Observation],
) -> ContradictionGraph:
    """Build the contradiction graph from a set of observations.

    All computation is deterministic. Identical observations always
    produce identical graphs.

    Parameters
    ----------
    observations:
        Reasoning layer observations (ordered list).
    """
    if len(observations) < 2:
        return ContradictionGraph()

    edges: list[ContradictionEdge] = []
    supporting = 0
    conflicting = 0
    neutral = 0
    total_intensity = 0.0

    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            a = observations[i]
            b = observations[j]

            relationship = _classify_relationship(a, b)
            intensity = _compute_edge_intensity(a, b, relationship)

            if relationship == "conflicting":
                reason = _build_conflict_reason(a, b)
            elif relationship == "supporting":
                reason = _build_support_reason(a, b)
            else:
                reason = (
                    f"No direct relationship between observations in '{a.dimension}' "
                    f"vs '{b.dimension}'"
                )

            edges.append(
                ContradictionEdge(
                    observation_a_idx=i,
                    observation_b_idx=j,
                    dimension=(
                        a.dimension
                        if a.dimension == b.dimension
                        else f"{a.dimension}:{b.dimension}"
                    ),
                    relationship=relationship,
                    intensity=intensity,
                    reason=reason,
                )
            )

            if relationship == "supporting":
                supporting += 1
            elif relationship == "conflicting":
                conflicting += 1
            else:
                neutral += 1
            total_intensity += intensity

    contradiction_intensity = _compute_overall_intensity(
        conflicting, len(edges), edges,
    )
    dominant = _find_dominant_conflict(edges, observations)
    per_dim = _compute_per_dimension(edges, observations)

    return ContradictionGraph(
        edges=tuple(edges),
        supporting_count=supporting,
        conflicting_count=conflicting,
        neutral_count=neutral,
        total_intensity=round(total_intensity, 4),
        contradiction_intensity=contradiction_intensity,
        dominant_conflict=dominant,
        per_dimension=per_dim,
    )


def _compute_overall_intensity(
    conflict_count: int,
    total_edges: int,
    edges: list[ContradictionEdge],
) -> float:
    """Compute overall contradiction intensity (0.0-1.0)."""
    if total_edges == 0:
        return 0.0
    conflict_ratio = conflict_count / total_edges
    conflict_intensities = [e.intensity for e in edges if e.is_conflict]
    mean_conflict_intensity = (
        sum(conflict_intensities) / len(conflict_intensities)
        if conflict_intensities
        else 0.0
    )
    intensity = conflict_ratio * 0.5 + mean_conflict_intensity * 0.5
    return round(max(0.0, min(1.0, intensity)), 4)


def _find_dominant_conflict(
    edges: list[ContradictionEdge],
    observations: list[Observation],
) -> DominantConflict | None:
    """Find the single highest-intensity conflicting edge."""
    conflict_edges = [e for e in edges if e.is_conflict]
    if not conflict_edges:
        return None

    dominant = max(conflict_edges, key=lambda e: e.intensity)
    obs_a = observations[dominant.observation_a_idx]
    obs_b = observations[dominant.observation_b_idx]

    return DominantConflict(
        observation_a_statement=obs_a.statement,
        observation_b_statement=obs_b.statement,
        dimension=dominant.dimension,
        intensity=dominant.intensity,
        explanation=dominant.reason,
    )


def _compute_per_dimension(
    edges: list[ContradictionEdge],
    observations: list[Observation],
) -> dict[str, DimensionContradictionSummary]:
    """Compute per-dimension contradiction summaries."""
    dim_edges: dict[str, list[ContradictionEdge]] = {}
    for edge in edges:
        dim_edges.setdefault(edge.dimension, []).append(edge)

    summaries: dict[str, DimensionContradictionSummary] = {}
    for dim, dim_edge_list in dim_edges.items():
        s = sum(1 for e in dim_edge_list if e.is_support)
        c = sum(1 for e in dim_edge_list if e.is_conflict)
        total = len(dim_edge_list)
        intensity = 0.0
        if total > 0:
            conflict_intensities = [e.intensity for e in dim_edge_list if e.is_conflict]
            mean_ci = (
                sum(conflict_intensities) / len(conflict_intensities)
                if conflict_intensities
                else 0.0
            )
            intensity = round(c / total * 0.5 + mean_ci * 0.5, 4)
        summaries[dim] = DimensionContradictionSummary(
            dimension=dim,
            supporting_count=s,
            conflicting_count=c,
            intensity=intensity,
        )

    return summaries
