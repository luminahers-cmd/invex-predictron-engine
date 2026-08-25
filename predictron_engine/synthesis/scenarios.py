"""Deterministic alternative scenario generation (Sprint 6C).

Scenarios answer three questions using only already-emitted scalars:
  - What would improve the decision?   (direction="improve")
  - What would worsen the decision?    (direction="worsen")
  - What information could change it?  (direction="information")

All text comes from fixed templates. All numeric projections are
bounded arithmetic derivations of existing decision scalars
(margin_to_next_category, data_quality_modifier, risk_modifier) or
propagated uncertainty values. No LLM generation, no randomness.
"""

from __future__ import annotations

from predictron_engine.decision.models import DecisionConfidence
from predictron_engine.models.report import (
    AlternativeScenario,
    InvestmentDecision,
    InvestmentReadiness,
)

# Plausibility composition (deterministic, documented).
_PLAUSIBILITY_BASE = 0.5

_MAX_SCENARIOS = 6


def _driver_value(confidence: DecisionConfidence | None, name: str) -> float:
    if confidence is None:
        return 0.0
    value = confidence.uncertainty_breakdown.value_of(name)
    return value if value is not None else 0.0


def _round(value: float, digits: int = 2) -> float:
    return round(value + 0.0, digits)


def generate_scenarios(
    decision: InvestmentDecision | None,
    decision_confidence: DecisionConfidence | None,
    readiness: InvestmentReadiness | None,
) -> list[AlternativeScenario]:
    """Generate deterministic scenarios from existing decision scalars."""
    scenarios: list[AlternativeScenario] = []
    if decision is None:
        return scenarios

    composite = decision.composite_score

    # --- Improve scenario 1: recover the data-quality discount ----------
    dq_recovery = 0.0
    if decision.data_quality_modifier > 0.0 and decision.data_quality_modifier < 1.0:
        dq_recovery = min(
            (composite / decision.data_quality_modifier) - composite,
            100.0 - composite,
        )
        plausibility = min(
            1.0,
            _PLAUSIBILITY_BASE
            + _driver_value(decision_confidence, "missing_evidence") / 2.0,
        )
        scenarios.append(
            AlternativeScenario(
                direction="improve",
                title="Improve data completeness",
                condition=(
                    "Raise overall data completeness so the data-quality "
                    f"modifier returns to 1.00 (currently "
                    f"{decision.data_quality_modifier:.2f})"
                ),
                description=(
                    f"With complete data the composite score would rise by "
                    f"up to {dq_recovery:.1f} points, from "
                    f"{composite:.1f} to {min(composite + dq_recovery, 100.0):.1f}."
                ),
                projected_impact=_round(dq_recovery),
                plausibility=_round(plausibility),
            )
        )

    # --- Improve scenario 2: close the margin to the next category ------
    if (
        decision.margin_to_next_category > 0.0
        and decision.next_category_threshold > 0.0
    ):
        margin = decision.margin_to_next_category
        plausibility = max(0.05, min(1.0, 1.0 - margin / 25.0))
        scenarios.append(
            AlternativeScenario(
                direction="improve",
                title="Reach the next decision category",
                condition=(
                    f"Lift the composite score by at least {margin:.1f} points "
                    f"to clear the next category threshold of "
                    f"{decision.next_category_threshold:.1f}"
                ),
                description=(
                    f"A composite score at or above "
                    f"{decision.next_category_threshold:.1f} would move the "
                    f"decision above its current category."
                ),
                projected_impact=_round(margin),
                plausibility=_round(plausibility),
            )
        )

    # --- Improve scenario 3: resolve conflicting signals -----------------
    conflicting_count = readiness.conflicting_count if readiness is not None else 0
    if conflicting_count > 0 and decision.risk_modifier < 1.0:
        recovery = min(
            (composite / decision.risk_modifier) - composite,
            100.0 - composite,
        )
        plausibility = min(
            1.0,
            _PLAUSIBILITY_BASE
            + _driver_value(decision_confidence, "conflicting_evidence") / 2.0,
        )
        scenarios.append(
            AlternativeScenario(
                direction="improve",
                title="Resolve conflicting signals",
                condition=(
                    f"Resolve {conflicting_count} conflicting cross-signal "
                    "relationship(s) so the risk modifier recovers"
                ),
                description=(
                    f"If the risk modifier returned to 1.00 (currently "
                    f"{decision.risk_modifier:.2f}), the composite score "
                    f"would gain up to {recovery:.1f} points."
                ),
                projected_impact=_round(recovery),
                plausibility=_round(plausibility),
            )
        )
    elif conflicting_count == 0 and decision.risk_modifier < 1.0:
        recovery = min(
            (composite / decision.risk_modifier) - composite,
            100.0 - composite,
        )
        scenarios.append(
            AlternativeScenario(
                direction="improve",
                title="Reduce feature-level risks",
                condition=(
                    "Reduce flagged feature risks so the risk modifier "
                    f"recovers toward 1.00 (currently "
                    f"{decision.risk_modifier:.2f})"
                ),
                description=(
                    f"Full risk-modifier recovery would add up to "
                    f"{recovery:.1f} points to the composite score."
                ),
                projected_impact=_round(recovery),
                plausibility=0.3,
            )
        )

    # --- Worsen scenario --------------------------------------------------
    uncertainty = (
        decision_confidence.uncertainty_score
        if decision_confidence is not None
        else 0.0
    )
    gap_count = len(readiness.gaps) if readiness is not None else 0
    scenarios.append(
        AlternativeScenario(
            direction="worsen",
            title="Further signal degradation",
            condition=(
                "Additional conflicting signals, reduced data completeness, "
                "or newly flagged feature risks"
            ),
            description=(
                "Any further degradation would push the composite score and "
                "the decision category downward; current calibrated "
                f"uncertainty is {uncertainty:.2f}"
                + (f" with {gap_count} known information gap(s)." if gap_count else ".")
            ),
            projected_impact=None,
            plausibility=_round(max(uncertainty, 0.05)),
        )
    )

    # --- Information scenarios -------------------------------------------
    info_sources: list[str] = []
    if decision.rationale.information_that_could_change_decision:
        info_sources.extend(
            decision.rationale.information_that_could_change_decision[:3]
        )
    if readiness is not None:
        info_sources.extend(readiness.gaps[:3])

    seen: set[str] = set()
    unique_info: list[str] = []
    for item in info_sources:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique_info.append(item)

    for index, item in enumerate(unique_info):
        plausibility = max(0.1, min(1.0, uncertainty + 0.1 * (len(unique_info) - index)))
        scenarios.append(
            AlternativeScenario(
                direction="information",
                title=f"Information that could change the decision #{index + 1}",
                condition=item,
                description=(
                    f"Obtaining this information would materially reduce "
                    f"decision uncertainty (currently {uncertainty:.2f}) and "
                    "could move the decision in either direction."
                ),
                projected_impact=None,
                plausibility=_round(plausibility),
            )
        )

    order = {"improve": 0, "worsen": 1, "information": 2}
    scenarios.sort(key=lambda s: (order.get(s.direction, 3), -s.plausibility))
    return scenarios[:_MAX_SCENARIOS]
