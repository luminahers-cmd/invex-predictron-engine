"""Sprint 9 — Rule effectiveness measurement.

Analyses which reasoning rules actually fire and how they contribute to
the analysis, across a collection of :class:`Report` objects (or their
:class:`PerformanceSnapshot` projections).  Purely deterministic — the
reports are supplied by the caller.

Rules are identified by ``Observation.source_rule``.  The contribution of
a rule is its share of all observations analysed.
"""

from __future__ import annotations

from predictron_engine.intelligence.models import (
    PerformanceSnapshot,
    RuleEffectiveness,
)
from predictron_engine.models.report import Observation, Report


def _obs_by_rule(
    observations: list[Observation],
) -> dict[str, list[Observation]]:
    grouped: dict[str, list[Observation]] = {}
    for obs in observations:
        grouped.setdefault(obs.source_rule, []).append(obs)
    return grouped


def _rule_effectiveness_from_observations(
    observations: list[Observation],
    total_observations: int,
) -> list[RuleEffectiveness]:
    grouped = _obs_by_rule(observations)
    findings: list[RuleEffectiveness] = []
    for rule in sorted(grouped):
        items = grouped[rule]
        dimensions = sorted({o.dimension for o in items})
        avg_importance = round(
            sum(o.importance for o in items) / len(items), 4
        )
        avg_confidence = round(
            sum(o.confidence for o in items) / len(items), 4
        )
        contribution = (
            round(len(items) / total_observations, 4)
            if total_observations > 0
            else 0.0
        )
        findings.append(
            RuleEffectiveness(
                source_rule=rule,
                hit_count=len(items),
                observed_dimensions=dimensions,
                avg_importance=avg_importance,
                avg_confidence=avg_confidence,
                contribution=contribution,
            )
        )
    return findings


def rule_effectiveness_from_reports(
    reports: list[Report],
) -> list[RuleEffectiveness]:
    """Compute rule effectiveness across a collection of reports."""
    all_observations: list[Observation] = []
    for report in reports:
        all_observations.extend(report.observations)
    return _rule_effectiveness_from_observations(
        all_observations, len(all_observations)
    )


def rule_effectiveness_from_snapshots(
    snapshots: list[PerformanceSnapshot],
    reports: list[Report] | None = None,
) -> list[RuleEffectiveness]:
    """Compute rule effectiveness from snapshot rule-hits.

    When ``reports`` is supplied it is used (richer, includes importance
    and confidence); otherwise only hit counts from the snapshots are
    reported with zeroed importance/confidence.
    """
    if reports is not None:
        return rule_effectiveness_from_reports(reports)

    total_hits = sum(
        sum(s.rule_hits.values()) for s in snapshots
    )
    aggregated: dict[str, int] = {}
    dimensions_by_rule: dict[str, set[str]] = {}
    for snapshot in snapshots:
        for rule, count in snapshot.rule_hits.items():
            aggregated[rule] = aggregated.get(rule, 0) + count
            dimensions_by_rule.setdefault(rule, set()).update(
                snapshot.dimension_scores.keys()
            )

    findings: list[RuleEffectiveness] = []
    for rule in sorted(aggregated):
        hits = aggregated[rule]
        findings.append(
            RuleEffectiveness(
                source_rule=rule,
                hit_count=hits,
                observed_dimensions=sorted(dimensions_by_rule.get(rule, set())),
                avg_importance=0.0,
                avg_confidence=0.0,
                contribution=(
                    round(hits / total_hits, 4) if total_hits > 0 else 0.0
                ),
            )
        )
    return findings
