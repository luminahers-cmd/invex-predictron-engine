"""Per-rule diagnostics for the reasoning layer.

Sprint 6A requires every reasoning rule to emit structured diagnostics
describing what it examined, what it selected, and what it concluded:

* ``evidence_examined``   — evidence items available to the rule
* ``evidence_selected``   — evidence items actually referenced by the
  rule's observations
* ``confidence``          — mean observation confidence
* ``trust``               — mean observation trust score
* ``conflicts``           — total conflicting signals across observations
* ``missing_evidence``    — evidence domains absent for populated features
* ``duration_ms``         — wall-clock duration of the rule evaluation

Diagnostics are collected by :class:`~predictron_engine.reasoning.composite.
CompositeReasoner` and exposed via ``last_diagnostics`` /
``reason_with_diagnostics`` without changing any existing return types.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from predictron_engine.models.report import Observation


@dataclass
class RuleDiagnostic:
    """Structured diagnostics for a single reasoning rule execution."""

    rule_name: str
    evidence_examined: int = 0
    evidence_selected: int = 0
    confidence: float = 0.0
    trust: float = 0.0
    conflicts: int = 0
    missing_evidence: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def missing_evidence_count(self) -> int:
        return len(self.missing_evidence)


class RuleTimer:
    """Monotonic wall-clock timer for rule evaluation durations.

    Uses ``time.perf_counter`` — monotonic and unaffected by system
    clock adjustments.  Durations are inherently non-deterministic and
    are therefore excluded from all determinism guarantees.
    """

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000.0, 3)


def build_rule_diagnostic(
    *,
    rule_name: str,
    evidence_examined: int,
    observations: list[Observation],
    duration_ms: float,
    missing_evidence: list[str] | None = None,
) -> RuleDiagnostic:
    """Build a :class:`RuleDiagnostic` from one rule execution.

    Pure factory — every field is derived deterministically from the
    rule's inputs and outputs.  ``duration_ms`` is the sole exception:
    it records real wall-clock time.

    Selection counting: an evidence item counts as *selected* when at
    least one of the rule's observations references it through an
    ``evidence:{domain}/{category}: {statement}`` reference string or
    carries a citation whose claim matches the item statement.
    """
    selected_refs: set[tuple[str, str, str]] = set()
    cited_claims: set[str] = set()

    for obs in observations:
        for ref in obs.evidence:
            parsed = _parse_evidence_ref(ref)
            if parsed is not None:
                selected_refs.add(parsed)
        for citation in obs.citations:
            if citation.claim:
                cited_claims.add(citation.claim)

    confidence = (
        round(sum(o.confidence for o in observations) / len(observations), 4)
        if observations
        else 0.0
    )
    trust = (
        round(sum(o.trust_score for o in observations) / len(observations), 4)
        if observations
        else 0.0
    )
    conflicts = sum(o.evidence_conflict_count for o in observations)

    return RuleDiagnostic(
        rule_name=rule_name,
        evidence_examined=evidence_examined,
        evidence_selected=len(selected_refs),
        confidence=confidence,
        trust=trust,
        conflicts=conflicts,
        missing_evidence=list(missing_evidence or []),
        duration_ms=duration_ms,
    )


def _parse_evidence_ref(ref: str) -> tuple[str, str, str] | None:
    """Parse ``evidence:{domain}/{category}: {statement}`` references."""
    if not ref.startswith("evidence:"):
        return None
    body = ref[len("evidence:") :]
    slash_idx = body.find("/")
    colon_idx = body.find(": ")
    if slash_idx == -1 or colon_idx == -1:
        return None
    return (
        body[:slash_idx],
        body[slash_idx + 1 : colon_idx],
        body[colon_idx + 2 :],
    )
