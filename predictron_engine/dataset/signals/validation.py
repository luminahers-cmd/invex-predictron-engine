"""Signal and timeline validation (Project E4).

Validation is **read-only**: it never repairs or silences anything.  A
signal or timeline with a finding is reported and left untouched, so
operators can decide how to act.  Findings are deterministic and sorted.

Checks performed
----------------
- per-signal: required company id, tz-aware timestamp (naive values are
  rejected rather than silently assumed), confidence in [0, 1], non-empty
  source, and a usable evidence reference.
- per-collection: chronological consistency (non-decreasing timestamps),
  duplicate signal ids, and single-company consistency.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from predictron_engine.dataset.signals.model import CompanySignal
from predictron_engine.dataset.signals.timeline import CompanyTimeline


@dataclass(frozen=True)
class SignalValidationIssue:
    """A single validation finding."""

    signal_id: str | None
    kind: str
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "signal_id": self.signal_id,
            "kind": self.kind,
            "detail": self.detail,
        }


@dataclass
class SignalValidationReport:
    """Aggregated validation findings."""

    issues: list[SignalValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def by_kind(self) -> dict[str, int]:
        counts: Counter[str] = Counter(i.kind for i in self.issues)
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "issue_count": self.issue_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def validate_signal(signal: CompanySignal) -> list[SignalValidationIssue]:
    """Validate a single signal; returns any findings (possibly empty)."""
    issues: list[SignalValidationIssue] = []
    sid = signal.signal_id
    if not signal.company_id:
        issues.append(SignalValidationIssue(sid, "empty_company_id"))
    if signal.timestamp.tzinfo is None:
        issues.append(SignalValidationIssue(sid, "naive_timestamp"))
    if not 0.0 <= signal.confidence <= 1.0:
        issues.append(SignalValidationIssue(sid, "invalid_confidence"))
    if not signal.source:
        issues.append(SignalValidationIssue(sid, "missing_source"))
    if not signal.evidence.reference and not signal.evidence.description:
        issues.append(SignalValidationIssue(sid, "missing_evidence"))
    return issues


def validate_signals(
    company_id: str, signals: Iterable[CompanySignal]
) -> SignalValidationReport:
    """Validate a raw collection of signals as a prospective timeline.

    Checks ordering, duplicate ids, company consistency, and per-signal
    validity.  The input is treated as *unsorted* so an out-of-order
    sequence is reported instead of being silently reordered.
    """
    report = SignalValidationReport()
    ordered = list(signals)
    for signal in ordered:
        report.issues.extend(validate_signal(signal))
        if signal.company_id != company_id:
            report.issues.append(
                SignalValidationIssue(
                    signal.signal_id,
                    "company_mismatch",
                    detail=(
                        f"signal company '{signal.company_id}' "
                        f"does not match '{company_id}'"
                    ),
                )
            )
    for i in range(1, len(ordered)):
        if ordered[i].timestamp < ordered[i - 1].timestamp:
            report.issues.append(
                SignalValidationIssue(
                    ordered[i].signal_id,
                    "out_of_order",
                    detail=(
                        f"timestamp {ordered[i].timestamp.isoformat()} "
                        f"precedes {ordered[i - 1].timestamp.isoformat()}"
                    ),
                )
            )
    counts: Counter[str] = Counter(s.signal_id for s in ordered)
    for signal_id, count in sorted(counts.items()):
        if count > 1:
            report.issues.append(
                SignalValidationIssue(
                    signal_id, "duplicate_signal_id", detail=f"seen {count} times"
                )
            )
    report.issues.sort(key=_issue_key)
    return report


def validate_timeline(timeline: CompanyTimeline) -> SignalValidationReport:
    """Validate a constructed timeline (already sorted and de-duplicated)."""
    if timeline.signal_count == 0:
        return SignalValidationReport()
    return validate_signals(timeline.company_id, timeline.signals)


def _issue_key(issue: SignalValidationIssue) -> tuple[str, str, str | None]:
    return (issue.kind, issue.detail, issue.signal_id)
