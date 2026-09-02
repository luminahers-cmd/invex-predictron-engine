"""Dataset validation (Part F).

Validates the integrity of a dataset held in a :class:`DatasetStore`:

  - duplicate IDs
  - missing predictions
  - missing outcomes
  - schema mismatches
  - future timestamps
  - invalid chronology

Validation never silently repairs data.  It reports issues deterministically
so operators can decide whether to act.  A record with an issue is flagged
but its stored representation is left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from predictron_engine.dataset.evaluation import PredictionEvaluation
from predictron_engine.dataset.models import DatasetRecord
from predictron_engine.dataset.store import DatasetStore


@dataclass
class ValidationIssue:
    """A single validation finding."""

    kind: str
    record_id: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "record_id": self.record_id,
            "detail": self.detail,
        }


@dataclass
class ValidationReport:
    """Aggregated validation results."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.kind] = counts.get(issue.kind, 0) + 1
        return counts

    def to_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "issue_count": self.issue_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def validate_dataset(
    store: DatasetStore,
    now: datetime | None = None,
) -> ValidationReport:
    """Validate all records, outcomes, and evaluations in a store.

    Parameters
    ----------
    store :
        The dataset store to validate.
    now :
        Reference time for future-timestamp checks, defaults to UTC now.

    Returns
    -------
    A ValidationReport listing every issue found.
    """
    report = ValidationReport()
    now = now or datetime.now(UTC)

    _validate_records(store, report, now)
    _validate_outcomes(store, report, now)
    _validate_evaluations(store, report)

    return report


def _validate_records(
    store: DatasetStore, report: ValidationReport, now: datetime
) -> None:
    record_ids = store.list_records()

    # Duplicate IDs (would collide on disk by construction, but check for
    # stale manifests or externally-edited stores).
    seen: set[str] = set()
    for record_id in record_ids:
        if record_id in seen:
            report.issues.append(
                ValidationIssue("duplicate_record_id", record_id)
            )
        seen.add(record_id)

    for record_id in record_ids:
        try:
            record = store.load_record(record_id)
        except Exception:  # noqa: BLE001 - schema mismatch on load
            report.issues.append(
                ValidationIssue("unreadable_record", record_id)
            )
            continue
        if record is None:
            report.issues.append(
                ValidationIssue("unreadable_record", record_id)
            )
            continue
        _validate_record(record, report, now)


def _validate_record(
    record: DatasetRecord, report: ValidationReport, now: datetime
) -> None:
    # Missing prediction
    if record.prediction is None:
        report.issues.append(
            ValidationIssue("missing_prediction", record.record_id)
        )
        return

    # Future analysis date
    if record.analysis_date > now:
        report.issues.append(
            ValidationIssue(
                "future_analysis_date",
                record.record_id,
                detail=f"analysis_date {record.analysis_date.isoformat()} "
                f"is in the future relative to {now.isoformat()}",
            )
        )

    # Invalid chronology: analysis_metadata may carry a processing/recorded
    # timestamp that must not precede the analysis date.
    recorded = record.analysis_metadata.get("recorded_at")
    recorded_dt = _parse_dt(recorded)
    if recorded_dt is not None and recorded_dt < record.analysis_date:
        report.issues.append(
            ValidationIssue(
                "invalid_chronology",
                record.record_id,
                detail="recorded_at precedes analysis_date",
            )
        )


def _validate_outcomes(
    store: DatasetStore, report: ValidationReport, now: datetime
) -> None:
    outcome_ids = store.list_outcomes()
    records = {
        rid: True for rid in store.list_records()
    }

    for outcome_id in outcome_ids:
        try:
            outcome = store.load_outcome(outcome_id)
        except Exception:  # noqa: BLE001 - schema mismatch on load
            report.issues.append(
                ValidationIssue("unreadable_outcome", None, outcome_id)
            )
            continue
        if outcome is None:
            report.issues.append(
                ValidationIssue("unreadable_outcome", None, outcome_id)
            )
            continue

        # Missing outcome data
        if outcome.outcome is None:
            report.issues.append(
                ValidationIssue("missing_outcome_data", None, outcome_id)
            )
            continue

        # Outcome references a record that does not exist -> orphan outcome
        if outcome.record_id not in records:
            report.issues.append(
                ValidationIssue(
                    "orphan_outcome",
                    outcome.record_id,
                    detail=f"outcome_id {outcome_id} references unknown record",
                )
            )

        # Future outcome date
        if outcome.outcome.latest_verification_date is not None and (
            outcome.outcome.latest_verification_date > now
        ):
            report.issues.append(
                ValidationIssue(
                    "future_verification_date",
                    outcome.record_id,
                    detail=(
                        "latest_verification_date "
                        f"{outcome.outcome.latest_verification_date.isoformat()} "
                        "is in the future"
                    ),
                )
            )


def _validate_evaluations(
    store: DatasetStore, report: ValidationReport
) -> None:
    evaluation_ids = store.list_evaluations()
    records = {
        rid: True for rid in store.list_records()
    }

    for evaluation_id in evaluation_ids:
        try:
            evaluation = store.load_evaluation(evaluation_id)
        except Exception:  # noqa: BLE001 - schema mismatch on load
            report.issues.append(
                ValidationIssue(
                    "unreadable_evaluation", None, evaluation_id
                )
            )
            continue
        if evaluation is None:
            report.issues.append(
                ValidationIssue(
                    "unreadable_evaluation", None, evaluation_id
                )
            )
            continue
        _validate_evaluation(evaluation, report, records)


def _validate_evaluation(
    evaluation: PredictionEvaluation,
    report: ValidationReport,
    records: dict[str, bool],
) -> None:
    if evaluation.record_id not in records:
        report.issues.append(
            ValidationIssue(
                "orphan_evaluation",
                evaluation.record_id,
                detail=f"evaluation_id {evaluation.evaluation_id} references unknown record",
            )
        )

    # Schema mismatch: the evaluation's frozen prediction must carry a
    # plausible decision/confidence/score that round-trips; if confidence
    # falls outside the valid range the model validation would already
    # reject it, but a direct check guards against schema drift.
    if evaluation.prediction.confidence < 0.0 or evaluation.prediction.confidence > 1.0:
        report.issues.append(
            ValidationIssue(
                "schema_mismatch",
                evaluation.record_id,
                detail="prediction.confidence outside [0, 1]",
            )
        )


def _parse_dt(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None
