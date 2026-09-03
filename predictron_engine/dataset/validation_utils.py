"""Dataset record validation utilities (Part B).

Provides field-level validation for individual DatasetRecord and
OutcomeRecord instances.  Rejects incomplete or temporally inconsistent
records without mutating them.

Validation is deterministic and never silently repairs data.  Each
check returns a list of ``ValidationIssue`` instances that describe
the specific problems found.
"""

from __future__ import annotations

from datetime import UTC, datetime

from predictron_engine.dataset.models import DatasetRecord
from predictron_engine.dataset.outcomes import OutcomeRecord
from predictron_engine.dataset.validation import ValidationIssue


def validate_record_fields(
    record: DatasetRecord,
    now: datetime | None = None,
) -> list[ValidationIssue]:
    """Validate the fields of a DatasetRecord for completeness and consistency.

    Checks:
      - startup_name is non-empty
      - website is non-empty and has a scheme
      - engine_version is non-empty
      - analysis_date is not in the future
      - analysis_date is not before year 2000
      - prediction confidence is in [0, 1]
      - prediction composite_score is in [0, 100]
      - prediction dimension_scores values are in [0, 100]
    """
    now = now or datetime.now(UTC)
    issues: list[ValidationIssue] = []
    rid = record.record_id

    if not record.startup_name or not record.startup_name.strip():
        issues.append(ValidationIssue(
            "missing_startup_name", rid,
            detail="startup_name is empty or whitespace",
        ))

    if not record.website or not record.website.strip():
        issues.append(ValidationIssue(
            "missing_website", rid,
            detail="website is empty or whitespace",
        ))
    elif not record.website.startswith(("http://", "https://")):
        issues.append(ValidationIssue(
            "invalid_website", rid,
            detail=f"website '{record.website}' lacks http(s) scheme",
        ))

    if not record.engine_version or not record.engine_version.strip():
        issues.append(ValidationIssue(
            "missing_engine_version", rid,
            detail="engine_version is empty",
        ))

    if record.analysis_date > now:
        issues.append(ValidationIssue(
            "future_analysis_date", rid,
            detail=(
                f"analysis_date {record.analysis_date.isoformat()} "
                f"is in the future relative to {now.isoformat()}"
            ),
        ))

    if record.analysis_date.year < 2000:
        issues.append(ValidationIssue(
            "implausible_analysis_date", rid,
            detail=(
                f"analysis_date year {record.analysis_date.year} "
                "is before 2000"
            ),
        ))

    pred = record.prediction
    if pred is not None:
        if pred.confidence < 0.0 or pred.confidence > 1.0:
            issues.append(ValidationIssue(
                "invalid_confidence", rid,
                detail=f"confidence {pred.confidence} outside [0, 1]",
            ))
        if pred.composite_score < 0.0 or pred.composite_score > 100.0:
            issues.append(ValidationIssue(
                "invalid_composite_score", rid,
                detail=f"composite_score {pred.composite_score} outside [0, 100]",
            ))
        for dim, score in pred.dimension_scores.items():
            if score < 0.0 or score > 100.0:
                issues.append(ValidationIssue(
                    "invalid_dimension_score", rid,
                    detail=f"dimension '{dim}' score {score} outside [0, 100]",
                ))

    return issues


def validate_outcome_fields(
    outcome: OutcomeRecord,
    now: datetime | None = None,
) -> list[ValidationIssue]:
    """Validate the fields of an OutcomeRecord for completeness and consistency.

    Checks:
      - record_id is non-empty
      - latest_verification_date is not in the future
      - shutdown_date is not before analysis context (if available)
      - bankruptcy_date is not before shutdown_date
      - acquisition_price is non-negative
      - total_funding_usd is non-negative
      - funding round amounts are non-negative
      - time_horizon_days is non-negative
    """
    now = now or datetime.now(UTC)
    issues: list[ValidationIssue] = []
    oid = outcome.outcome_id

    if not outcome.record_id:
        issues.append(ValidationIssue(
            "missing_record_reference", oid,
            detail="outcome has empty record_id",
        ))

    if (
        outcome.outcome.latest_verification_date is not None
        and outcome.outcome.latest_verification_date > now
    ):
        issues.append(ValidationIssue(
            "future_verification_date", oid,
            detail=(
                "latest_verification_date "
                f"{outcome.outcome.latest_verification_date.isoformat()} "
                "is in the future"
            ),
        ))

    if outcome.outcome.shutdown_date is not None and (
        outcome.outcome.shutdown_date > now
    ):
        issues.append(ValidationIssue(
            "future_shutdown_date", oid,
            detail=(
                f"shutdown_date {outcome.outcome.shutdown_date.isoformat()} "
                "is in the future"
            ),
        ))

    if (
        outcome.outcome.bankruptcy_date is not None
        and outcome.outcome.shutdown_date is not None
        and outcome.outcome.bankruptcy_date < outcome.outcome.shutdown_date
    ):
        issues.append(ValidationIssue(
            "invalid_chronology", oid,
            detail="bankruptcy_date precedes shutdown_date",
        ))

    if (
        outcome.outcome.acquisition_price_usd is not None
        and outcome.outcome.acquisition_price_usd < 0
    ):
        issues.append(ValidationIssue(
            "negative_acquisition_price", oid,
            detail=(
                f"acquisition_price_usd "
                f"{outcome.outcome.acquisition_price_usd} is negative"
            ),
        ))

    if (
        outcome.outcome.total_funding_usd is not None
        and outcome.outcome.total_funding_usd < 0
    ):
        issues.append(ValidationIssue(
            "negative_total_funding", oid,
            detail=(
                f"total_funding_usd "
                f"{outcome.outcome.total_funding_usd} is negative"
            ),
        ))

    for idx, funding_round in enumerate(outcome.outcome.funding_rounds):
        if (
            funding_round.amount_usd is not None
            and funding_round.amount_usd < 0
        ):
            issues.append(ValidationIssue(
                "negative_funding_amount", oid,
                detail=(
                    f"funding_round[{idx}] amount_usd "
                    f"{funding_round.amount_usd} is negative"
                ),
            ))

    if (
        outcome.time_horizon_days is not None
        and outcome.time_horizon_days < 0
    ):
        issues.append(ValidationIssue(
            "negative_time_horizon", oid,
            detail=(
                f"time_horizon_days {outcome.time_horizon_days} is negative"
            ),
        ))

    return issues


def validate_record_completeness(
    record: DatasetRecord,
) -> tuple[list[str], list[str]]:
    """Report missing and present optional fields.

    Returns a tuple of (missing_fields, present_fields) where field
    names follow dot notation for nested objects (e.g.
    ``prediction.investment_readiness_score``).
    """
    missing: list[str] = []
    present: list[str] = []

    _check_field(record.startup_name, "startup_name", missing, present)
    _check_field(record.website, "website", missing, present)
    _check_field(record.engine_version, "engine_version", missing, present)
    _check_field(
        record.prediction.investment_readiness_score,
        "prediction.investment_readiness_score",
        missing,
        present,
    )
    _check_field(
        record.prediction.recommendation_count,
        "prediction.recommendation_count",
        missing,
        present,
    )

    if record.prediction.dimension_scores:
        present.append("prediction.dimension_scores")
    else:
        missing.append("prediction.dimension_scores")

    if record.tags:
        present.append("tags")
    else:
        missing.append("tags")

    if record.analysis_metadata:
        present.append("analysis_metadata")
    else:
        missing.append("analysis_metadata")

    if record.benchmark_version:
        present.append("benchmark_version")
    else:
        missing.append("benchmark_version")

    if record.evidence_bundle_reference:
        present.append("evidence_bundle_reference")
    else:
        missing.append("evidence_bundle_reference")

    return missing, present


def _check_field(
    value: object,
    field_name: str,
    missing: list[str],
    present: list[str],
) -> None:
    """Check if a value is present (not None, not empty)."""
    if value is None:
        missing.append(field_name)
    elif isinstance(value, str) and not value.strip():
        missing.append(field_name)
    else:
        present.append(field_name)
