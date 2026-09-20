"""Deterministic cohort validation reports (Phase 4 prediction validation).

Builds a JSON report document (and a compact Markdown rendering) from a
:class:`~benchmarks.cohort.execute.CohortExecutionResult`.  The report is
deterministic in *content* — every table/breakdown is sorted and metrics
are computed with fixed rounding — and ``generated_at`` is the only
wall-clock value (explicitly informational).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.cohort.execute import (
    STATUS_FROZEN,
    CohortExecutionResult,
)


def build_cohort_report(result: CohortExecutionResult) -> dict[str, Any]:
    """Build a deterministic JSON report document for a cohort execution."""
    rows = [
        {
            "company_id": company_id,
            "status": company.status,
            "frozen": company.frozen,
            "prediction_id": company.prediction_id,
            "decision": (
                company.prediction.prediction.decision.value
                if company.prediction is not None
                else None
            ),
            "confidence": (
                float(company.prediction.prediction.confidence)
                if company.prediction is not None
                else None
            ),
            "composite_score": (
                _round(float(company.prediction.prediction.composite_score))
                if company.prediction is not None
                else None
            ),
            "horizon_satisfied": _horizon_satisfied(company),
            "blocked_lookahead": company.blocked_lookahead,
            "evaluation_verdict": (
                company.evaluation.verdict.value
                if company.evaluation is not None
                else None
            ),
            "outcome_verdict": (
                company.outcome_record.verdict.value
                if company.outcome_record is not None
                else None
            ),
            "alignment": (
                company.evaluation.alignment.value
                if company.evaluation is not None
                else None
            ),
            "decision_match": (
                company.evaluation.decision_match
                if company.evaluation is not None
                else None
            ),
            "error": company.error,
        }
        for company_id, company in sorted(result.company_results.items())
    ]

    return {
        "report_type": "cohort_validation",
        "dataset_name": result.dataset_name,
        "benchmark_version": result.benchmark_version,
        "manifest_hash": result.manifest_hash,
        "engine_version": result.engine_version,
        "evaluation_horizon_days": result.evaluation_horizon_days,
        "freeze_store_root": (
            str(result.freeze_store_root) if result.freeze_store_root is not None else None
        ),
        "summary": {
            "companies_total": len(result.company_results),
            "frozen": len(result.frozen_company_ids),
            "pending_ground_truth": len(result.pending_company_ids),
            "failed": len(result.failed_company_ids),
            "predictions_frozen": len(result.frozen_predictions),
            "evaluations": len(result.evaluations),
        },
        "metrics": result.metrics.summary(),
        "calibration": _calibration_document(result.calibration),
        "companies": rows,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def build_cohort_report_markdown(result: CohortExecutionResult) -> str:
    """Render the cohort validation document as compact Markdown."""
    lines: list[str] = [
        f"# Cohort validation: {result.dataset_name}",
        "",
        f"- Benchmark version: `{result.benchmark_version}`",
        f"- Manifest hash: `{result.manifest_hash}`",
        f"- Engine version: `{result.engine_version}`",
        f"- Evaluation horizon (days): `{result.evaluation_horizon_days}`",
        f"- Predictions frozen: `{len(result.frozen_predictions)}`",
        f"- Evaluations: `{len(result.evaluations)}`",
    ]
    metrics = result.metrics.summary()
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| metric | value |",
            "| --- | --- |",
            f"| scoreable | {metrics['scoreable']} |",
            f"| accuracy | {metrics['accuracy']} |",
            f"| precision | {metrics['precision']} |",
            f"| recall | {metrics['recall']} |",
            f"| f1 | {metrics['f1']} |",
            f"| coverage | {metrics['coverage']} |",
        ]
    )
    calibration = result.calibration
    lines.extend(
        [
            "",
            "## Calibration",
            "",
            "| metric | value |",
            "| --- | --- |",
            f"| ECE | {calibration.get('expected_calibration_error')} |",
            f"| overconfidence_detected | {calibration.get('overconfidence_detected')} |",
            f"| total_samples | {calibration.get('total_samples')} |",
        ]
    )
    lines.extend(["", "## Companies", ""])
    for row in build_cohort_report(result)["companies"]:
        lines.append(
            f"- `{row['company_id']}` [{row['status']}] "
            f"decision=`{row['decision']}` confidence=`{row['confidence']}` "
            f"verdict=`{row['evaluation_verdict']}` "
            f"alignment=`{row['alignment']}`"
        )
    return "\n".join(lines) + "\n"


def dump_cohort_report(
    result: CohortExecutionResult,
    path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a cohort validation report to disk (refuse-to-clobber by default)."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"cohort report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        build_cohort_report(result), indent=2, sort_keys=True, ensure_ascii=False, default=str
    ) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return path


def _horizon_satisfied(company: Any) -> bool:
    if company.outcome_record is None:
        return False
    return company.outcome_record.verdict.value not in ("unknown",)


def _calibration_document(calibration: dict[str, Any]) -> dict[str, Any]:
    return {
        "expected_calibration_error": calibration.get("expected_calibration_error", 0.0),
        "maximum_calibration_error": calibration.get("maximum_calibration_error", 0.0),
        "overconfidence_detected": calibration.get("overconfidence_detected", False),
        "overconfident_bins_count": calibration.get("overconfident_bins_count", 0),
        "total_samples": calibration.get("total_samples", 0),
        "calibration_quality": calibration.get("calibration_quality", "insufficient_data"),
        "bins": calibration.get("bins", []),
    }


def _round(value: float) -> float:
    return round(value, 4)


__all__ = [
    "build_cohort_report",
    "build_cohort_report_markdown",
    "dump_cohort_report",
    "STATUS_FROZEN",
]
