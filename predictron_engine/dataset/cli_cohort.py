"""Cohort predicate-validation CLI commands (Phase 4).

Three new subcommands for the ``predictron-dataset`` entry point:

  cohort-execute   Run a cohort manifest (JSON or CSV): freeze predictions,
                   evaluate outcomes, and emit the validation report
  cohort-status    List frozen predictions from a PredictionStore
  cohort-report    Freeze/evaluate and write the report doc (JSON or Markdown)

These are fully additive — no existing command is modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from predictron_engine.dataset.prediction_store import PredictionStore


def _json_out(data: object) -> None:
    print(json.dumps(data, indent=2, default=str))


def cmd_cohort_execute(args: argparse.Namespace) -> int:
    """Freeze predictions and evaluate outcomes for a cohort manifest."""
    from benchmarks.cohort.execute import run_cohort_execution
    from benchmarks.cohort.manifest_csv import load_manifest_any
    from benchmarks.cohort.report import (
        build_cohort_report,
        build_cohort_report_markdown,
        dump_cohort_report,
    )

    manifest = load_manifest_any(args.manifest)
    result = run_cohort_execution(
        manifest,
        freeze_root=getattr(args, "freeze_root", None),
    )

    output = getattr(args, "output", None)
    if output:
        if getattr(args, "markdown", False):
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(
                build_cohort_report_markdown(result), encoding="utf-8"
            )
        else:
            dump_cohort_report(result, output)
        print(output)
    else:
        document = build_cohort_report(result)
        if getattr(args, "markdown", False):
            print(build_cohort_report_markdown(result), end="")
        else:
            _json_out(document)

    if result.failed_company_ids:
        print(
            f"error: {len(result.failed_company_ids)} company record(s) failed: "
            + ", ".join(result.failed_company_ids),
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_cohort_status(args: argparse.Namespace) -> int:
    """List frozen predictions stored for a cohort."""
    store = PredictionStore(
        getattr(args, "freeze_root", None) or PredictionStore().root
    )
    company_id = getattr(args, "company_id", None)
    summaries = store.list_summaries()
    if company_id is not None:
        summaries = [s for s in summaries if s.company_id == company_id]
    _json_out(
        {
            "root": str(store.root),
            "count": len(summaries),
            "predictions": [entry.to_dict() for entry in summaries],
        }
    )
    return 0


def add_cohort_subparsers(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the cohort validation subcommands (called from cli.build_parser)."""
    p_execute = sub.add_parser(
        "cohort-execute",
        help=(
            "Run a cohort manifest (JSON/CSV): freeze time-scoped predictions, "
            "evaluate outcomes, and emit the validation report"
        ),
    )
    p_execute.add_argument("manifest", type=str, help="Path to the cohort manifest (JSON/CSV)")
    p_execute.add_argument(
        "--freeze-root",
        type=str,
        default=None,
        help="Directory holding the append-only frozen-prediction store",
    )
    p_execute.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the report (JSON, unless --markdown is given)",
    )
    p_execute.add_argument(
        "--markdown",
        action="store_true",
        default=False,
        help="Render the report as Markdown instead of JSON",
    )
    p_execute.set_defaults(func=cmd_cohort_execute)

    p_status = sub.add_parser(
        "cohort-status",
        help="List frozen predictions stored for a cohort",
    )
    p_status.add_argument(
        "--freeze-root",
        type=str,
        default=None,
        help="Directory holding the append-only frozen-prediction store",
    )
    p_status.add_argument(
        "--company-id",
        type=str,
        default=None,
        help="Restrict to predictions for this company id",
    )
    p_status.set_defaults(func=cmd_cohort_status)


__all__ = ["add_cohort_subparsers"]
