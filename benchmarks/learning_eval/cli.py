"""``predictron-learning-eval`` command-line interface (CIH Phase 7).

Subcommands
-----------
* ``eval-run``      — run the deterministic learning-eval suite, store it
  append-only in history, and print a pass/fail summary.
* ``eval-report``   — render a stored run as a human-readable report.
* ``eval-history``  — list stored runs and verify their integrity.

The suite is deterministic: identical inputs always yield the identical run
id, and a second ``eval-run`` for that id is rejected by the append-only
store rather than silently overwritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmarks.learning_eval.history import (
    DEFAULT_HISTORY_DIR,
    HistoryError,
    LearningEvalHistory,
)
from benchmarks.learning_eval.models import LearningEvalRun
from benchmarks.learning_eval.suite import fail_counts, run_suite

__all__ = ["main"]

_SUCCESS_MARK = "PASS"
_FAILURE_MARK = "FAIL"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predictron-learning-eval",
        description=(
            "Deterministic self-evaluation of the continuous learning layer "
            "(CIH Phase 7)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("eval-run", help="Run the learning-eval suite")
    run_p.add_argument(
        "--history",
        type=Path,
        default=None,
        help="History directory (append-only store). Defaults to the package dir.",
    )
    run_p.add_argument(
        "--no-store",
        action="store_true",
        help="Compute the suite without persisting to history",
    )
    run_p.add_argument(
        "--json",
        action="store_true",
        help="Print the full machine-readable report as JSON",
    )
    run_p.set_defaults(func=_cmd_run)

    report_p = sub.add_parser("eval-report", help="Render a stored run")
    report_p.add_argument("--run", required=True, dest="run_id")
    report_p.add_argument(
        "--history", type=Path, default=None, help="History directory"
    )
    report_p.add_argument(
        "--json", action="store_true", help="Print the run as JSON instead"
    )
    report_p.set_defaults(func=_cmd_report)

    history_p = sub.add_parser("eval-history", help="List or verify stored runs")
    history_p.add_argument(
        "--history", type=Path, default=None, help="History directory"
    )
    history_p.add_argument(
        "--verify",
        action="store_true",
        help="Re-verify integrity hashes of all stored runs",
    )
    history_p.set_defaults(func=_cmd_history)
    return parser


def _history_for(value: Path | None) -> LearningEvalHistory:
    return LearningEvalHistory(value if value is not None else DEFAULT_HISTORY_DIR)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
        return int(result)
    except (HistoryError, AssertionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def _cmd_run(args: argparse.Namespace) -> int:
    report, run_id = run_suite()
    failed, total = fail_counts(report.run)
    print(
        f"learning-eval {run_id} — "
        f"{total - failed}/{total} assertions passed "
        f"({len(report.snapshots)} fixtures, engine {report.run.engine_version})"
    )
    if args.json:
        import json

        print(
            json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str)
        )
    if not args.no_store:
        history = _history_for(args.history)
        path = history.append(report)
        print(f"stored append-only: {path}")
    if total == 0 or failed > 0:
        return 1
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    history = _history_for(args.history)
    run = history.read(args.run_id)
    if args.json:
        import json

        print(json.dumps(run.to_dict(), indent=2, sort_keys=True, default=str))
        return 0
    print(render_report(run))
    return 0


def _cmd_history(args: argparse.Namespace) -> int:
    history = _history_for(args.history)
    if args.verify:
        failures = history.verify()
        for failure in failures:
            print(f"corrupt: {failure}")
        runs = history.runs()
        print(f"verified {len(runs)} run(s), {len(failures)} failure(s)")
        return 1 if failures else 0
    runs = history.runs()
    if not runs:
        print("no learning-eval runs stored")
        return 0
    for run in runs:
        failed, total = fail_counts(run)
        print(
            f"{run.run_id}  {run.created_at}  engine={run.engine_version}  "
            f"assertions={total - failed}/{total}  "
            f"[{_SUCCESS_MARK if run.passed else _FAILURE_MARK}]"
        )
    return 0


def render_report(run: LearningEvalRun) -> str:
    """Human-readable rendering of one suite run."""
    lines = [
        f"Learning-eval suite run: {run.run_id}",
        f"Engine version        : {run.engine_version}",
        f"Created at            : {run.created_at}",
        f"Overall               : "
        f"[{_SUCCESS_MARK if run.passed else _FAILURE_MARK}]",
    ]
    for case in run.cases:
        failed = sum(1 for a in case.assertions if not a.passed)
        lines.append("")
        lines.append(
            f"  case {case.case_id}: "
            f"{case.total - failed}/{case.total} assertions "
            f"[{_SUCCESS_MARK if failed == 0 else _FAILURE_MARK}]"
        )
        for assertion in case.assertions:
            if not assertion.passed:
                lines.append(f"    - {assertion.name}: {assertion.detail}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
