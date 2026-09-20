"""``predictron-benchmark`` command-line interface (Project E5).

Subcommands
-----------
* ``benchmark-run``      — run a golden dataset through the current engine,
  store it append-only in history, compute metrics, optionally export.
* ``benchmark-report``   — generate one deterministic report for a run.
* ``benchmark-history``  — list stored runs / verify integrity.
* ``benchmark-compare``  — compare two runs (engine drift + metric drift).
* ``benchmark-export``   — export a run/metrics/reports to JSON files.

The CLI is deterministic: identical inputs produce identical metric files.
History writes are append-only and never overwrite.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.ground_truth_eval.dataset import (
    DEFAULT_GOLDEN_DIR,
    GoldenDatasetError,
    discover_golden_datasets,
    load_golden_dataset,
)
from benchmarks.ground_truth_eval.drift import DriftDetector, detect_metric_drift
from benchmarks.ground_truth_eval.export import (
    ExportError,
    export_bundle,
    export_metrics,
    export_run,
    make_bundle_paths,
    write_json,
)
from benchmarks.ground_truth_eval.history import BenchmarkHistory, HistoryError
from benchmarks.ground_truth_eval.metrics import compute_metrics, metrics_from_dict
from benchmarks.ground_truth_eval.reports import REPORT_KINDS, build_report
from benchmarks.ground_truth_eval.runner import BenchmarkRunner, RunnerError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predictron-benchmark",
        description="Ground-truth evaluation & venture benchmark platform (Project E5).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("benchmark-run", help="Run a golden dataset through the engine")
    run_p.add_argument("--dataset", required=True, help="Dataset path or bare name")
    run_p.add_argument(
        "--history",
        type=Path,
        default=None,
        help="History directory (append-only store). Defaults to the package history dir.",
    )
    run_p.add_argument("--run-id", default=None, help="Explicit run id")
    run_p.add_argument("--no-store", action="store_true", help="Do not persist to history")
    run_p.add_argument("--export-dir", type=Path, default=None, help="Also export artifacts here")
    run_p.set_defaults(func=_cmd_run)

    report_p = sub.add_parser("benchmark-report", help="Generate one report for a run")
    report_p.add_argument("--run", required=True, dest="run_id")
    report_p.add_argument("--kind", default="executive", choices=REPORT_KINDS)
    report_p.add_argument("--history", type=Path, default=None, help="History directory")
    report_p.add_argument("--out", type=Path, default=None, help="Output JSON file")
    report_p.set_defaults(func=_cmd_report)

    history_p = sub.add_parser("benchmark-history", help="List or verify stored runs")
    history_p.add_argument("--history", type=Path, default=None, help="History directory")
    history_p.add_argument("--dataset", default=None, help="Filter by dataset name")
    history_p.add_argument(
        "--verify", action="store_true", help="Re-verify result hashes of stored runs"
    )
    history_p.set_defaults(func=_cmd_history)

    compare_p = sub.add_parser("benchmark-compare", help="Compare two runs")
    compare_p.add_argument("--run-a", required=True, dest="run_a")
    compare_p.add_argument("--run-b", required=True, dest="run_b")
    compare_p.add_argument("--history", type=Path, default=None, help="History directory")
    compare_p.add_argument("--out", type=Path, default=None, help="Output JSON file")
    compare_p.set_defaults(func=_cmd_compare)

    export_p = sub.add_parser("benchmark-export", help="Export run artifacts to JSON")
    export_p.add_argument("--run", required=True, dest="run_id")
    export_p.add_argument("--history", type=Path, default=None, help="History directory")
    export_p.add_argument("--out", type=Path, default=None, help="Output directory")
    export_p.set_defaults(func=_cmd_export)
    return parser


def _history_for(value: Path | None) -> BenchmarkHistory:
    from benchmarks.ground_truth_eval.history import DEFAULT_HISTORY_DIR

    return BenchmarkHistory(value if value is not None else DEFAULT_HISTORY_DIR)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (HistoryError, RunnerError, GoldenDatasetError, ExportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


def _resolve_dataset(dataset_arg: str) -> Any:
    path = Path(dataset_arg)
    if path.exists():
        return load_golden_dataset(path)
    for candidate in discover_golden_datasets():
        if candidate.stem == dataset_arg:
            return load_golden_dataset(candidate)
    try:
        return load_golden_dataset(DEFAULT_GOLDEN_DIR / f"{dataset_arg}.json")
    except OSError:
        raise GoldenDatasetError(
            f"dataset {dataset_arg!r} not found as a file or a discovered dataset"
        )


def _cmd_run(args: argparse.Namespace) -> None:
    dataset = _resolve_dataset(args.dataset)
    history = _history_for(args.history)
    runner = BenchmarkRunner()
    run = runner.run(dataset, run_id=args.run_id)
    metrics = compute_metrics(dataset, run)

    print(f"run_id:      {run.run_id}")
    print(f"engine:      {run.engine_version}")
    print(f"benchmark:   {run.benchmark_version} ({run.dataset_name})")
    print(f"dataset_hash:{run.dataset_hash}")
    print(f"entries:     {len(run.entries)} run, {len(run.successful_entries)} successful")
    print(
        f"accuracy:    {metrics.confusion.accuracy!s}"
        f"  ece: {metrics.calibration.expected_calibration_error!s}"
    )

    if not args.no_store:
        path = history.append(run, metrics=metrics.to_dict())
        print(f"stored:      {path}")
    if args.export_dir is not None:
        _write_export_bundle(args.export_dir, dataset, run, metrics)
        print(f"exported:    {args.export_dir}")


def _write_export_bundle(directory: Path, dataset: Any, run: Any, metrics: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    paths = make_bundle_paths(directory, run_id=run.run_id)
    export_run(run, paths["run"])
    export_metrics(metrics, paths["metrics"])
    export_bundle(
        dataset=dataset,
        run=run,
        metrics=metrics,
        reports=None,
        path=paths["bundle"],
    )


def _cmd_report(args: argparse.Namespace) -> None:
    history = _history_for(args.history)
    run = history.load(args.run_id)
    block = history.load_metrics(args.run_id)
    if args.kind == "trend":
        report = build_report(args.kind, history=history, dataset_name=run.dataset_name)
    elif args.kind == "engine_drift":
        raise ValueError("engine_drift report requires benchmark-compare; two runs are needed")
    elif args.kind == "leaderboard":
        raise ValueError("leaderboard report requires the dataset; use --dataset")
    else:
        metrics_obj = metrics_from_dict(block) if block is not None else None
        if metrics_obj is None:
            raise ValueError(f"run {args.run_id} has no stored metrics; re-run with benchmark-run")
        report = build_report(args.kind, metrics=metrics_obj)

    payload = report if isinstance(report, dict) else report
    if args.out is not None:
        write_json(payload, args.out)
        print(f"wrote: {args.out}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _cmd_history(args: argparse.Namespace) -> None:
    history = _history_for(args.history)
    if args.verify:
        status = history.verify_integrity()
        for run_id in sorted(status):
            print(f"{run_id}: {status[run_id]}")
        print(f"verified {len(status)} run(s)")
        return
    summaries = history.list_summaries()
    if args.dataset is not None:
        summaries = [s for s in summaries if s.dataset_name == args.dataset]
    if not summaries:
        print("no runs stored")
        return
    for s in summaries:
        print(
            f"{s.run_id}  engine={s.engine_version}  benchmark={s.benchmark_version}"
            f"  entries={s.entry_count}(ok={s.successful_count})"
            f"  dataset={s.dataset_name}"
        )
    print(f"{len(summaries)} run(s)")


def _cmd_compare(args: argparse.Namespace) -> None:
    history = _history_for(args.history)
    run_a = history.load(args.run_a)
    run_b = history.load(args.run_b)
    drift = DriftDetector().detect(run_a, run_b)
    metrics_a = history.load_metrics(args.run_a)
    metrics_b = history.load_metrics(args.run_b)
    metric_drift = None
    if metrics_a is not None and metrics_b is not None:
        metric_drift = detect_metric_drift(metrics_a, metrics_b)

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "drift": drift.to_dict(),
        "metric_drift": metric_drift.to_dict() if metric_drift is not None else None,
    }
    if args.out is not None:
        write_json(payload, args.out)
        print(f"wrote: {args.out}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _cmd_export(args: argparse.Namespace) -> None:
    history = _history_for(args.history)
    run = history.load(args.run_id)
    out_dir = args.out or Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = make_bundle_paths(out_dir, run_id=run.run_id)
    export_run(run, paths["run"])
    block = history.load_metrics(args.run_id)
    if block is not None:
        metrics_obj = metrics_from_dict(block)
        export_metrics(metrics_obj, paths["metrics"])
        print(f"wrote: {paths['metrics']}")
    print(f"wrote: {paths['run']}")


if __name__ == "__main__":
    raise SystemExit(main())
