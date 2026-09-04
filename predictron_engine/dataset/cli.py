"""Dataset command-line interface (Part A).

Provides commands for the historical startup dataset pipeline:

  import    Import records/outcomes from a JSON file into a store
  analyze   Run the PredictronEngine over stored records
  evaluate  Create and store evaluations for records with outcomes
  verify    Validate dataset integrity
  stats     Print dataset summary, coverage, outcomes, and metrics
  export    Write a JSON dataset report to a file

All commands delegate to the existing dataset API.  No business logic is
duplicated here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from predictron_engine.dataset.analysis import AnalysisPipeline
from predictron_engine.dataset.evaluation_pipeline import EvaluationPipeline
from predictron_engine.dataset.imports import (
    ImportPipeline,
    ImportSourceRegistry,
)
from predictron_engine.dataset.reports import DatasetReportBuilder
from predictron_engine.dataset.statistics import compute_dataset_stats
from predictron_engine.dataset.store import DatasetStore
from predictron_engine.dataset.validation import ValidationReport, validate_dataset
from predictron_engine.version import ENGINE_VERSION


def _build_store(args: argparse.Namespace) -> DatasetStore:
    store = DatasetStore(args.dataset)
    store.initialize()
    return store


def _json_out(data: object) -> None:
    print(json.dumps(data, indent=2, default=str))


def cmd_import(args: argparse.Namespace) -> int:
    """Import records/outcomes from a supported data file into a store."""
    store = _build_store(args)
    source_name = getattr(args, "source", None) or _infer_source(args.file)
    registry = ImportSourceRegistry.default()
    source = registry.get(source_name)
    if source is None:
        print(
            f"error: unknown import source '{source_name}'",
            file=sys.stderr,
        )
        return 1

    pipeline = ImportPipeline(source)
    result = pipeline.run(args.file)

    for record in result.imported_records:
        store.save_record(record)
    for outcome in result.imported_outcomes:
        outcome.verdict = outcome.derive_verdict()
        store.save_outcome(outcome)

    for idx, errors in result.validation_errors:
        print(
            f"warning: record {idx} rejected: {'; '.join(errors)}",
            file=sys.stderr,
        )

    _json_out(
        {
            "imported": result.records_imported,
            "rejected": result.records_failed,
            "source": source_name,
            "file": args.file,
        }
    )
    return 0


def cmd_acquire(args: argparse.Namespace) -> int:
    """Acquire data from a source into the dataset store."""
    from datetime import UTC, datetime

    from predictron_engine.dataset.acquisition import (
        AcquireOptions,
        AcquisitionManager,
    )

    store = _build_store(args)
    manager = AcquisitionManager(store)

    since: datetime | None = None
    since_str = getattr(args, "since", None)
    if since_str:
        try:
            since = datetime.fromisoformat(since_str).replace(tzinfo=UTC)
        except (ValueError, TypeError):
            print(f"error: invalid --since date: {since_str}", file=sys.stderr)
            return 1

    options = AcquireOptions(
        source_name=getattr(args, "source", None),
        file_path=getattr(args, "file", None),
        directory=getattr(args, "directory", None),
        resume=getattr(args, "resume", False),
        since=since,
        limit=getattr(args, "limit", None),
        dry_run=getattr(args, "dry_run", False),
    )

    if options.resume:
        results = manager.resume(options.source_name)
        for result in results:
            _json_out(result.to_dict())
        return 0

    result = manager.acquire(options)
    _json_out(result.to_dict())
    return 0 if not result.metrics.errors else 1


def cmd_acquire_status(args: argparse.Namespace) -> int:
    """Show acquisition status."""
    from predictron_engine.dataset.acquisition import AcquisitionManager

    store = _build_store(args)
    manager = AcquisitionManager(store)
    _json_out(manager.status())
    return 0


def cmd_acquire_sources(args: argparse.Namespace) -> int:
    """List available source connectors."""
    from predictron_engine.dataset.acquisition.sources import (
        SourceConnectorRegistry,
    )

    registry = SourceConnectorRegistry.default()
    _json_out(registry.list_descriptors())
    return 0


def cmd_acquire_schedule(args: argparse.Namespace) -> int:
    """Manage import schedules."""
    from predictron_engine.dataset.acquisition.scheduler import (
        AcquisitionScheduler,
        ImportSchedule,
    )

    store = _build_store(args)
    scheduler = AcquisitionScheduler(store._root)
    scheduler.initialize()

    subcmd = getattr(args, "schedule_action", "list")

    if subcmd == "list":
        schedules = scheduler.load_schedules()
        _json_out([{
            "source_name": s.source_name,
            "frequency": s.frequency,
            "enabled": s.enabled,
            "last_run": s.last_run,
            "next_run": s.next_run,
            "priority": s.priority,
        } for s in schedules])
        return 0

    if subcmd == "add":
        source = getattr(args, "schedule_source", "")
        freq = getattr(args, "frequency", "weekly")
        if not source:
            print("error: --schedule-source is required", file=sys.stderr)
            return 1
        scheduler.add_schedule(ImportSchedule(
            source_name=source,
            frequency=freq,
        ))
        _json_out({"added": source, "frequency": freq})
        return 0

    if subcmd == "remove":
        source = getattr(args, "schedule_source", "")
        if not source:
            print("error: --schedule-source is required", file=sys.stderr)
            return 1
        removed = scheduler.remove_schedule(source)
        _json_out({"removed": source, "found": removed})
        return 0

    print(f"error: unknown schedule action: {subcmd}", file=sys.stderr)
    return 1


def cmd_dedup(args: argparse.Namespace) -> int:
    """Report duplicate startup records within the store."""
    from predictron_engine.dataset.dedup import find_duplicates

    store = _build_store(args)
    records = [
        store.load_record(rid)
        for rid in store.list_records()
    ]
    valid = [r for r in records if r is not None]
    report = find_duplicates(valid)
    _json_out(
        {
            "group_count": report.group_count,
            "duplicate_record_count": report.duplicate_record_count,
            "method_counts": report.method_counts,
            "groups": [
                [r.record_id for r in group] for group in report.groups
            ],
        }
    )
    return 0


def cmd_statistics(args: argparse.Namespace) -> int:
    """Print dataset statistics: sectors, stages, years, countries, missing, duplicates."""
    store = _build_store(args)
    stats = compute_dataset_stats(store)
    _json_out(stats.to_dict())
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run the PredictronEngine over stored records."""
    store = _build_store(args)
    record_id = getattr(args, "record_id", None)

    records: list[Any] = []
    if record_id:
        record = store.load_record(record_id)
        if record is None:
            print(f"error: record '{record_id}' not found", file=sys.stderr)
            return 1
        records = [record]
    else:
        records = [
            store.load_record(rid)
            for rid in store.list_records()
            if store.load_record(rid) is not None
        ]

    if not records:
        print("no records to analyze", file=sys.stderr)
        return 1

    engine = _build_engine()
    pipeline = AnalysisPipeline(engine)
    results = pipeline.analyze_all(
        records,
        benchmark_version=getattr(args, "benchmark_version", None),
    )

    for result in results:
        store.save_run(result.run)

    _json_out(
        {
            "analyzed": len(results),
            "record_id": record_id,
        }
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Create and store evaluations for records with outcomes."""
    store = _build_store(args)
    evaluation_pipeline = EvaluationPipeline(store)
    result = evaluation_pipeline.evaluate_all()
    _json_out(
        {
            "evaluated": result.evaluated_count,
            "skipped_no_outcome": result.skipped_count,
            "missing_records": result.missing_records,
        }
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Validate dataset integrity."""
    store = _build_store(args)
    report: ValidationReport = validate_dataset(store)
    _json_out(report.to_dict())
    return 0 if report.is_valid else 1


def cmd_stats(args: argparse.Namespace) -> int:
    """Print dataset summary, coverage, outcomes, and metrics."""
    store = _build_store(args)
    builder = DatasetReportBuilder(
        store,
        engine_version=ENGINE_VERSION,
        benchmark_version=getattr(args, "benchmark_version", None),
    )
    report = builder.build()
    _json_out(report)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Write a JSON dataset report to a file."""
    store = _build_store(args)
    builder = DatasetReportBuilder(
        store,
        engine_version=ENGINE_VERSION,
        benchmark_version=getattr(args, "benchmark_version", None),
    )
    report = builder.build()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(path)
    return 0


def _build_engine() -> Any:
    """Build a PredictronEngine instance."""
    from predictron_engine.engine import PredictronEngine

    return PredictronEngine()


def _infer_source(path: str) -> str:
    """Infer the import source adapter from the file extension."""
    lower = path.lower()
    if lower.endswith(".csv"):
        return "csv_file"
    if lower.endswith(".json"):
        return "json_file"
    return "json_file"


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset store root directory",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="predictron-dataset",
        description="Historical Startup Dataset Builder CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # import
    p_import = sub.add_parser("import", help="Import records/outcomes from a data file")
    p_import.add_argument("file", type=str, help="Path to the data file (JSON/CSV)")
    _add_common_args(p_import)
    p_import.add_argument("--source", type=str, default=None)
    p_import.set_defaults(func=cmd_import)

    # dedup
    p_dedup = sub.add_parser("dedup", help="Report duplicate startup records")
    _add_common_args(p_dedup)
    p_dedup.set_defaults(func=cmd_dedup)

    # statistics
    p_stats = sub.add_parser(
        "statistics", help="Print dataset statistics (sectors, stages, years, countries)"
    )
    _add_common_args(p_stats)
    p_stats.set_defaults(func=cmd_statistics)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Run the engine over records")
    _add_common_args(p_analyze)
    p_analyze.add_argument("--record-id", type=str, default=None)
    p_analyze.add_argument("--benchmark-version", type=str, default=None)
    p_analyze.set_defaults(func=cmd_analyze)

    # evaluate
    p_evaluate = sub.add_parser("evaluate", help="Store evaluations for outcomes")
    _add_common_args(p_evaluate)
    p_evaluate.set_defaults(func=cmd_evaluate)

    # verify
    p_verify = sub.add_parser("verify", help="Validate dataset integrity")
    _add_common_args(p_verify)
    p_verify.set_defaults(func=cmd_verify)

    # stats
    p_stats = sub.add_parser("stats", help="Print dataset summary and metrics")
    _add_common_args(p_stats)
    p_stats.add_argument("--benchmark-version", type=str, default=None)
    p_stats.set_defaults(func=cmd_stats)

    # export
    p_export = sub.add_parser("export", help="Write a JSON dataset report")
    p_export.add_argument("--output", type=str, required=True)
    _add_common_args(p_export)
    p_export.add_argument("--benchmark-version", type=str, default=None)
    p_export.set_defaults(func=cmd_export)

    # acquire (main command)
    p_acquire = sub.add_parser("acquire", help="Acquire data from public sources")
    _add_common_args(p_acquire)
    p_acquire.add_argument("--source", type=str, default=None)
    p_acquire.add_argument("--file", type=str, default=None)
    p_acquire.add_argument("--directory", type=str, default=None)
    p_acquire.add_argument("--resume", action="store_true", default=False)
    p_acquire.add_argument("--since", type=str, default=None)
    p_acquire.add_argument("--limit", type=int, default=None)
    p_acquire.add_argument("--dry-run", action="store_true", default=False)
    p_acquire.set_defaults(func=cmd_acquire)

    # acquire status
    p_acq_status = sub.add_parser("acquire-status", help="Show acquisition status")
    _add_common_args(p_acq_status)
    p_acq_status.set_defaults(func=cmd_acquire_status)

    # acquire sources
    p_acq_sources = sub.add_parser("acquire-sources", help="List available source connectors")
    p_acq_sources.set_defaults(func=cmd_acquire_sources)

    # acquire schedule
    p_acq_sched = sub.add_parser("acquire-schedule", help="Manage import schedules")
    _add_common_args(p_acq_sched)
    p_acq_sched.add_argument(
        "schedule_action",
        type=str,
        choices=["list", "add", "remove"],
        help="Schedule action",
    )
    p_acq_sched.add_argument("--schedule-source", type=str, default=None)
    p_acq_sched.add_argument("--frequency", type=str, default="weekly")
    p_acq_sched.set_defaults(func=cmd_acquire_schedule)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(Any, args.func)
    return cast(int, handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
