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
    """Import records/outcomes from a JSON file into a store."""
    store = _build_store(args)
    source_name = getattr(args, "source", None) or "json_file"
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
    p_import = sub.add_parser("import", help="Import records/outcomes from JSON")
    p_import.add_argument("file", type=str, help="Path to the JSON data file")
    _add_common_args(p_import)
    p_import.add_argument("--source", type=str, default=None)
    p_import.set_defaults(func=cmd_import)

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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(Any, args.func)
    return cast(int, handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
