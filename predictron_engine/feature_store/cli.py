"""Feature Store CLI.

Commands:
    feature-build      Compute features for records in the dataset store
    feature-report     Generate a feature store report
    feature-validate   Validate computed features
    feature-history    Show feature value history for a company
    feature-export     Export feature store data
    feature-registry   List registered features
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from predictron_engine.feature_store.engine import FeatureEngine
from predictron_engine.feature_store.features import ALL_FEATURES
from predictron_engine.feature_store.registry import FeatureRegistry
from predictron_engine.feature_store.reports import FeatureReportBuilder
from predictron_engine.feature_store.store import FeatureStore
from predictron_engine.feature_store.validation import FeatureValidator


def _build_registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    registry.register_all(ALL_FEATURES)
    return registry


def _build_engine() -> FeatureEngine:
    return FeatureEngine(_build_registry())


def _dataset_store_path(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "data_dir", "data/dataset"))


def _feature_store_path(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "feature_dir", "data/features"))


def _load_dataset_store(args: argparse.Namespace) -> Any:
    from predictron_engine.dataset.store import DatasetStore
    ds = DatasetStore(_dataset_store_path(args))
    ds.initialize()
    return ds


def _load_feature_store(args: argparse.Namespace) -> FeatureStore:
    fs = FeatureStore(_feature_store_path(args))
    fs.initialize()
    return fs


# ---- feature-build ----

def cmd_feature_build(args: argparse.Namespace) -> None:
    """Compute features for all records in the dataset store."""
    ds = _load_dataset_store(args)
    fs = _load_feature_store(args)
    engine = _build_engine()

    record_ids = ds.list_records()
    if not record_ids:
        print("No records found in dataset store.")
        return

    count = 0
    for rid in record_ids:
        record = ds.load_record(rid)
        if record is None:
            continue
        timeline = ds.load_timeline(rid)
        feature_set = engine.build_company_features(
            record, timeline=timeline, as_of=datetime.now(UTC),
        )
        fs.save_company_features(feature_set)
        count += 1
        if getattr(args, "verbose", False):
            print(f"  Built {feature_set.feature_count()} features for {rid}")

    print(f"Built features for {count} companies.")


# ---- feature-report ----

def cmd_feature_report(args: argparse.Namespace) -> None:
    """Generate a feature store report."""
    fs = _load_feature_store(args)
    registry = _build_registry()
    builder = FeatureReportBuilder(registry)

    feature_sets = []
    for company_id in fs.list_companies():
        fs_set = fs.load_company_features(company_id)
        if fs_set:
            feature_sets.append(fs_set)

    report = builder.build_report(feature_sets=feature_sets)

    output = getattr(args, "output", None)
    if output:
        Path(output).write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Report written to {output}")
    else:
        print(json.dumps(report.to_dict(), indent=2, default=str))


# ---- feature-validate ----

def cmd_feature_validate(args: argparse.Namespace) -> None:
    """Validate computed features."""
    fs = _load_feature_store(args)
    registry = _build_registry()
    validator = FeatureValidator(registry)

    company_id = getattr(args, "company_id", None)
    if company_id:
        feature_sets = []
        fs_set = fs.load_company_features(company_id)
        if fs_set:
            feature_sets.append(fs_set)
    else:
        feature_sets = []
        for cid in fs.list_companies():
            fs_set = fs.load_company_features(cid)
            if fs_set:
                feature_sets.append(fs_set)

    reports = validator.validate_all(feature_sets)
    for report in reports:
        data = report.to_dict()
        print(json.dumps(data, indent=2, default=str))

    total_errors = sum(r.error_count for r in reports)
    if total_errors > 0:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print(f"\nAll {len(reports)} companies validated successfully.")


# ---- feature-history ----

def cmd_feature_history(args: argparse.Namespace) -> None:
    """Show feature value history for a company."""
    fs = _load_feature_store(args)
    company_id = getattr(args, "company_id", None)
    feature_id = getattr(args, "feature_id", None)

    if not company_id:
        print("Error: --company-id is required.")
        sys.exit(1)

    if feature_id:
        snapshots = fs.get_feature_history(company_id, feature_id)
        for snap in snapshots:
            print(f"  {snap.computed_at.isoformat()}: {snap.value} "
                  f"(status={snap.status.value})")
    else:
        history = fs.get_company_history(company_id)
        for fs_set in history:
            print(f"\n--- {fs_set.built_at.isoformat()} ---")
            for fid, snap in sorted(fs_set.features.items()):
                print(f"  {fid}: {snap.value} (status={snap.status.value})")


# ---- feature-export ----

def cmd_feature_export(args: argparse.Namespace) -> None:
    """Export feature store data."""
    fs = _load_feature_store(args)
    output = getattr(args, "output", None) or "feature_export.json"

    data = fs.export_all()
    Path(output).write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Exported to {output}")


# ---- feature-registry ----

def cmd_feature_registry(args: argparse.Namespace) -> None:
    """List registered features."""
    registry = _build_registry()

    category = getattr(args, "category", None)
    if category:
        from predictron_engine.feature_store.models import FeatureCategory
        try:
            cat = FeatureCategory(category)
            definitions = registry.list_by_category(cat)
        except ValueError:
            print(f"Unknown category: {category}")
            sys.exit(1)
    else:
        definitions = registry.list_all()

    for defn in definitions:
        print(f"  {defn.feature_id:30s} [{defn.category.value:15s}] {defn.feature_name}")
        print(f"    {defn.description}")
        if defn.dependencies:
            print(f"    Dependencies: {', '.join(defn.dependencies)}")
        print()

    print(f"Total: {len(definitions)} features")


# ---- Main ----

def main(argv: list[str] | None = None) -> None:
    """Feature Store CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="predictron-feature",
        description="Venture Intelligence Feature Store CLI",
    )
    parser.add_argument(
        "--data-dir", default="data/dataset",
        help="Dataset store directory",
    )
    parser.add_argument(
        "--feature-dir", default="data/features",
        help="Feature store directory",
    )

    sub = parser.add_subparsers(dest="command")

    # feature-build
    build_p = sub.add_parser("feature-build", help="Compute features")
    build_p.add_argument("-v", "--verbose", action="store_true")

    # feature-report
    report_p = sub.add_parser("feature-report", help="Generate report")
    report_p.add_argument("-o", "--output", help="Output file path")

    # feature-validate
    validate_p = sub.add_parser("feature-validate", help="Validate features")
    validate_p.add_argument("--company-id", help="Validate single company")

    # feature-history
    history_p = sub.add_parser("feature-history", help="Feature history")
    history_p.add_argument("--company-id", help="Company ID")
    history_p.add_argument("--feature-id", help="Feature ID")

    # feature-export
    export_p = sub.add_parser("feature-export", help="Export features")
    export_p.add_argument("-o", "--output", help="Output file path")

    # feature-registry
    registry_p = sub.add_parser("feature-registry", help="List features")
    registry_p.add_argument("--category", help="Filter by category")

    args = parser.parse_args(argv)

    if args.command == "feature-build":
        cmd_feature_build(args)
    elif args.command == "feature-report":
        cmd_feature_report(args)
    elif args.command == "feature-validate":
        cmd_feature_validate(args)
    elif args.command == "feature-history":
        cmd_feature_history(args)
    elif args.command == "feature-export":
        cmd_feature_export(args)
    elif args.command == "feature-registry":
        cmd_feature_registry(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
