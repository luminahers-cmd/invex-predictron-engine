"""Cohort builder CLI (Milestone V1.4).

Subcommands
-----------
* ``make-manifest`` — generate a deterministic starter cohort manifest from
  the committed benchmark cases (``benchmarks/startup_cases/cases.py``).
* ``build``         — build a golden dataset from a cohort manifest, running
  the current engine under per-company time-scoped evidence (look-ahead
  guard) to record pinned historical predictions, then dump the dataset.

Examples
--------
    python scripts/build_cohort.py make-manifest \\
        --out benchmarks/cohort_sources/starter_v1.json --overwrite
    python scripts/build_cohort.py build \\
        --manifest benchmarks/cohort_sources/starter_v1.json \\
        --out benchmarks/golden_datasets/starter_v1.json

Both paths are deterministic: identical inputs always produce identical
outputs (modulo the wall-clock fields excluded from every hash).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.cohort.build import CohortBuildError, build_cohort_dataset  # noqa: E402
from benchmarks.cohort.manifest import (  # noqa: E402
    CohortManifest,
    ManifestError,
    ManifestSourceRecord,
    dump_manifest,
    load_manifest,
)
from benchmarks.ground_truth.schema import (  # noqa: E402
    OutcomeEvent,
    OutcomeEventKind,
    StartupStatus,
)
from benchmarks.ground_truth_eval.dataset import dump_golden_dataset  # noqa: E402

# Fixed anchors for the deterministic starter manifest.  The committed
# offline corpora normalize every document's fetched_at to 2026-01-01 UTC
# (see benchmarks/gen_case_corpora.py), so the analysis is pinned there.
STARTER_ANALYSIS_TIMESTAMP = datetime(2026, 1, 1, tzinfo=UTC)
STARTER_VERIFICATION_DATE = date(2026, 6, 30)
STARTER_CREATED_AT = datetime(2026, 9, 20, tzinfo=UTC)

# Illustrative best-known outcomes mirroring benchmarks/golden_datasets/
# example_v1.json (fictional labels explicitly flagged is_example on build).
# status -> (status, exit_value_usd, arr_usd)
_COHORT_OUTCOMES: dict[str, tuple[str, float | None, float | None]] = {
    "b2b_saas": ("acquired", 120_000_000, None),
    "healthcare_ai": ("operating", None, 5_000_000),
    "fintech": ("shutdown", None, None),
    "devtools": ("operating", None, None),
    "marketplace": ("acquired", 200_000_000, None),
    "consumer_app": ("shutdown", None, None),
    "climate_tech": ("operating", None, 3_000_000),
    "robotics": ("shutdown", None, None),
    "enterprise_software": ("acquired", 80_000_000, None),
    "ai_infrastructure": ("operating", None, None),
    "deep_tech": ("shutdown", None, None),
    "edtech": ("acquired", 40_000_000, None),
    "healthtech_device": ("operating", None, 2_000_000),
}


def build_starter_manifest() -> CohortManifest:
    """Build the deterministic starter manifest from the benchmark cases."""
    from benchmarks.startup_cases.cases import BENCHMARK_CASES

    records: list[ManifestSourceRecord] = []
    for case in sorted(BENCHMARK_CASES, key=lambda c: c["id"]):
        case_id = str(case["id"])
        request = dict(case["request"])
        metadata = case["metadata"]
        outcome = None
        if case_id in _COHORT_OUTCOMES:
            outcome = _outcome_block(*_COHORT_OUTCOMES[case_id])
        records.append(
            ManifestSourceRecord(
                company_id=case_id,
                company_name=str(request["startup_name"]),
                description=str(request["description"]),
                website=str(request["website"]) if request.get("website") else None,
                sector=str(metadata.industry_category),
                stage=str(metadata.company_stage),
                evidence_corpus=case_id,
                analysis_timestamp=STARTER_ANALYSIS_TIMESTAMP,
                outcome=outcome,
                metadata={
                    "sources": [
                        "benchmarks/startup_cases/cases.py",
                        "benchmarks/offline_evidence",
                    ]
                },
            )
        )
    return CohortManifest(
        dataset_name="predictron_cohort_starter",
        benchmark_version="1.0",
        created_at=STARTER_CREATED_AT,
        evaluation_horizon_days=None,
        source_records=records,
        notes=(
            "Starter cohort: real analysis-time inputs from the committed "
            "benchmark cases and real committed evidence corpora, with the "
            "analysis pinned to the corpora's fixed retrieval instant. "
            "Outcomes are illustrative and NOT independently verified."
        ),
    )


def _outcome_block(status: str, exit_value_usd: float | None, arr_usd: float | None) -> Any:
    from benchmarks.cohort.manifest import ManifestOutcome

    events: list[OutcomeEvent] = []
    if status == "acquired":
        events.append(
            OutcomeEvent(
                kind=OutcomeEventKind.ACQUISITION,
                occurred_at=date(2026, 3, 15),
                value_currency_usd=exit_value_usd,
                description="Illustrative placeholder outcome",
                sources=["benchmarks/startup_cases/cases.py"],
            )
        )
    elif status == "shutdown":
        events.append(
            OutcomeEvent(
                kind=OutcomeEventKind.SHUTDOWN,
                occurred_at=date(2026, 3, 1),
                description="Illustrative placeholder outcome",
                sources=["benchmarks/startup_cases/cases.py"],
            )
        )
    elif arr_usd:
        events.append(
            OutcomeEvent(
                kind=OutcomeEventKind.ARR_MILESTONE,
                occurred_at=date(2026, 4, 1),
                amount_nominal_units=arr_usd,
                description="Illustrative placeholder outcome",
                sources=["benchmarks/startup_cases/cases.py"],
            )
        )
    return ManifestOutcome(
        status=StartupStatus(status),
        verification_date=STARTER_VERIFICATION_DATE,
        outcome_events=events,
        exit_value_usd=exit_value_usd,
        sources=["benchmarks/startup_cases/cases.py"],
        verified=False,
        notes=(
            "Illustrative placeholder outcome — not independently verified. "
            "Set verified=true with sources to treat as real ground truth."
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_cohort",
        description="Cohort builder — manifests to golden datasets (V1.4).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make-manifest", help="Generate the starter cohort manifest")
    make.add_argument("--out", type=Path, required=True, help="Output manifest JSON")
    make.add_argument("--overwrite", action="store_true", help="Replace an existing manifest")
    make.set_defaults(func=_cmd_make_manifest)

    build = sub.add_parser("build", help="Build a golden dataset from a manifest")
    build.add_argument("--manifest", type=Path, required=True, help="Cohort manifest JSON")
    build.add_argument("--out", type=Path, default=None, help="Output golden dataset JSON")
    build.add_argument("--overwrite", action="store_true", help="Replace an existing dataset")
    build.add_argument(
        "--allow-errors",
        action="store_true",
        help="Write the dataset even when validation reports errors",
    )
    build.set_defaults(func=_cmd_build)
    return parser


def _cmd_make_manifest(args: argparse.Namespace) -> None:
    manifest = build_starter_manifest()
    path = dump_manifest(manifest, args.out, overwrite=args.overwrite)
    counted = sum(1 for r in manifest.source_records if r.outcome is not None)
    pending = len(manifest.source_records) - counted
    print(f"wrote {path}")
    print(f"records={len(manifest.source_records)} with_outcome={counted} pending={pending}")


def _cmd_build(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    try:
        build = build_cohort_dataset(manifest)
    except (CohortBuildError, ManifestError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    report = build.to_report()
    validation = report["validation"] or {}
    errors = int(validation.get("error_count", 0))
    if errors and not args.allow_errors:
        for issue in validation.get("issues", []):
            if issue.get("severity") == "error":
                label = issue.get("company_id") or "dataset"
                print(f"validation error [{label}]: {issue['message']}", file=sys.stderr)
        print(
            f"aborting: dataset {report['dataset_name']} has {errors} validation error(s); "
            "fix the manifest or pass --allow-errors",
            file=sys.stderr,
        )
        raise SystemExit(1)

    golden_dir = REPO_ROOT / "benchmarks" / "golden_datasets"
    out = args.out or golden_dir / f"{report['dataset_name']}.json"
    dump_golden_dataset(build.dataset, out, overwrite=args.overwrite)

    print(f"wrote {out}")
    print(f"dataset:      {report['dataset_name']} (v{report['benchmark_version']})")
    print(f"manifest_hash:{report['manifest_hash']}")
    print(f"dataset_hash: {report['dataset_hash']}")
    print(f"entries:      {report['entries_built']} built, "
          f"{len(report['pending_ground_truth'])} pending, "
          f"{len(report['failed'])} failed")
    if report["lookahead_blocked_company_ids"]:
        print(f"look-ahead blocked: {','.join(report['lookahead_blocked_company_ids'])}")
    if errors:
        print(f"warnings:     {validation.get('warning_count', 0)}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        args.func(args)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
