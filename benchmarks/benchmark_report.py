"""Benchmark report — generates structured reports and regression diffs.

Produces human-readable and machine-readable reports from benchmark
results. Supports cross-version regression comparison by diffing
snapshots from different engine versions.

v0.11.0 enhancements:
  - Integrated benchmark metrics (score stats, confidence calibration, etc.)
  - Validation findings per case
  - Improved diff formatting with severity indicators
  - Coverage analysis by industry and stage
  - Most-affected cases highlight

Usage:
    python -m benchmarks.benchmark_report
    python -m benchmarks.benchmark_report --version 0.6.5
    python -m benchmarks.benchmark_report --diff 0.6.5 0.7.0
    python -m benchmarks.benchmark_report --output report.md
    python -m benchmarks.benchmark_report --metrics-only
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.benchmark_metrics import BenchmarkMetrics
from benchmarks.benchmark_runner import (
    EXPECTED_OUTPUTS_DIR,
    CaseResult,
    load_snapshot,
    run_benchmark,
    save_snapshot,
)
from benchmarks.benchmark_validator import BenchmarkValidator
from benchmarks.startup_cases.cases import (
    BENCHMARK_CASES,
    get_industry_coverage,
    get_stage_coverage,
)
from predictron_engine.engine import ENGINE_VERSION


@dataclass
class VersionDiff:
    """Comparison between two engine version snapshots."""

    version_a: str
    version_b: str
    case_id: str
    case_label: str
    overall_score_delta: float
    overall_confidence_delta: float
    feature_changes: dict[str, Any]
    score_changes: dict[str, Any]
    observation_count_delta: int
    recommendation_count_delta: int
    processing_time_delta_ms: float

    @property
    def has_changes(self) -> bool:
        return (
            abs(self.overall_score_delta) > 0.01
            or abs(self.overall_confidence_delta) > 0.01
            or self.feature_changes
            or self.score_changes
            or self.observation_count_delta != 0
            or self.recommendation_count_delta != 0
        )

    @property
    def severity(self) -> str:
        """Classify the change severity."""
        if not self.has_changes:
            return "unchanged"
        score_impact = abs(self.overall_score_delta)
        if score_impact > 10.0:
            return "major"
        if score_impact > 3.0:
            return "moderate"
        if score_impact > 0.5:
            return "minor"
        return "negligible"


def generate_text_report(results: list[CaseResult]) -> str:
    """Generate a human-readable text report from benchmark results."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("PREDICTRON ENGINE BENCHMARK REPORT")
    lines.append(f"Engine Version: {ENGINE_VERSION}")
    lines.append(f"Cases Evaluated: {len(results)}")
    lines.append("=" * 70)

    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed

    lines.append("\nSUMMARY")
    lines.append(f"  Passed: {passed}")
    lines.append(f"  Failed: {failed}")

    if results:
        successful = [r for r in results if r.success and r.report]
        if successful:
            avg_score = sum(r.report.overall_score for r in successful) / len(successful)
            avg_conf = sum(r.report.overall_confidence for r in successful) / len(successful)
            avg_time = sum(r.processing_time_ms for r in successful) / len(successful)
            lines.append(f"  Avg Overall Score: {avg_score:.1f}")
            lines.append(f"  Avg Confidence: {avg_conf:.2f}")
            lines.append(f"  Avg Processing Time: {avg_time:.1f}ms")

    lines.append(f"\n{'-' * 70}")
    lines.append("COVERAGE ANALYSIS")
    lines.append(f"{'-' * 70}")

    industry_cov = get_industry_coverage()
    lines.append("\n  By Industry:")
    for cat, count in sorted(industry_cov.items()):
        lines.append(f"    {cat:<20} {count} case(s)")

    stage_cov = get_stage_coverage()
    lines.append("\n  By Stage:")
    for stage, count in sorted(stage_cov.items()):
        lines.append(f"    {stage:<20} {count} case(s)")

    lines.append(f"\n{'-' * 70}")
    lines.append("PER-CASE RESULTS")
    lines.append(f"{'-' * 70}")

    for result in results:
        case_meta = _get_case_metadata(result.case_id)
        lines.append(f"\n  Case: {result.case_id}")
        lines.append(f"  Label: {result.case_label}")
        if case_meta:
            lines.append(
                f"  Category: {case_meta['industry_category']} | "
                f"Stage: {case_meta['company_stage']}"
            )
        lines.append(f"  Status: {'PASS' if result.success else 'FAIL'}")

        if not result.success:
            lines.append(f"  Error: {result.error}")
            continue

        r = result.report

        lines.append("\n  Extracted Features:")
        lines.append(f"    Industry: {r.features.industry}")
        lines.append(f"    Sub-Industry: {r.features.sub_industry}")
        lines.append(f"    Business Model: {r.features.business_model}")
        lines.append(f"    Customer Type: {r.features.customer_type}")
        lines.append(f"    Geography: {r.features.geography}")
        lines.append(f"    Funding Stage: {r.features.funding_stage}")
        lines.append(f"    Has Revenue: {r.features.has_revenue}")
        lines.append(f"    Keywords: {r.features.key_keywords}")
        lines.append(f"    Tech Stack: {r.features.technology_stack}")
        lines.append(f"    Data Completeness: {r.features.data_completeness:.0%}")

        if r.evidence:
            lines.append(f"\n  Evidence ({len(r.evidence)} items):")
            for i, e in enumerate(r.evidence, 1):
                lines.append(f"    {i}. [{e.domain}/{e.category}] {e.statement}")
                lines.append(f"       Source: {e.source} (relevance: {e.relevance_score:.2f})")

        if r.observations:
            lines.append(f"\n  Observations ({len(r.observations)} items):")
            for i, o in enumerate(r.observations, 1):
                lines.append(
                    f"    {i}. [{o.dimension}] {o.statement}"
                )
                lines.append(
                    f"       Rule: {o.source_rule} | "
                    f"Confidence: {o.confidence:.2f} | "
                    f"Importance: {o.importance:.2f}"
                )

        if r.dimension_assessments:
            lines.append(f"\n  Dimension Assessments ({len(r.dimension_assessments)}):")
            for a in r.dimension_assessments:
                lines.append(f"    [{a.dimension}]")
                lines.append(f"      Summary: {a.summary}")
                lines.append(f"      Score: {a.score}")
                lines.append(f"      Confidence: {a.confidence:.2f}")

        if r.scores:
            lines.append("\n  Scores:")
            for s in r.scores:
                lines.append(f"    {s.dimension}: {s.score:.1f}")
                if s.rationale:
                    lines.append(f"      Rationale: {s.rationale}")
            lines.append(f"    Overall: {r.overall_score:.1f}")

        if r.recommendations:
            lines.append(f"\n  Recommendations ({len(r.recommendations)}):")
            for i, rec in enumerate(r.recommendations, 1):
                lines.append(f"    {i}. [{rec.priority.upper()}] {rec.category}")
                lines.append(f"       Action: {rec.action}")
                if rec.title:
                    lines.append(f"       Title: {rec.title}")

        if r.confidence:
            lines.append("\n  Confidence Assessments:")
            for c in r.confidence:
                lines.append(
                    f"    {c.dimension}: {c.confidence:.2f} "
                    f"(data completeness: {c.data_completeness:.0%})"
                )
            lines.append(f"    Overall: {r.overall_confidence:.2f}")

        lines.append(f"\n  Processing Time: {result.processing_time_ms:.1f}ms")

        if result.stage_timings:
            lines.append("  Stage Breakdown:")
            for stage, ms in result.stage_timings.items():
                if stage != "total":
                    lines.append(f"    {stage}: {ms:.1f}ms")

    lines.append(f"\n{'=' * 70}")
    lines.append("END OF REPORT")
    lines.append(f"{'=' * 70}")

    return "\n".join(lines)


def generate_metrics_report(results: list[CaseResult]) -> str:
    """Generate a metrics-focused report from benchmark results."""
    metrics_engine = BenchmarkMetrics()
    report = metrics_engine.compute(results)

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("PREDICTRON ENGINE BENCHMARK METRICS REPORT")
    lines.append(f"Engine Version: {ENGINE_VERSION}")
    lines.append(f"Cases: {report.successful_cases}/{report.total_cases} successful")
    lines.append("=" * 70)

    for metric in report.metrics:
        lines.append(f"\n  {metric.name}")
        lines.append(f"    Value: {metric.value}")
        lines.append(f"    {metric.description}")
        if metric.details:
            for key, val in metric.details.items():
                lines.append(f"    {key}: {val}")

    lines.append(f"\n{'=' * 70}")
    lines.append("END OF METRICS REPORT")
    lines.append(f"{'=' * 70}")

    return "\n".join(lines)


def generate_validation_report(results: list[CaseResult]) -> str:
    """Generate a validation-focused report from benchmark results."""
    validator = BenchmarkValidator()
    validations = validator.validate_all(BENCHMARK_CASES, results)

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("PREDICTRON ENGINE BENCHMARK VALIDATION REPORT")
    lines.append(f"Engine Version: {ENGINE_VERSION}")
    lines.append(f"Cases Validated: {len(validations)}")
    lines.append("=" * 70)

    total_pass = sum(1 for v in validations if v.passed)
    total_fail = len(validations) - total_pass
    total_findings = sum(
        v.pass_count + v.warn_count + v.fail_count + v.info_count
        for v in validations
    )
    total_warnings = sum(v.warn_count for v in validations)
    total_errors = sum(v.fail_count for v in validations)

    lines.append("\nSUMMARY")
    lines.append(f"  Passed: {total_pass}/{len(validations)}")
    lines.append(f"  Failed: {total_fail}/{len(validations)}")
    lines.append(f"  Total Findings: {total_findings}")
    lines.append(f"  Warnings: {total_warnings}")
    lines.append(f"  Errors: {total_errors}")

    lines.append(f"\n{'-' * 70}")
    lines.append("PER-CASE VALIDATION")
    lines.append(f"{'-' * 70}")

    for v in validations:
        status = "PASS" if v.passed else "FAIL"
        lines.append(f"\n  [{status}] {v.case_id} ({v.case_label})")
        lines.append(
            f"    Pass: {v.pass_count} | Warn: {v.warn_count} | "
            f"Fail: {v.fail_count} | Info: {v.info_count}"
        )

        for finding in v.findings:
            icon = {
                "pass": "+",
                "warn": "!",
                "fail": "X",
                "info": "i",
            }.get(finding.severity.value, "?")
            lines.append(f"    [{icon}] {finding.field_name}: {finding.message or 'OK'}")

    lines.append(f"\n{'=' * 70}")
    lines.append("END OF VALIDATION REPORT")
    lines.append(f"{'=' * 70}")

    return "\n".join(lines)


def generate_regression_diff(
    snapshot_a: dict[str, Any],
    snapshot_b: dict[str, Any],
) -> list[VersionDiff]:
    """Compare two version snapshots and produce regression diffs.

    Args:
        snapshot_a: The baseline snapshot (older version).
        snapshot_b: The comparison snapshot (newer version).

    Returns:
        List of VersionDiff objects, one per case that exists in both.
    """
    results_a = {r["case_id"]: r for r in snapshot_a.get("results", [])}
    results_b = {r["case_id"]: r for r in snapshot_b.get("results", [])}

    diffs: list[VersionDiff] = []

    for case_id in results_a:
        if case_id not in results_b:
            continue

        ra = results_a[case_id]
        rb = results_b[case_id]

        label = ""
        for case in BENCHMARK_CASES:
            if case["id"] == case_id:
                label = case["label"]
                break

        score_a = ra.get("overall_score", 0)
        score_b = rb.get("overall_score", 0)
        conf_a = ra.get("overall_confidence", 0)
        conf_b = rb.get("overall_confidence", 0)

        fa = ra.get("extracted_features", {})
        fb = rb.get("extracted_features", {})
        feature_changes: dict[str, Any] = {}
        for key in set(list(fa.keys()) + list(fb.keys())):
            if fa.get(key) != fb.get(key):
                feature_changes[key] = {"from": fa.get(key), "to": fb.get(key)}

        sa = {s["dimension"]: s["score"] for s in ra.get("scores", [])}
        sb = {s["dimension"]: s["score"] for s in rb.get("scores", [])}
        score_changes: dict[str, Any] = {}
        for dim in set(list(sa.keys()) + list(sb.keys())):
            va = sa.get(dim, 0)
            vb = sb.get(dim, 0)
            if abs(va - vb) > 0.01:
                score_changes[dim] = {"from": va, "to": vb, "delta": round(vb - va, 2)}

        obs_a = len(ra.get("observations", []))
        obs_b = len(rb.get("observations", []))
        rec_a = len(ra.get("recommendations", []))
        rec_b = len(rb.get("recommendations", []))
        time_a = ra.get("processing_time_ms", 0)
        time_b = rb.get("processing_time_ms", 0)

        diff = VersionDiff(
            version_a=snapshot_a.get("engine_version", "?"),
            version_b=snapshot_b.get("engine_version", "?"),
            case_id=case_id,
            case_label=label,
            overall_score_delta=round(score_b - score_a, 2),
            overall_confidence_delta=round(conf_b - conf_a, 4),
            feature_changes=feature_changes,
            score_changes=score_changes,
            observation_count_delta=obs_b - obs_a,
            recommendation_count_delta=rec_b - rec_a,
            processing_time_delta_ms=round(time_b - time_a, 2),
        )
        diffs.append(diff)

    return diffs


def format_regression_diff(diffs: list[VersionDiff]) -> str:
    """Format regression diffs as a human-readable report."""
    if not diffs:
        return "No diff data available. Run benchmarks for both versions first."

    va = diffs[0].version_a
    vb = diffs[0].version_b

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("REGRESSION COMPARISON REPORT")
    lines.append(f"Baseline: v{va} -> Comparison: v{vb}")
    lines.append("=" * 70)

    improved = [d for d in diffs if d.overall_score_delta > 0.01]
    regressed = [d for d in diffs if d.overall_score_delta < -0.01]
    unchanged = [d for d in diffs if abs(d.overall_score_delta) <= 0.01]

    lines.append("\nSUMMARY")
    lines.append(f"  Improved: {len(improved)} cases")
    lines.append(f"  Regressed: {len(regressed)} cases")
    lines.append(f"  Unchanged: {len(unchanged)} cases")

    if improved:
        lines.append("\n  Improved cases:")
        for d in sorted(improved, key=lambda x: x.overall_score_delta, reverse=True):
            lines.append(
                f"    [+] {d.case_id}: score {d.overall_score_delta:+.1f}, "
                f"confidence {d.overall_confidence_delta:+.4f}"
            )

    if regressed:
        lines.append("\n  Regressed cases:")
        for d in sorted(regressed, key=lambda x: x.overall_score_delta):
            lines.append(
                f"    [-] {d.case_id}: score {d.overall_score_delta:+.1f}, "
                f"confidence {d.overall_confidence_delta:+.4f}"
            )

    most_affected = sorted(
        [d for d in diffs if d.has_changes],
        key=lambda d: abs(d.overall_score_delta),
        reverse=True,
    )
    if most_affected:
        lines.append("\n  Most affected cases (by score delta):")
        for d in most_affected[:5]:
            lines.append(
                f"    {d.case_id}: {d.overall_score_delta:+.1f} ({d.severity})"
            )

    lines.append(f"\n{'-' * 70}")
    lines.append("PER-CASE DETAILS")
    lines.append(f"{'-' * 70}")

    for diff in diffs:
        if not diff.has_changes:
            continue

        severity_marker = {
            "major": "!!!",
            "moderate": "!! ",
            "minor": "!  ",
            "negligible": ".  ",
            "unchanged": "   ",
        }.get(diff.severity, "   ")

        lines.append(
            f"\n  [{severity_marker}] {diff.case_id} ({diff.case_label})"
        )

        if diff.overall_score_delta != 0:
            direction = "+" if diff.overall_score_delta > 0 else "-"
            lines.append(
                f"    Overall Score: {diff.overall_score_delta:+.1f} {direction}"
            )
        if diff.overall_confidence_delta != 0:
            lines.append(
                f"    Overall Confidence: {diff.overall_confidence_delta:+.4f}"
            )
        if diff.observation_count_delta != 0:
            lines.append(f"    Observations: {diff.observation_count_delta:+d}")
        if diff.recommendation_count_delta != 0:
            lines.append(f"    Recommendations: {diff.recommendation_count_delta:+d}")
        if abs(diff.processing_time_delta_ms) > 0.1:
            lines.append(f"    Processing Time: {diff.processing_time_delta_ms:+.1f}ms")

        if diff.feature_changes:
            lines.append("    Feature Changes:")
            for key, change in diff.feature_changes.items():
                lines.append(f"      {key}: {change['from']} -> {change['to']}")

        if diff.score_changes:
            lines.append("    Score Changes:")
            for dim, change in diff.score_changes.items():
                lines.append(
                    f"      {dim}: {change['from']:.1f} -> {change['to']:.1f} "
                    f"({change['delta']:+.1f})"
                )

    lines.append(f"\n{'=' * 70}")
    lines.append("END OF REGRESSION REPORT")
    lines.append(f"{'=' * 70}")

    return "\n".join(lines)


def generate_regression_diff_summary(
    snapshot_a: dict[str, Any],
    snapshot_b: dict[str, Any],
) -> str:
    """Generate a concise regression diff summary."""
    diffs = generate_regression_diff(snapshot_a, snapshot_b)
    if not diffs:
        return "No diff data available."

    improved = [d for d in diffs if d.overall_score_delta > 0.01]
    regressed = [d for d in diffs if d.overall_score_delta < -0.01]
    unchanged = [d for d in diffs if abs(d.overall_score_delta) <= 0.01]

    total_delta = sum(d.overall_score_delta for d in diffs)
    avg_delta = total_delta / len(diffs) if diffs else 0.0

    lines: list[str] = []
    ver_a = snapshot_a.get("engine_version", "?")
    ver_b = snapshot_b.get("engine_version", "?")
    lines.append(f"v{ver_a} -> v{ver_b}")
    lines.append(
        f"Improved: {len(improved)} | Regressed: {len(regressed)} | "
        f"Unchanged: {len(unchanged)} | Avg delta: {avg_delta:+.2f}"
    )

    if regressed:
        worst = min(regressed, key=lambda d: d.overall_score_delta)
        lines.append(f"Worst regression: {worst.case_id} ({worst.overall_score_delta:+.1f})")

    return "\n".join(lines)


def _get_case_metadata(case_id: str) -> dict[str, Any] | None:
    """Look up metadata for a benchmark case."""
    for case in BENCHMARK_CASES:
        if case["id"] == case_id:
            meta = case.get("metadata")
            if meta is not None:
                return {
                    "industry_category": meta.industry_category,
                    "company_stage": meta.company_stage,
                    "coverage_tags": meta.coverage_tags,
                }
    return None


def main() -> None:
    """CLI entry point for the benchmark report generator."""
    parser = argparse.ArgumentParser(
        description="Predictron Engine Benchmark Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m benchmarks.benchmark_report\n"
            "  python -m benchmarks.benchmark_report --version 0.6.5\n"
            "  python -m benchmarks.benchmark_report --diff 0.6.5 0.7.0\n"
            "  python -m benchmarks.benchmark_report --output report.md\n"
            "  python -m benchmarks.benchmark_report --metrics-only\n"
            "  python -m benchmarks.benchmark_report --validate-only\n"
        ),
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Engine version label for this run (default: current)",
    )
    parser.add_argument(
        "--diff",
        nargs=2,
        metavar=("VERSION_A", "VERSION_B"),
        help="Compare two version snapshots (e.g., 0.6.5 0.7.0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write report to file instead of stdout",
    )
    parser.add_argument(
        "--case",
        nargs="*",
        help="Specific case IDs to include in report",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only generate the metrics report",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only generate the validation report",
    )

    args = parser.parse_args()

    if args.diff:
        version_a, version_b = args.diff
        snap_a = load_snapshot(version_a)
        snap_b = load_snapshot(version_b)

        if snap_a is None:
            print(f"Error: No snapshot found for v{version_a}")
            print(f"Expected: {EXPECTED_OUTPUTS_DIR / f'snapshot_v{version_a}.json'}")
            sys.exit(1)
        if snap_b is None:
            print(f"Error: No snapshot found for v{version_b}")
            print(f"Expected: {EXPECTED_OUTPUTS_DIR / f'snapshot_v{version_b}.json'}")
            sys.exit(1)

        diffs = generate_regression_diff(snap_a, snap_b)
        report_text = format_regression_diff(diffs)

        if args.output:
            Path(args.output).write_text(report_text, encoding="utf-8")
            print(f"Regression report written to {args.output}")
        else:
            print(report_text)

        has_regressions = any(d.overall_score_delta < -0.01 for d in diffs)
        sys.exit(1 if has_regressions else 0)

    case_ids = args.case if args.case else None
    results = run_benchmark(case_ids=case_ids)

    if args.metrics_only:
        report_text = generate_metrics_report(results)
    elif args.validate_only:
        report_text = generate_validation_report(results)
    else:
        report_text = generate_text_report(results)

    if args.output:
        Path(args.output).write_text(report_text, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(report_text)

    if not args.metrics_only and not args.validate_only:
        save_snapshot(results, version=args.version)
        print(f"\nSnapshot saved for v{args.version or ENGINE_VERSION}")


if __name__ == "__main__":
    main()
