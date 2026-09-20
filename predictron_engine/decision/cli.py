"""Decision Intelligence CLI.

Commands:
    decision-report      Generate a decision intelligence report
    decision-trace       Show the decision reasoning trace
    decision-explain     Produce a human-readable explanation
    decision-compare     Compare decisions across companies
    decision-calibrate   Calibrate decision confidence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from predictron_engine.decision.calibration_layer import CalibrationPoint
from predictron_engine.decision.intelligence_models import DecisionVerdict
from predictron_engine.decision.service import DecisionIntelligenceService
from predictron_engine.feature_store.store import FeatureStore


def _load_feature_store(args: argparse.Namespace) -> FeatureStore:
    fs = FeatureStore(Path(getattr(args, "feature_dir", "data/features")))
    fs.initialize()
    return fs


def _build_service() -> DecisionIntelligenceService:
    return DecisionIntelligenceService()


def _require_company_id(args: argparse.Namespace) -> str:
    raw = getattr(args, "company_id", None)
    if not raw:
        print("Error: --company-id is required.")
        sys.exit(1)
    return str(raw)


def _resolve_verdict(args: argparse.Namespace) -> DecisionVerdict | None:
    verdict = getattr(args, "verdict", None)
    if not verdict:
        return None
    try:
        return DecisionVerdict(verdict)
    except ValueError:
        print(f"Error: unknown verdict '{verdict}'")
        sys.exit(1)


def _write_output(args: argparse.Namespace, data: dict[str, Any]) -> None:
    output = getattr(args, "output", None)
    dump = json.dumps(data, indent=2, default=str)
    if output:
        Path(output).write_text(dump, encoding="utf-8")
        print(f"Written to {output}")
    else:
        print(dump)


# ---- decision-report ----

def cmd_decision_report(args: argparse.Namespace) -> None:
    """Generate a decision intelligence report for a company."""
    company_id = _require_company_id(args)
    fs = _load_feature_store(args)
    service = _build_service()

    feature_set = fs.load_company_features(company_id)
    if feature_set is None:
        print(f"Error: no feature set found for company '{company_id}'")
        sys.exit(1)

    report = service.produce_decision(
        feature_set,
        verdict=_resolve_verdict(args),
        confidence=getattr(args, "confidence", None),
    )

    if getattr(args, "markdown", False):
        print(service.report_builder.render_markdown(report))
    else:
        _write_output(args, report.to_dict())


# ---- decision-trace ----

def cmd_decision_trace(args: argparse.Namespace) -> None:
    """Show the deterministic reasoning trace for a company."""
    company_id = _require_company_id(args)
    fs = _load_feature_store(args)
    service = _build_service()

    feature_set = fs.load_company_features(company_id)
    if feature_set is None:
        print(f"Error: no feature set found for company '{company_id}'")
        sys.exit(1)

    trace = service.produce_trace(
        feature_set,
        confidence=getattr(args, "confidence", None),
    )

    compact = getattr(args, "compact", False)
    if compact:
        for node in trace.nodes:
            print(f"  [{node.node_type.value}] {node.label}: {node.value}")
        print(f"Overall score: {trace.overall_score}")
        print(f"Verdict: {trace.verdict.value}")
    else:
        _write_output(args, trace.to_dict())


# ---- decision-explain ----

def cmd_decision_explain(args: argparse.Namespace) -> None:
    """Produce a human-readable explanation for a company."""
    company_id = _require_company_id(args)
    fs = _load_feature_store(args)
    service = _build_service()

    feature_set = fs.load_company_features(company_id)
    if feature_set is None:
        print(f"Error: no feature set found for company '{company_id}'")
        sys.exit(1)

    explanation = service.produce_explanation(
        feature_set,
        verdict=_resolve_verdict(args),
        recommendation_text=getattr(args, "recommendation", None),
    )

    print(explanation.headline)
    print()
    print(explanation.full_explanation)
    if explanation.recommendation:
        print()
        print(f"Recommendation: {explanation.recommendation}")


# ---- decision-compare ----

def cmd_decision_compare(args: argparse.Namespace) -> None:
    """Compare decisions across companies."""
    fs = _load_feature_store(args)
    service = _build_service()

    company_ids = getattr(args, "company_ids", None)
    if company_ids:
        ids = [cid.strip() for cid in company_ids.split(",") if cid.strip()]
    else:
        ids = fs.list_companies()

    if not ids:
        print("No companies found in the feature store.")
        return

    reports = []
    for cid in ids:
        feature_set = fs.load_company_features(cid)
        if feature_set is not None:
            reports.append(
                service.produce_decision(feature_set),
            )

    if not reports:
        print("No reports could be produced.")
        sys.exit(1)

    comparisons = service.compare(reports)
    output = getattr(args, "output", None)
    dump = json.dumps(comparisons, indent=2, default=str)
    if output:
        Path(output).write_text(dump, encoding="utf-8")
        print(f"Written to {output}")
    else:
        print(dump)


# ---- decision-calibrate ----

def cmd_decision_calibrate(args: argparse.Namespace) -> None:
    """Calibrate decision confidence against benchmark ground truth."""
    service = _build_service()

    expected = getattr(args, "expected_conf", None)
    actual = getattr(args, "actual_conf", None)
    case_id = getattr(args, "case_id", "default")

    if expected is None or actual is None:
        print("Error: --expected-conf and --actual-conf are required.")
        sys.exit(1)

    if getattr(args, "history", None):
        history_path = Path(getattr(args, "history"))
        if history_path.exists():
            import json as _json

            data = _json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    point_id = item.get("benchmark_case_id", case_id)
                    expected_conf = float(item.get("expected_confidence", 0.0))
                    actual_conf = float(item.get("actual_confidence", 0.0))
                    service.add_historical_calibration_point(
                        CalibrationPoint(
                            benchmark_case_id=point_id,
                            expected_confidence=expected_conf,
                            actual_confidence=actual_conf,
                            timestamp=item.get("timestamp", ""),
                        ),
                    )

    adjustment = service.calibrate(
        expected_confidence=float(expected),
        actual_confidence=float(actual),
        benchmark_case_id=case_id,
    )

    output = getattr(args, "output", None)
    dump = json.dumps(adjustment.to_dict(), indent=2, default=str)
    if output:
        Path(output).write_text(dump, encoding="utf-8")
        print(f"Written to {output}")
    else:
        print(dump)


# ---- Main ----

def main(argv: list[str] | None = None) -> None:
    """Decision Intelligence CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="predictron-decision",
        description="Decision Intelligence CLI — transparent investment reasoning",
    )
    parser.add_argument(
        "--feature-dir", default="data/features",
        help="Feature store directory",
    )

    sub = parser.add_subparsers(dest="command")

    # decision-report
    report_p = sub.add_parser("decision-report", help="Generate decision report")
    report_p.add_argument("--company-id", help="Company ID")
    report_p.add_argument("--verdict", help="Verdict override")
    report_p.add_argument("--confidence", type=float, help="Confidence override")
    report_p.add_argument("--markdown", action="store_true", help="Render markdown")
    report_p.add_argument("-o", "--output", help="Output file path")

    # decision-trace
    trace_p = sub.add_parser("decision-trace", help="Show reasoning trace")
    trace_p.add_argument("--company-id", help="Company ID")
    trace_p.add_argument("--confidence", type=float, help="Confidence override")
    trace_p.add_argument("--compact", action="store_true", help="Compact output")
    trace_p.add_argument("-o", "--output", help="Output file path")

    # decision-explain
    explain_p = sub.add_parser("decision-explain", help="Explain a decision")
    explain_p.add_argument("--company-id", help="Company ID")
    explain_p.add_argument("--verdict", help="Verdict override")
    explain_p.add_argument("--recommendation", help="Recommendation override")

    # decision-compare
    compare_p = sub.add_parser("decision-compare", help="Compare companies")
    compare_p.add_argument("--company-ids", help="Comma-separated company IDs")
    compare_p.add_argument("-o", "--output", help="Output file path")

    # decision-calibrate
    calibrate_p = sub.add_parser("decision-calibrate", help="Calibrate confidence")
    calibrate_p.add_argument("--expected-conf", type=float, help="Expected confidence")
    calibrate_p.add_argument("--actual-conf", type=float, help="Actual confidence")
    calibrate_p.add_argument("--case-id", default="default", help="Benchmark case ID")
    calibrate_p.add_argument("--history", help="Historical calibration JSON file")
    calibrate_p.add_argument("-o", "--output", help="Output file path")

    args = parser.parse_args(argv)

    if args.command == "decision-report":
        cmd_decision_report(args)
    elif args.command == "decision-trace":
        cmd_decision_trace(args)
    elif args.command == "decision-explain":
        cmd_decision_explain(args)
    elif args.command == "decision-compare":
        cmd_decision_compare(args)
    elif args.command == "decision-calibrate":
        cmd_decision_calibrate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
