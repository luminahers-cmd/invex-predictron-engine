"""Benchmark runner — executes startup cases through the Predictron Engine.

Runs every benchmark case through the complete analysis pipeline and
captures the full output for each stage. Produces structured results
that can be used for regression comparison and report generation.

Usage:
    python -m benchmarks.benchmark_runner
    python -m benchmarks.benchmark_runner --case b2b_saas
    python -m benchmarks.benchmark_runner --save-snapshot
    python -m benchmarks.benchmark_runner --save-snapshot --version 0.6.5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.schemas.analysis import StartupAnalysisRequest
from benchmarks.startup_cases.cases import (
    BENCHMARK_CASES,
    get_case_by_id,
)
from predictron_engine.engine import ENGINE_VERSION, PredictronEngine
from predictron_engine.models.report import Report

logger = logging.getLogger(__name__)

BENCHMARKS_DIR = Path(__file__).parent
EXPECTED_OUTPUTS_DIR = BENCHMARKS_DIR / "expected_outputs"


@dataclass
class CaseResult:
    """Result of running a single benchmark case through the engine."""

    case_id: str
    case_label: str
    request: dict[str, Any]
    success: bool
    processing_time_ms: float
    report: Report | None = None
    error: str | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the case result for JSON output."""
        if self.report is None:
            return {
                "case_id": self.case_id,
                "case_label": self.case_label,
                "success": self.success,
                "error": self.error,
                "processing_time_ms": self.processing_time_ms,
            }

        r = self.report
        return {
            "case_id": self.case_id,
            "case_label": self.case_label,
            "success": self.success,
            "processing_time_ms": self.processing_time_ms,
            "engine_version": r.analysis_metadata.engine_version,
            "extracted_features": {
                "industry": r.features.industry,
                "sub_industry": r.features.sub_industry,
                "business_model": r.features.business_model,
                "funding_stage": r.features.funding_stage,
                "geography": r.features.geography,
                "customer_type": r.features.customer_type,
                "technology_stack": r.features.technology_stack,
                "team_size_indicator": r.features.team_size_indicator,
                "founded_year": r.features.founded_year,
                "has_revenue": r.features.has_revenue,
                "key_keywords": r.features.key_keywords,
                "has_pitch_deck": r.features.has_pitch_deck,
                "founder_profile_count": r.features.founder_profile_count,
                "data_completeness": r.features.data_completeness,
                "founder_team_type": r.features.founder_team_type,
                "domain_expertise_signals": r.features.domain_expertise_signals,
                "serial_founder_indicators": r.features.serial_founder_indicators,
                "leadership_roles": r.features.leadership_roles,
                "hiring_signals": r.features.hiring_signals,
                "advisor_mentions": r.features.advisor_mentions,
                "engineering_strength": r.features.engineering_strength,
                "product_strength": r.features.product_strength,
                "founder_market_fit_signals": r.features.founder_market_fit_signals,
                "execution_signals": r.features.execution_signals,
                "founder_confidence": r.features.founder_confidence,
                "product_category": r.features.product_category,
                "product_type": r.features.product_type,
                "saas_model": r.features.saas_model,
                "ai_orientation": r.features.ai_orientation,
                "primary_capabilities": r.features.primary_capabilities,
                "feature_signals": r.features.feature_signals,
                "integration_ecosystem": r.features.integration_ecosystem,
                "deployment_model": r.features.deployment_model,
                "target_workflow": r.features.target_workflow,
                "automation_level": r.features.automation_level,
                "product_maturity": r.features.product_maturity,
                "differentiation_signals": r.features.differentiation_signals,
                "defensibility_signals": r.features.defensibility_signals,
                "technical_complexity": r.features.technical_complexity,
                "scalability_indicators": r.features.scalability_indicators,
                "innovation_signals": r.features.innovation_signals,
                "product_keywords": r.features.product_keywords,
                "product_confidence": r.features.product_confidence,
                "secondary_business_model": r.features.secondary_business_model,
                "revenue_model": r.features.revenue_model,
                "pricing_model": r.features.pricing_model,
                "monetization_strategy": r.features.monetization_strategy,
                "customer_acquisition_model": r.features.customer_acquisition_model,
                "sales_motion": r.features.sales_motion,
                "distribution_model": r.features.distribution_model,
                "value_proposition_signals": r.features.value_proposition_signals,
                "recurring_revenue_signal": r.features.recurring_revenue_signal,
                "marketplace_dynamics": r.features.marketplace_dynamics,
                "network_effects_signals": r.features.network_effects_signals,
                "platform_characteristics": r.features.platform_characteristics,
                "switching_cost_indicators": r.features.switching_cost_indicators,
                "unit_economics_indicators": r.features.unit_economics_indicators,
                "business_model_maturity": r.features.business_model_maturity,
                "business_model_keywords": r.features.business_model_keywords,
                "business_model_confidence": r.features.business_model_confidence,
                "primary_technology_domain": r.features.primary_technology_domain,
                "secondary_technology_domain": r.features.secondary_technology_domain,
                "programming_language_signals": r.features.programming_language_signals,
                "framework_signals": r.features.framework_signals,
                "cloud_infrastructure_signals": r.features.cloud_infrastructure_signals,
                "api_strategy": r.features.api_strategy,
                "data_architecture_signals": r.features.data_architecture_signals,
                "security_signals": r.features.security_signals,
                "infrastructure_maturity": r.features.infrastructure_maturity,
                "open_source_signals": r.features.open_source_signals,
                "developer_tooling_signals": r.features.developer_tooling_signals,
                "engineering_maturity": r.features.engineering_maturity,
                "technology_keywords": r.features.technology_keywords,
                "technology_confidence": r.features.technology_confidence,
                "direct_competitor_signals": r.features.direct_competitor_signals,
                "indirect_competitor_signals": r.features.indirect_competitor_signals,
                "incumbent_signals": r.features.incumbent_signals,
                "market_concentration": r.features.market_concentration,
                "competitive_density": r.features.competitive_density,
                "fragmentation_signals": r.features.fragmentation_signals,
                "winner_take_most_signals": r.features.winner_take_most_signals,
                "network_effect_competition": r.features.network_effect_competition,
                "switching_cost_signals": r.features.switching_cost_signals,
                "competitive_moat_indicators": r.features.competitive_moat_indicators,
                "barriers_to_entry": r.features.barriers_to_entry,
                "substitute_product_signals": r.features.substitute_product_signals,
                "platform_dependency": r.features.platform_dependency,
                "ecosystem_dependency": r.features.ecosystem_dependency,
                "open_source_competition": r.features.open_source_competition,
                "regulatory_competition": r.features.regulatory_competition,
                "geographic_competition": r.features.geographic_competition,
                "pricing_pressure": r.features.pricing_pressure,
                "competitive_keywords": r.features.competitive_keywords,
                "competition_confidence": r.features.competition_confidence,
                "market_risk": r.features.market_risk,
                "founder_risk": r.features.founder_risk,
                "execution_risk": r.features.execution_risk,
                "product_risk": r.features.product_risk,
                "technology_risk": r.features.technology_risk,
                "business_model_risk": r.features.business_model_risk,
                "traction_risk": r.features.traction_risk,
                "competitive_risk": r.features.competitive_risk,
                "regulatory_risk": r.features.regulatory_risk,
                "operational_risk": r.features.operational_risk,
                "platform_dependency_risk": r.features.platform_dependency_risk,
                "customer_concentration_risk": r.features.customer_concentration_risk,
                "hiring_risk": r.features.hiring_risk,
                "funding_risk": r.features.funding_risk,
                "scaling_risk": r.features.scaling_risk,
                "security_risk": r.features.security_risk,
                "compliance_risk": r.features.compliance_risk,
                "risk_keywords": r.features.risk_keywords,
                "risk_confidence": r.features.risk_confidence,
            },
            "evidence": [
                {
                    "domain": e.domain,
                    "category": e.category,
                    "statement": e.statement,
                    "source": e.source,
                    "relevance_score": e.relevance_score,
                }
                for e in r.evidence
            ],
            "observations": [
                {
                    "dimension": o.dimension,
                    "category": o.category,
                    "statement": o.statement,
                    "confidence": o.confidence,
                    "importance": o.importance,
                    "source_rule": o.source_rule,
                }
                for o in r.observations
            ],
            "dimension_assessments": [
                {
                    "dimension": a.dimension,
                    "summary": a.summary,
                    "rationale": a.rationale,
                    "confidence": a.confidence,
                    "score": a.score,
                }
                for a in r.dimension_assessments
            ],
            "scores": [
                {
                    "dimension": s.dimension,
                    "score": s.score,
                    "rationale": s.rationale,
                }
                for s in r.scores
            ],
            "overall_score": r.overall_score,
            "recommendations": [
                {
                    "category": rec.category,
                    "action": rec.action,
                    "priority": rec.priority,
                    "title": rec.title,
                    "confidence": rec.confidence,
                }
                for rec in r.recommendations
            ],
            "confidence": [
                {
                    "dimension": c.dimension,
                    "confidence": c.confidence,
                    "data_completeness": c.data_completeness,
                }
                for c in r.confidence
            ],
            "overall_confidence": r.overall_confidence,
            "investment_decision": (
                {
                    "category": r.investment_decision.category.value,
                    "conviction": r.investment_decision.conviction.value,
                    "composite_score": r.investment_decision.composite_score,
                    "data_quality_modifier": r.investment_decision.data_quality_modifier,
                    "risk_modifier": r.investment_decision.risk_modifier,
                }
                if r.investment_decision
                else None
            ),
            "stage_timings": self.stage_timings,
        }


def _build_request(case: dict[str, Any]) -> StartupAnalysisRequest:
    """Convert a benchmark case dict into a StartupAnalysisRequest."""
    raw = case["request"]
    kwargs: dict[str, Any] = {
        "startup_name": raw["startup_name"],
        "website": raw["website"],
        "description": raw["description"],
    }
    if raw.get("pitch_deck_url"):
        kwargs["pitch_deck_url"] = raw["pitch_deck_url"]
    if raw.get("founder_linkedin_urls"):
        kwargs["founder_linkedin_urls"] = raw["founder_linkedin_urls"]
    return StartupAnalysisRequest(**kwargs)


def _run_with_stage_timings(
    engine: PredictronEngine, request: StartupAnalysisRequest
) -> tuple[Report, dict[str, float]]:
    """Run the engine and capture per-stage timing breakdowns.

    This manually executes each pipeline stage with timing instrumentation
    to provide visibility into which stages consume the most time.
    """
    timings: dict[str, float] = {}
    start = time.perf_counter()

    startup = engine._normalizer.normalize(request)
    timings["normalize"] = _elapsed(start)

    t = time.perf_counter()
    collected_data = engine._collector.collect(startup)
    timings["collect"] = _elapsed(t)

    t = time.perf_counter()
    features = engine._extractor.extract(startup, collected_data)
    timings["extract"] = _elapsed(t)

    t = time.perf_counter()
    evidence_set = engine._evidence.gather(features)
    timings["evidence"] = _elapsed(t)

    t = time.perf_counter()
    observations = engine._reasoning.reason(features, evidence_set.items)
    timings["reasoning"] = _elapsed(t)

    t = time.perf_counter()
    evaluation_result = engine._evaluation.evaluate(
        features, observations, evidence_set.items
    )
    timings["evaluation"] = _elapsed(t)

    t = time.perf_counter()
    scores = engine._scoring.score(features, observations)
    timings["scoring"] = _elapsed(t)

    t = time.perf_counter()
    from predictron_engine.evaluation.investment_readiness import (
        compute_investment_readiness,
    )
    assessments = evaluation_result.assessments
    investment_readiness = compute_investment_readiness(
        features, observations, scores, assessments,
    )
    timings["investment_readiness"] = _elapsed(t)

    t = time.perf_counter()
    recs = engine._recommendations.recommend(
        features, observations, scores, assessments
    )
    timings["recommendations"] = _elapsed(t)

    t = time.perf_counter()
    conf = engine._confidence.assess(
        features, observations, scores, assessments
    )
    timings["confidence"] = _elapsed(t)

    t = time.perf_counter()
    decision = engine._decision.decide(
        features,
        observations,
        scores,
        conf,
        assessments,
        investment_readiness.signal_relationships,
    )
    timings["decision"] = _elapsed(t)

    t = time.perf_counter()
    report = engine._report_builder.build(
        startup,
        features,
        evidence_set.items,
        observations,
        scores,
        recs,
        conf,
        assessments,
        decision,
        investment_readiness,
    )
    timings["report_builder"] = _elapsed(t)

    total = (time.perf_counter() - start) * 1000
    report.analysis_metadata.processing_time_ms = round(total, 2)
    timings["total"] = round(total, 2)

    return report, timings


def _elapsed(start: float) -> float:
    """Return elapsed milliseconds since start."""
    return round((time.perf_counter() - start) * 1000, 2)


def run_benchmark(
    case_ids: list[str] | None = None,
    engine: PredictronEngine | None = None,
) -> list[CaseResult]:
    """Run benchmark cases through the engine and return results.

    Args:
        case_ids: Specific case IDs to run. If None, runs all cases.
        engine: Engine instance to use. If None, creates a default engine.

    Returns:
        List of CaseResult objects with full output capture.
    """
    if engine is None:
        engine = PredictronEngine()

    if case_ids is None:
        target_cases = BENCHMARK_CASES
    else:
        target_cases = []
        for cid in case_ids:
            case = get_case_by_id(cid)
            if case is None:
                logger.warning("Unknown case ID: %s, skipping", cid)
                continue
            target_cases.append(case)

    results: list[CaseResult] = []

    for case in target_cases:
        case_id = case["id"]
        logger.info("Running benchmark: %s (%s)", case_id, case["label"])

        try:
            request = _build_request(case)
            report, stage_timings = _run_with_stage_timings(engine, request)

            result = CaseResult(
                case_id=case_id,
                case_label=case["label"],
                request=case["request"],
                success=True,
                processing_time_ms=stage_timings["total"],
                report=report,
                stage_timings=stage_timings,
            )
        except Exception as exc:
            logger.error("Benchmark %s failed: %s", case_id, exc, exc_info=True)
            result = CaseResult(
                case_id=case_id,
                case_label=case["label"],
                request=case["request"],
                success=False,
                processing_time_ms=0.0,
                error=str(exc),
            )

        results.append(result)

    return results


def save_snapshot(results: list[CaseResult], version: str | None = None) -> Path:
    """Save benchmark results as a JSON snapshot for regression comparison.

    Args:
        results: List of CaseResult objects from run_benchmark().
        version: Engine version string. Defaults to ENGINE_VERSION.

    Returns:
        Path to the saved snapshot file.
    """
    if version is None:
        version = ENGINE_VERSION

    snapshot = {
        "engine_version": version,
        "benchmark_version": "2.0.0",
        "total_cases": len(results),
        "successful_cases": sum(1 for r in results if r.success),
        "failed_cases": sum(1 for r in results if not r.success),
        "results": [r.to_dict() for r in results],
    }

    snapshot_dir = EXPECTED_OUTPUTS_DIR
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"snapshot_v{version}.json"

    snapshot_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    logger.info("Snapshot saved to %s", snapshot_path)
    return snapshot_path


def load_snapshot(version: str) -> dict[str, Any] | None:
    """Load a previously saved benchmark snapshot.

    Args:
        version: Engine version to load snapshot for.

    Returns:
        Parsed snapshot dict, or None if not found.
    """
    snapshot_path = EXPECTED_OUTPUTS_DIR / f"snapshot_v{version}.json"
    if not snapshot_path.exists():
        return None
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def _get_case_metadata(case_id: str) -> tuple[str, str] | tuple[None, None]:
    """Look up industry category and stage for a case."""
    for case in BENCHMARK_CASES:
        if case["id"] == case_id:
            meta = case.get("metadata")
            if meta is not None:
                return (meta.industry_category, meta.company_stage)
    return (None, None)


def _print_case_summary(result: CaseResult) -> None:
    """Print a concise summary for a single case result."""
    status = "PASS" if result.success else "FAIL"
    cat, stage = _get_case_metadata(result.case_id)
    meta_str = f" | {cat}/{stage}" if cat and stage else ""
    print(f"\n  [{status}] {result.case_id}: {result.case_label}{meta_str}")

    if not result.success:
        print(f"    Error: {result.error}")
        return

    r = result.report
    print(f"    Industry: {r.features.industry} | Model: {r.features.business_model}")
    print(f"    Features completeness: {r.features.data_completeness:.0%}")
    print(f"    Evidence items: {len(r.evidence)}")
    print(f"    Observations: {len(r.observations)}")
    print(f"    Assessments: {len(r.dimension_assessments)}")
    print(f"    Scores: {len(r.scores)} | Overall: {r.overall_score:.1f}")
    print(f"    Recommendations: {len(r.recommendations)}")
    print(f"    Overall confidence: {r.overall_confidence:.2f}")
    print(f"    Processing time: {result.processing_time_ms:.1f}ms")


def main() -> None:
    """CLI entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Predictron Engine Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m benchmarks.benchmark_runner\n"
            "  python -m benchmarks.benchmark_runner --case b2b_saas fintech\n"
            "  python -m benchmarks.benchmark_runner --save-snapshot\n"
            "  python -m benchmarks.benchmark_runner --save-snapshot --version 0.6.5\n"
            "  python -m benchmarks.benchmark_runner --list-cases\n"
        ),
    )
    parser.add_argument(
        "--case",
        nargs="*",
        help="Specific case IDs to run (default: all)",
    )
    parser.add_argument(
        "--save-snapshot",
        action="store_true",
        help="Save results as a versioned JSON snapshot",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Engine version for snapshot naming (default: current version)",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List all available benchmark cases and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.list_cases:
        print("Available benchmark cases:\n")
        print(f"  {'ID':<25} {'Category':<15} {'Stage':<12} {'Label'}")
        print(f"  {'-'*25} {'-'*15} {'-'*12} {'-'*30}")
        for case in BENCHMARK_CASES:
            meta = case.get("metadata")
            cat = meta.industry_category if meta else "?"
            stage = meta.company_stage if meta else "?"
            print(f"  {case['id']:<25} {cat:<15} {stage:<12} {case['label']}")
        print(f"\nTotal: {len(BENCHMARK_CASES)} cases")
        return

    case_ids = args.case if args.case else None

    print(f"Predictron Engine Benchmark Suite (v{ENGINE_VERSION})")
    print("=" * 60)

    results = run_benchmark(case_ids=case_ids)

    if args.json:
        output = {
            "engine_version": ENGINE_VERSION,
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print(f"\n{'Case':<25} {'Status':<8} {'Time (ms)':<12} {'Score':<8} {'Conf':<8}")
        print("-" * 61)

        for result in results:
            if result.success:
                r = result.report
                print(
                    f"{result.case_id:<25} {'PASS':<8} "
                    f"{result.processing_time_ms:<12.1f} "
                    f"{r.overall_score:<8.1f} "
                    f"{r.overall_confidence:<8.2f}"
                )
            else:
                print(
                    f"{result.case_id:<25} {'FAIL':<8} "
                    f"{'N/A':<12} {'N/A':<8} {'N/A':<8}"
                )

        print("-" * 61)
        passed = sum(1 for r in results if r.success)
        total = len(results)
        print(f"\n{passed}/{total} cases passed")

        if passed < total:
            print("\nFailed cases:")
            for r in results:
                if not r.success:
                    print(f"  - {r.case_id}: {r.error}")

        print("\nDetailed Results:")
        for result in results:
            _print_case_summary(result)

    if args.save_snapshot:
        path = save_snapshot(results, version=args.version)
        print(f"\nSnapshot saved: {path}")


if __name__ == "__main__":
    main()
