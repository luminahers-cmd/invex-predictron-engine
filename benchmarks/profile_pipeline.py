"""Performance profiler for the Predictron API analysis pipeline.

Profiles each stage of the analysis pipeline and measures:
  - total request latency
  - engine execution time
  - database time
  - serialization time

Usage:
    python -m benchmarks.profile_pipeline
    python -m benchmarks.profile_pipeline --iterations 50
    python -m benchmarks.profile_pipeline --json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from app.schemas.analysis import StartupAnalysisRequest
from benchmarks.startup_cases.cases import BENCHMARK_CASES
from predictron_engine.engine import PredictronEngine
from predictron_engine.models.report import Report

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class StageProfile:
    name: str
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0.0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.times_ms) if self.times_ms else 0.0

    @property
    def stdev_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "stdev_ms": round(self.stdev_ms, 3),
            "samples": len(self.times_ms),
        }


@dataclass
class ProfileResult:
    profiles: dict[str, StageProfile]

    def to_dict(self) -> dict[str, Any]:
        return {
            stage: p.to_dict() for stage, p in sorted(self.profiles.items())
        }


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _build_request(case: dict[str, Any]) -> StartupAnalysisRequest:
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


def _profile_engine_stages(
    engine: PredictronEngine, request: StartupAnalysisRequest
) -> dict[str, float]:
    timings: dict[str, float] = {}

    t = time.perf_counter()
    startup = engine._normalizer.normalize(request)
    timings["normalize"] = _elapsed_ms(t)

    t = time.perf_counter()
    collected_data = engine._collector.collect(startup)
    timings["collect"] = _elapsed_ms(t)

    t = time.perf_counter()
    features = engine._extractor.extract(startup, collected_data)
    timings["extract"] = _elapsed_ms(t)

    t = time.perf_counter()
    evidence_set = engine._evidence.gather(features)
    timings["evidence"] = _elapsed_ms(t)

    t = time.perf_counter()
    observations = engine._reasoning.reason(features, evidence_set.items)
    timings["reasoning"] = _elapsed_ms(t)

    t = time.perf_counter()
    evaluation_result = engine._evaluation.evaluate(
        features, observations, evidence_set.items
    )
    timings["evaluation"] = _elapsed_ms(t)

    t = time.perf_counter()
    scores = engine._scoring.score(features, observations)
    timings["scoring"] = _elapsed_ms(t)

    t = time.perf_counter()
    from predictron_engine.evaluation.investment_readiness import compute_investment_readiness
    assessments = evaluation_result.assessments
    investment_readiness = compute_investment_readiness(
        features, observations, scores, assessments,
    )
    timings["investment_readiness"] = _elapsed_ms(t)

    t = time.perf_counter()
    recs = engine._recommendations.recommend(features, observations, scores, assessments)
    timings["recommendations"] = _elapsed_ms(t)

    t = time.perf_counter()
    conf = engine._confidence.assess(features, observations, scores, assessments)
    timings["confidence"] = _elapsed_ms(t)

    t = time.perf_counter()
    decision = engine._decision.decide(
        features, observations, scores, conf, assessments,
        investment_readiness.signal_relationships,
    )
    timings["decision"] = _elapsed_ms(t)

    t = time.perf_counter()
    report = engine._report_builder.build(
        startup, features, evidence_set.items, observations,
        scores, recs, conf, assessments, decision, investment_readiness,
    )
    timings["report_builder"] = _elapsed_ms(t)

    return timings, report


def _profile_serialization(
    report: Report, startup_name: str
) -> float:
    from app.services.analysis import _report_to_response

    t = time.perf_counter()
    response = _report_to_response(startup_name, report)
    ser_ms = _elapsed_ms(t)

    t = time.perf_counter()
    _ = response.model_dump(mode="json")
    dump_ms = _elapsed_ms(t)

    return ser_ms + dump_ms


def _profile_persistence_serialization(
    report: Report,
    response: Any,
    request: StartupAnalysisRequest,
) -> float:
    from app.models.analysis import AnalysisReport, AnalysisRequest

    t = time.perf_counter()
    db_request = AnalysisRequest(
        user_id=None,
        startup_name=request.startup_name,
        website=str(request.website),
        description=request.description,
        pitch_deck_url=str(request.pitch_deck_url) if request.pitch_deck_url else None,
        founder_linkedin_urls=[str(u) for u in request.founder_linkedin_urls],
    )
    _ = AnalysisReport(
        request_id=db_request.id,
        startup_name=response.startup_name,
        venture_score=response.venture_score,
        market_score=response.market_score,
        founder_score=response.founder_score,
        traction_score=response.traction_score,
        recommendations=response.recommendations,
        confidence=response.confidence,
        engine_version=report.analysis_metadata.engine_version,
        processing_time_ms=report.analysis_metadata.processing_time_ms,
        full_report=report.model_dump(mode="json"),
    )
    return _elapsed_ms(t)


def run_profile(iterations: int = 10, case_ids: list[str] | None = None) -> ProfileResult:
    engine = PredictronEngine()
    profiles: dict[str, StageProfile] = {
        "total": StageProfile("total"),
        "engine_total": StageProfile("engine_total"),
        "serialization": StageProfile("serialization"),
        "persistence_model_creation": StageProfile("persistence_model_creation"),
        "normalize": StageProfile("normalize"),
        "collect": StageProfile("collect"),
        "extract": StageProfile("extract"),
        "evidence": StageProfile("evidence"),
        "reasoning": StageProfile("reasoning"),
        "evaluation": StageProfile("evaluation"),
        "scoring": StageProfile("scoring"),
        "investment_readiness": StageProfile("investment_readiness"),
        "recommendations": StageProfile("recommendations"),
        "confidence": StageProfile("confidence"),
        "decision": StageProfile("decision"),
        "report_builder": StageProfile("report_builder"),
    }

    target_cases = [c for c in BENCHMARK_CASES if case_ids is None or c["id"] in case_ids]
    if not target_cases:
        target_cases = BENCHMARK_CASES

    for iteration in range(iterations):
        for case in target_cases:
            request = _build_request(case)

            # Total round-trip via run_analysis service
            total_start = time.perf_counter()

            stage_timings, report = _profile_engine_stages(engine, request)
            engine_total = _elapsed_ms(total_start)

            # Serialization
            from app.services.analysis import _report_to_response
            t = time.perf_counter()
            response = _report_to_response(request.startup_name, report)
            ser_ms = _elapsed_ms(t)

            t = time.perf_counter()
            _ = response.model_dump(mode="json")
            ser_ms += _elapsed_ms(t)

            # Persistence model creation
            persist_ms = _profile_persistence_serialization(report, response, request)

            total_ms = _elapsed_ms(total_start)

            profiles["total"].times_ms.append(total_ms)
            profiles["engine_total"].times_ms.append(engine_total)
            profiles["serialization"].times_ms.append(ser_ms)
            profiles["persistence_model_creation"].times_ms.append(persist_ms)

            for stage, ms in stage_timings.items():
                if stage in profiles:
                    profiles[stage].times_ms.append(ms)

    return ProfileResult(profiles=profiles)


def format_profile_report(result: ProfileResult) -> str:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("PREDICTRON API PERFORMANCE PROFILE")
    lines.append("=" * 70)

    summary_stages = ["total", "engine_total", "serialization", "persistence_model_creation"]
    lines.append("\n[SUMMARY]")
    fmt = "  {:<30} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6}"
    lines.append(fmt.format("Stage", "Mean", "Median", "Min", "Max", "Stdev", "N"))
    lines.append(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for name in summary_stages:
        p = result.profiles.get(name)
        if p and p.times_ms:
            lines.append(
                f"  {name:<30} {p.mean_ms:>8.2f} {p.median_ms:>8.2f} "
                f"{p.min_ms:>8.2f} {p.max_ms:>8.2f} {p.stdev_ms:>8.2f} {len(p.times_ms):>6}"
            )

    engine_stages = [
        k for k in result.profiles
        if k not in summary_stages and result.profiles[k].times_ms
    ]
    lines.append("\n[ENGINE STAGE BREAKDOWN]")
    lines.append(f"  {'Stage':<25} {'Mean':>8} {'Median':>8} {'% of Total':>10}")
    lines.append(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10}")
    total_mean = result.profiles["total"].mean_ms if result.profiles["total"].times_ms else 1
    for name in sorted(engine_stages):
        p = result.profiles[name]
        if p.times_ms:
            pct = (p.mean_ms / total_mean) * 100
            lines.append(
                f"  {name:<25} {p.mean_ms:>8.3f} {p.median_ms:>8.3f} {pct:>9.1f}%"
            )

    lines.append(f"\n{'=' * 70}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the Predictron analysis pipeline")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--case", nargs="*", help="Specific case IDs to profile")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = run_profile(iterations=args.iterations, case_ids=args.case)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_profile_report(result))

        total = result.profiles["total"]
        engine = result.profiles["engine_total"]
        ser = result.profiles["serialization"]
        persist = result.profiles["persistence_model_creation"]
        if total.times_ms:
            print("\nKey ratios:")
            print(f"  Engine as % of total:  {engine.mean_ms / total.mean_ms * 100:.1f}%")
            print(f"  Serialization as % of total: {ser.mean_ms / total.mean_ms * 100:.1f}%")
            pct = persist.mean_ms / total.mean_ms * 100
            print(f"  Persistence model prep as % of total: {pct:.1f}%")


if __name__ == "__main__":
    main()
