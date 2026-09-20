"""Generate the *example* golden dataset for the Ground Truth Evaluation Platform.

The output file is ``benchmarks/golden_datasets/example_v1.json``.  It is an
**illustrative, synthetic sample** — every verified outcome is fictional and
explicitly flagged ``provenance.is_example = true`` so the platform never
mistakenly treats these labels as real.  Real benchmark datasets must be
curated from verifiable sources and must NOT set ``is_example``.

The *historical predictions* come from a real committed engine snapshot
(``benchmarks/expected_outputs/snapshot_v0.13.0.json``); only the outcomes
are invented for demonstration.  Re-running this script is deterministic.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.ground_truth.schema import (  # noqa: E402
    OutcomeEvent,
    OutcomeEventKind,
    StartupStatus,
)
from benchmarks.ground_truth_eval.dataset import dump_golden_dataset  # noqa: E402
from benchmarks.ground_truth_eval.models import (  # noqa: E402
    GoldenDataset,
    GoldenEntry,
    HistoricalEvidence,
    HistoricalPrediction,
    Provenance,
    VerifiedOutcome,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = REPO_ROOT / "benchmarks" / "expected_outputs" / "snapshot_v0.13.0.json"
OUTPUT_PATH = REPO_ROOT / "benchmarks" / "golden_datasets" / "example_v1.json"
VERIFICATION_DATE = date(2026, 6, 30)

# company_id -> (status, exit_value_usd, arr_usd, country, stage)
# stage/country/sector labels are illustrative values for breakdown reports.
OUTCOMES: dict[str, tuple[str, float | None, float | None, str, str]] = {
    "b2b_saas": ("acquired", 120_000_000, None, "us", "Series A"),
    "healthcare_ai": ("operating", None, 5_000_000, "us", "Series A"),
    "fintech": ("shutdown", None, None, "global", "Series B"),
    "devtools": ("operating", None, None, "us", "Series A"),
    "marketplace": ("acquired", 200_000_000, None, "uk", "Series B"),
    "consumer_app": ("shutdown", None, None, "us", "Seed"),
    "climate_tech": ("operating", None, 3_000_000, "eu", "Series A"),
    "robotics": ("shutdown", None, None, "us", "Series A"),
    "enterprise_software": ("acquired", 80_000_000, None, "us", "Series A"),
    "ai_infrastructure": ("operating", None, None, "us", "Series A"),
    "deep_tech": ("shutdown", None, None, "de", "Series B"),
    "edtech": ("acquired", 40_000_000, None, "global", "Series A"),
    "healthtech_device": ("operating", None, 2_000_000, "us", "Series B"),
}


def _decision_for(score: float) -> str:
    if score >= 75.0:
        return "strong_invest"
    if score >= 60.0:
        return "invest"
    if score >= 45.0:
        return "watch"
    if score >= 30.0:
        return "investigate_further"
    return "pass"


def _outcome_events(
    status: str, exit_value_usd: float | None, arr_usd: float | None
) -> list[OutcomeEvent]:
    if status == "acquired":
        return [
            OutcomeEvent(
                kind=OutcomeEventKind.ACQUISITION,
                occurred_at=date(2026, 3, 15),
                value_currency_usd=exit_value_usd,
                description="Acquired (example/illustrative outcome)",
                sources=["example_corpus"],
            )
        ]
    if status == "shutdown":
        return [
            OutcomeEvent(
                kind=OutcomeEventKind.SHUTDOWN,
                occurred_at=date(2025, 11, 1),
                description="Ceased operations (example/illustrative outcome)",
                sources=["example_corpus"],
            )
        ]
    if arr_usd:
        return [
            OutcomeEvent(
                kind=OutcomeEventKind.ARR_MILESTONE,
                occurred_at=date(2026, 4, 1),
                amount_nominal_units=arr_usd,
                description="ARR milestone (example/illustrative outcome)",
                sources=["example_corpus"],
            )
        ]
    return []


def _load_snapshot() -> dict[str, dict[str, Any]]:
    document = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    return {r["case_id"]: r for r in document["results"]}


def build_entries(
    seed: dict[str, dict[str, Any]], cases: list[dict[str, Any]]
) -> list[GoldenEntry]:
    entries: list[GoldenEntry] = []
    for case in sorted(cases, key=lambda c: c["id"]):
        case_id = case["id"]
        if case_id not in seed:
            continue
        result = seed[case_id]
        if not result["success"]:
            continue
        request = dict(case["request"])
        score = float(result["overall_score"])
        confidence = float(result["overall_confidence"])
        status, exit_value, arr_usd, country, stage = OUTCOMES[case_id]
        status_enum = StartupStatus(status)

        entry = GoldenEntry(
            company_id=case_id,
            company_name=str(request["startup_name"]),
            request=request,
            sector=str(case["metadata"].industry_category),
            country=country,
            stage=stage,
            historical_prediction=HistoricalPrediction(
                overall_score=round(score, 4),
                overall_confidence=round(confidence, 4),
                decision=_decision_for(score),
                recommendation_categories=sorted(
                    {r["category"] for r in result["recommendations"]}
                ),
            ),
            historical_evidence=HistoricalEvidence(
                corpus_name=case_id,
                sources=["benchmarks/offline_evidence"],
            ),
            verified_outcome=VerifiedOutcome(
                status=status_enum,
                verification_date=VERIFICATION_DATE,
                outcome_events=_outcome_events(status, exit_value, arr_usd),
                exit_value_usd=exit_value,
                sources=["example_corpus"],
            ),
            provenance=Provenance(
                sources=["benchmarks/startup_cases/cases.py", "snapshot_v0.13.0.json"],
                collector="gen_example_golden_dataset.py",
                verified_by="benchmark-platform",
                is_example=True,
                notes=(
                    "Synthetic example dataset: historical predictions come from a "
                    "real engine snapshot; outcomes are invented for demonstration. "
                    "Do not treat as real ground truth."
                ),
            ),
            metadata={
                "source_snapshot": "snapshot_v0.13.0.json",
                "generated_at": datetime(2026, 6, 30, tzinfo=UTC).isoformat(),
            },
        )
        entries.append(entry)
    return entries


def main() -> int:
    from benchmarks.startup_cases.cases import BENCHMARK_CASES

    seed = _load_snapshot()
    entries = build_entries(seed, BENCHMARK_CASES)
    dataset = GoldenDataset(
        dataset_name="predictron_example",
        benchmark_version="1.0",
        created_at=datetime(2026, 6, 30, tzinfo=UTC),
        entries=entries,
    )
    dump_golden_dataset(dataset, OUTPUT_PATH, overwrite=True)
    print(f"wrote {OUTPUT_PATH} with {len(entries)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
