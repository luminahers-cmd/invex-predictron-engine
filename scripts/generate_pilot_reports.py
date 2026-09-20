#!/usr/bin/env python3
"""Generate V5 pilot deliverables from the populated dataset store.

Produces:
  - validation report (validate_dataset)
  - quality report (generate_quality_report)
  - dataset statistics (compute_dataset_stats)
  - pilot review with sector/country/dup/completeness/quality score

All outputs are generated from existing framework modules; nothing is
fabricated.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from datetime import UTC, datetime

from predictron_engine.dataset.models import DatasetRecord
from predictron_engine.dataset.quality import generate_quality_report
from predictron_engine.dataset.statistics import compute_dataset_stats
from predictron_engine.dataset.store import DatasetStore
from predictron_engine.dataset.validation import validate_dataset


def _load_records(store: DatasetStore) -> list[DatasetRecord]:
    return [r for r in (store.load_record(i) for i in store.list_records()) if r]


def assess_quality_score(records, quality: dict, stats: dict, validation) -> dict:
    """Compute an honest import-quality score (0-100).

    Known source limitations (empty website, unknown funding stage) are
    reported separately, not double-penalized as import defects.
    """
    total = max(quality.get("records_checked", 0), 1)
    findings = quality.get("finding_count", 0)
    missing = stats.get("missing_fields", {}).get("missing_fields", {})

    # Structural cleanliness: findings that are NOT the documented website
    # source limitation (every current finding is missing_required_field
    # for website, which SEC EDGAR never publishes).
    structural_findings = max(findings - missing.get("website", 0), 0)
    structure_ok = 1.0 - structural_findings / total

    dup_ok = 1.0 - stats.get("duplicate_report", {}).get(
        "duplicate_record_count", 0
    ) / total

    # Completeness: identifier coverage the source actually provides.
    cik = sum(1 for r in records if r.analysis_metadata.get("sec_cik"))
    state = sum(1 for r in records if r.analysis_metadata.get("state_of_incorporation"))
    country = sum(1 for r in records if r.analysis_metadata.get("country"))
    completeness = (cik + state + country) / (3.0 * total)

    provenance_ok = 1.0 - missing.get("provenance", 0) / total

    valid = 1.0 if validation.is_valid else 0.0

    score = round(
        100
        * (0.25 * structure_ok + 0.20 * dup_ok + 0.30 * completeness
           + 0.15 * provenance_ok + 0.10 * valid),
        2,
    )
    return {
        "score": score,
        "structural_cleanliness_component": round(100 * structure_ok, 2),
        "duplication_component": round(100 * dup_ok, 2),
        "completeness_component": round(100 * completeness, 2),
        "provenance_component": round(100 * provenance_ok, 2),
        "validation_component": round(100 * valid, 2),
        "identifier_coverage": {"sec_cik": cik, "state": state, "country": country},
        "website_coverage_rate": round(1 - missing.get("website", 0) / total, 4),
        "methodology": (
            "100*(0.25*structural + 0.20*dup_ok + 0.30*completeness + "
            "0.15*provenance + 0.10*validation). The website field is not "
            "counted as an import defect (SEC EDGAR does not publish it); "
            "it is reported separately as website_coverage_rate."
        ),
    }


def main() -> None:
    store_path = sys.argv[1]
    report_dir = Path(sys.argv[2])
    report_dir.mkdir(parents=True, exist_ok=True)

    store = DatasetStore(store_path)
    store.initialize()
    records = _load_records(store)

    validation = validate_dataset(store)
    quality_report = generate_quality_report(store)
    stats = compute_dataset_stats(store)

    quality_dict = quality_report.to_dict()
    stats_dict = stats.to_dict()
    qscore = assess_quality_score(records, quality_dict, stats_dict, validation)

    pilot_review = {
        "project": "V5 Pilot Historical Dataset Population",
        "generated_at": datetime.now(UTC).isoformat(),
        "model": "Predictron historical dataset",
        "imported_companies": len(records),
        "sector_distribution": stats_dict["sectors"],
        "sector_count": len(stats_dict["sectors"]),
        "country_distribution": stats_dict["countries"],
        "funding_stage_distribution": stats_dict["stages"],
        "analysis_year_distribution": stats_dict["years"],
        "duplicate_rate": round(
            stats_dict["duplicate_report"]["duplicate_record_count"]
            / max(len(records), 1),
            4,
        ),
        "duplicate_group_count": stats_dict["duplicate_report"]["group_count"],
        "completeness_statistics": stats_dict["missing_fields"],
        "import_quality_score": qscore,
        "identifiers": {
            "records_with_sec_cik": sum(
                1 for r in records if r.analysis_metadata.get("sec_cik")
            ),
            "records_with_state_of_incorporation": sum(
                1 for r in records if r.analysis_metadata.get("state_of_incorporation")
            ),
            "records_with_country": sum(
                1 for r in records if r.analysis_metadata.get("country")
            ),
            "records_with_website": sum(1 for r in records if r.website),
        },
        "validation": {
            "is_valid": validation.is_valid,
            "issue_count": validation.issue_count,
            "issues": validation.issues,
        },
        "quality": {
            k: v for k, v in quality_dict.items() if k != "findings"
        },
        "known_limitations": [
            "website is empty for all records — SEC EDGAR does not publish "
            "company websites (quality flags missing_required_field)",
            "funding_stage_at_analysis is UNKNOWN â€” SEC EDGAR does not provide funding stage",
            "foreign companies with non-US state codes have country left unknown (never guessed)",
        ],
    }

    (report_dir / "validation_report_v5.json").write_text(
        json.dumps(validation.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    (report_dir / "quality_report_v5.json").write_text(
        json.dumps(quality_dict, indent=2, default=str), encoding="utf-8"
    )
    (report_dir / "statistics_v5.json").write_text(
        json.dumps(stats_dict, indent=2, default=str), encoding="utf-8"
    )
    (report_dir / "pilot_review_v5.json").write_text(
        json.dumps(pilot_review, indent=2, default=str), encoding="utf-8"
    )

    print("Wrote:")
    for f in ("validation_report_v5", "quality_report_v5", "statistics_v5", "pilot_review_v5"):
        print("  ", report_dir / f"{f}.json")


if __name__ == "__main__":
    main()
