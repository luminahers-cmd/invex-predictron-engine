#!/usr/bin/env python3
"""Curate SEC EDGAR acquisition output for the V5 pilot.

Removes non-operating investment vehicles (SPACs, funds, REITs, trusts,
ADR shells) and deduplicates by CIK.  This is a deterministic filtering
step — it never fabricates or alters facts; it only excludes records
that are not operating companies.

Run after acquire_sec_edgar.py:

    python scripts/curate_sec_edgar.py <input.json> <output.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# SIC codes that identify non-operating vehicles rather than companies.
_NON_OPERATING_SIC = {
    "6770",  # Blank Check
    "6798",  # Real Estate Investment Trust
    "6722",  # Management Investment Offices, Open-End
    "6726",  # Investment Offices, Not Elsewhere Classified
    "6732",  # Unit Investment Trusts, Closed-End
    "6733",  # Trusts, Except Educational, Religious, and Charitable
    "6799",  # Investors, Not Elsewhere Classified
    "6099",  # Functions Related To Depository Banking, Not Elsewhere Classified
    "6211",  # Security Brokers, Dealers, and Flotation Companies
    "6282",  # Investment Advice
}

# Name markers that identify vehicles rather than operating companies.
# Note: "/ADR" suffixes denote Depositary Receipts of *operating* foreign
# companies and are intentionally NOT filtered.
_VEHICLE_MARKERS = (
    "trust",
    "spac",
    "acquisition corp",
    "acquisition co",
    "merger corp",
    "etf",
    "fund",
    "reit",
    "units",
    "blank check",
    "income fund",
    "strategic investment",
    "government income",
    "depositor inc",
)

# Valid US state and territory abbreviations (SEC state_of_incorporation codes).
_US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "AS", "GU", "MP", "PR", "VI",
}


def enrich_country(record: dict) -> None:
    """Deterministically derive country from SEC state_of_incorporation.

    Valid US state/territory codes map to "United States".  Foreign or
    unmapped jurisdictions are left unknown (never guessed).
    """
    state = str(record.get("state_of_incorporation", "")).strip().upper()
    if state in _US_STATES:
        record.setdefault("country", "United States")


def is_vehicle(record: dict) -> bool:
    """Return True if the record is a non-operating investment vehicle."""
    sic = str(record.get("sic_code", "")).strip()
    if sic in _NON_OPERATING_SIC:
        return True

    name = str(record.get("company_name", "")).lower()
    return any(marker in name for marker in _VEHICLE_MARKERS)


def curate(records: list[dict]) -> dict:
    """Curate records.  Returns summary with the curated list."""
    kept: list[dict] = []
    removed_by_sic = 0
    removed_by_name = 0
    removed_by_dup = 0

    for record in records:
        if is_vehicle(record):
            if str(record.get("sic_code", "")) in _NON_OPERATING_SIC:
                removed_by_sic += 1
            else:
                removed_by_name += 1
            continue
        enrich_country(record)
        kept.append(record)

    # Deduplicate by CIK (the tickers file lists multiple ticker classes).
    cik_seen: set[str] = set()
    unique: list[dict] = []
    for record in kept:
        cik = str(record.get("cik", "")).strip()
        if not cik or cik in cik_seen:
            removed_by_dup += 1
            continue
        cik_seen.add(cik)
        unique.append(record)

    return {
        "records_input": len(records),
        "removed_by_sic": removed_by_sic,
        "removed_by_name": removed_by_name,
        "removed_duplicate_cik": removed_by_dup,
        "records_kept": len(unique),
        "records": unique,
    }


def main() -> None:
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit(f"Input file {input_path} must contain a JSON array")

    result = curate(records)
    output_path.write_text(
        json.dumps(result["records"], indent=2, default=str),
        encoding="utf-8",
    )

    print(json.dumps(
        {k: v for k, v in result.items() if k != "records"},
        indent=2,
    ))


if __name__ == "__main__":
    main()
