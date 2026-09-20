#!/usr/bin/env python3
"""SEC EDGAR data acquisition script for V5 pilot population.

Downloads company data from SEC EDGAR's public APIs and produces
JSON files compatible with the existing SecEdgarSource adapter.

Sources used:
  - https://www.sec.gov/files/company_tickers.json  (all companies)
  - https://data.sec.gov/submissions/CIK{cik}.json  (per-company detail)

All data is public and free. SEC requires a descriptive User-Agent header
and rate limiting to 10 requests/second.

This script does NOT modify any engine, scoring, or benchmark code.
It produces data files that feed into the existing import pipeline.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_USER_AGENT = "predictron-engine/0.12.1 (research; pilot-dataset-population)"
_BASE_URL = "https://data.sec.gov"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_RATE_LIMIT_DELAY = 0.1  # 100ms between requests = 10 req/sec

# SIC codes for technology/software companies (broad selection).
_TECH_SIC_CODES = {
    # Software
    "7371", "7372", "7373", "7374", "7375", "7376", "7377", "7378", "7379",
    # Computer and data processing services
    "7370", "7371", "7372", "7373", "7374", "7375", "7376", "7377", "7378", "7379",
    # Communications equipment
    "3661", "3663",
    # Electronic components
    "3670", "3672", "3674", "3675", "3676", "3677", "3678", "3679",
    # Computer hardware
    "3570", "3571", "3572", "3575", "3577", "3578", "3579",
    # Internet/digital services (pre-computer SIC, sometimes used)
    "7371", "7372",
    # Business services
    "7380", "7381", "7382", "7383", "7384", "7385", "7389",
    # Engineering and management services
    "8711", "8713", "8731", "8734",
    # Prepackaged software
    "7372",
    # Data processing
    "7374",
    # Computer programming
    "7371",
    # Computer integrated systems design
    "7373",
    # Information retrieval services
    "7375",
    # Computer facilities management
    "7376",
    # Computer processing and data preparation
    "7374",
    # Medical/life sciences (subset)
    "8731", "8734",
    # Internet services
    "7375",
}

# Broader tech-adjacent SIC codes.
_TECH_ADJACENT_SIC_CODES = {
    # Electric lighting and wiring equipment
    "3610", "3612", "3613", "3620", "3621", "3622", "3623", "3624", "3625",
    # Industrial instruments
    "3812", "3822", "3823", "3824", "3825", "3826", "3827", "3829",
    # Medical instruments
    "3841", "3842", "3843", "3844", "3845", "3846",
    # Communications
    "4812", "4813", "4822", "4899",
    # Electric services
    "4911",
    # Engineering services
    "8711",
}


def _get_client() -> httpx.Client:
    """Create an HTTP client with the required User-Agent header."""
    return httpx.Client(
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
        },
        timeout=30.0,
        follow_redirects=True,
    )


def fetch_company_tickers(client: httpx.Client) -> list[dict[str, Any]]:
    """Fetch the full list of SEC-registered companies.

    Returns a list of dicts with keys: ticker, cik, title.
    """
    logger.info("Fetching company tickers from SEC EDGAR...")
    resp = client.get(_TICKERS_URL)
    resp.raise_for_status()
    data = resp.json()

    companies = []
    if isinstance(data, dict):
        for _key, entry in data.items():
            if isinstance(entry, dict) and "cik_str" in entry:
                companies.append({
                    "ticker": str(entry.get("ticker", "")),
                    "cik": str(entry.get("cik_str", "")),
                    "title": str(entry.get("title", "")),
                })
            elif isinstance(entry, list) and len(entry) >= 3:
                companies.append({
                    "ticker": entry[0],
                    "cik": str(entry[2]).zfill(10),
                    "title": entry[1],
                })

    logger.info("Fetched %d company tickers", len(companies))
    return companies


def fetch_company_detail(
    client: httpx.Client, cik: str
) -> dict[str, Any] | None:
    """Fetch detailed company data from EDGAR submissions API.

    Returns company detail dict or None on failure.
    """
    cik_padded = cik.zfill(10)
    url = f"{_BASE_URL}/submissions/CIK{cik_padded}.json"

    try:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.warning("Failed to fetch CIK %s: %s", cik, exc)
        return None


def extract_company_record(detail: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a EDGAR-format company record from the submissions API response.

    Maps the EDGAR submissions API fields to the format expected by
    SecEdgarSource.
    """
    name = str(detail.get("name", "")).strip()
    if not name:
        return None

    cik = str(detail.get("cik", "")).strip()
    sic = str(detail.get("sic", "")).strip()
    sic_desc = str(detail.get("sicDescription", "")).strip()
    state = str(detail.get("stateOfIncorporation", "")).strip()

    # Try to extract latest filing date from recent filings.
    latest_filing_date = ""
    form_type = ""
    recent = detail.get("filings", {}).get("recent", {})
    if isinstance(recent, dict):
        dates = recent.get("filingDate", [])
        forms = recent.get("form", [])
        if isinstance(dates, list) and dates:
            latest_filing_date = str(dates[0])
        if isinstance(forms, list) and forms:
            form_type = str(forms[0])

    record: dict[str, Any] = {
        "cik": cik,
        "company_name": name,
        "website_url": "",  # SEC does not publish company websites
        "state_of_incorporation": state,
        "sic_code": sic,
        "sic_description": sic_desc,
        "latest_filing_date": latest_filing_date,
        "form_type": form_type,
    }

    # Additional metadata not in SecEdgarSource but useful for provenance.
    tickers = detail.get("tickers", [])
    if tickers and isinstance(tickers, list):
        record["tickers"] = tickers

    exchanges = detail.get("exchanges", [])
    if exchanges and isinstance(exchanges, list):
        record["exchanges"] = exchanges

    return record


def select_tech_companies(
    companies: list[dict[str, Any]],
    limit: int = 1500,
) -> list[dict[str, Any]]:
    """Select technology-relevant companies from the full list.

    Uses a broad selection to ensure we get at least `limit` after
    detail fetching and filtering.
    """
    # Take a spread across the alphabet to avoid clustering.
    step = max(1, len(companies) // limit)
    selected = companies[::step][:limit]

    # If we don't have enough, fill from the beginning.
    if len(selected) < limit:
        remaining = [c for c in companies if c not in selected]
        selected.extend(remaining[: (limit - len(selected))])

    logger.info("Selected %d companies for detail fetching", len(selected))
    return selected


def run_acquisition(
    output_dir: str,
    limit: int = 1500,
    *,
    skip_sic_filter: bool = False,
) -> dict[str, Any]:
    """Run the full SEC EDGAR acquisition pipeline.

    Parameters
    ----------
    output_dir :
        Directory to write the output JSON files.
    limit :
        Maximum number of companies to fetch details for.
    skip_sic_filter :
        If True, include all companies regardless of SIC code.

    Returns
    -------
    Acquisition summary dict.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    summary: dict[str, Any] = {}

    with _get_client() as client:
        # Step 1: Fetch all company tickers.
        all_companies = fetch_company_tickers(client)
        summary["total_companies_available"] = len(all_companies)

        # Step 2: Select a subset for detail fetching.
        selected = select_tech_companies(all_companies, limit=limit)

        # Step 3: Fetch detail for each selected company.
        records: list[dict[str, Any]] = []
        skipped = 0
        failed = 0
        tech_filtered = 0

        for idx, company in enumerate(selected):
            cik = company["cik"]
            logger.info(
                "[%d/%d] Fetching %s (CIK %s)...",
                idx + 1, len(selected), company["title"], cik,
            )

            detail = fetch_company_detail(client, cik)
            if detail is None:
                failed += 1
                continue

            record = extract_company_record(detail)
            if record is None:
                skipped += 1
                continue

            # Filter by SIC code unless skip_sic_filter is set.
            if not skip_sic_filter:
                sic = record.get("sic_code", "")
                if sic not in _TECH_SIC_CODES and sic not in _TECH_ADJACENT_SIC_CODES:
                    tech_filtered += 1
                    continue

            records.append(record)

            # Rate limit compliance.
            time.sleep(_RATE_LIMIT_DELAY)

            # Progress logging every 100 companies.
            if (idx + 1) % 100 == 0:
                logger.info(
                    "Progress: %d/%d fetched, %d kept, %d tech-filtered, %d failed",
                    idx + 1, len(selected), len(records), tech_filtered, failed,
                )

    # Step 4: Write output file.
    output_file = out_path / "sec_edgar_companies.json"
    output_file.write_text(
        json.dumps(records, indent=2, default=str),
        encoding="utf-8",
    )

    elapsed = time.monotonic() - start_time

    summary.update({
        "records_imported": len(records),
        "records_skipped": skipped,
        "records_tech_filtered": tech_filtered,
        "records_failed": failed,
        "output_file": str(output_file),
        "elapsed_seconds": round(elapsed, 2),
        "throughput_per_second": round(len(records) / max(elapsed, 0.01), 2),
    })

    logger.info(
        "Acquisition complete: %d records written to %s (%.1fs)",
        len(records), output_file, elapsed,
    )

    return summary


def run_acquisition_with_sic_filter(
    output_dir: str,
    target_count: int = 1200,
) -> dict[str, Any]:
    """Two-pass acquisition: first try tech SIC filter, then fill with all companies.

    Pass 1: Fetch companies with tech SIC codes.
    Pass 2: If under target, fetch additional companies (no SIC filter).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()

    with _get_client() as client:
        all_companies = fetch_company_tickers(client)

        # Pass 1: Tech-filtered
        selected_tech = select_tech_companies(all_companies, limit=target_count)
        records: list[dict[str, Any]] = []
        fetched_ciks: set[str] = set()

        for idx, company in enumerate(selected_tech):
            cik = company["cik"]
            detail = fetch_company_detail(client, cik)
            if detail is None:
                continue
            record = extract_company_record(detail)
            if record is None:
                continue

            sic = record.get("sic_code", "")
            if sic in _TECH_SIC_CODES or sic in _TECH_ADJACENT_SIC_CODES:
                records.append(record)
                fetched_ciks.add(cik)

            time.sleep(_RATE_LIMIT_DELAY)

            if (idx + 1) % 100 == 0:
                logger.info("Pass 1: %d/%d fetched, %d tech companies kept",
                           idx + 1, len(selected_tech), len(records))

        logger.info("Pass 1 complete: %d tech companies", len(records))

        # Pass 2: Fill if under target.
        if len(records) < target_count:
            remaining_needed = target_count - len(records)
            logger.info(
                "Pass 2: need %d more companies, fetching without SIC filter",
                remaining_needed,
            )

            fill_candidates = [c for c in all_companies if c["cik"] not in fetched_ciks]
            # Spread selection
            step = max(1, len(fill_candidates) // (remaining_needed * 2))
            fill_selected = fill_candidates[::step][:remaining_needed * 2]

            for company in fill_selected:
                if len(records) >= target_count:
                    break
                cik = company["cik"]
                if cik in fetched_ciks:
                    continue

                detail = fetch_company_detail(client, cik)
                if detail is None:
                    continue
                record = extract_company_record(detail)
                if record is None:
                    continue

                records.append(record)
                fetched_ciks.add(cik)
                time.sleep(_RATE_LIMIT_DELAY)

            logger.info("Pass 2 complete: %d total companies", len(records))

    # Write output.
    output_file = out_path / "sec_edgar_companies.json"
    output_file.write_text(
        json.dumps(records, indent=2, default=str),
        encoding="utf-8",
    )

    elapsed = time.monotonic() - start_time
    return {
        "records_imported": len(records),
        "output_file": str(output_file),
        "elapsed_seconds": round(elapsed, 2),
        "throughput_per_second": round(len(records) / max(elapsed, 0.01), 2),
    }


if __name__ == "__main__":
    import sys

    output = sys.argv[1] if len(sys.argv) > 1 else "data/population/sources"
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 1200

    summary = run_acquisition(output, limit=target, skip_sic_filter=True)
    print(json.dumps(summary, indent=2))
