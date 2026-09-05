# V5 Pilot — Historical Dataset Population (881 Companies)

Date: 2026-09-05 · Dataset store: `data/population/store` · Source: SEC EDGAR

## Objective

Populate a historical dataset of real, publicly registered companies using only
the existing acquisition/population infrastructure. No changes were made to the
Predictron Engine, scoring, benchmark system, API, or evidence replay. All
additions are additive; no values were fabricated.

## Method

1. **Source audit** — `data/population/source_compatibility_matrix.json`
   - SEC EDGAR: immediately usable, public API, no key, no scraping required.
   - Companies House and YC OSS: viable but require manual/licensed download.
   - Outcome: SEC EDGAR was the only zero-friction source.
2. **Acquisition** — `scripts/acquire_sec_edgar.py`
   - `https://www.sec.gov/files/company_tickers.json` (10,412 registered
     companies) then per-company detail from
     `https://data.sec.gov/submissions/CIK{CIK10}.json` (SIC, state, filings).
   - Fixed the tickers JSON structure (object keyed `"0".."N"`, not a list).
   - Results: `data/population/sources/sec_edgar_companies.json` (1,045 records).
3. **Curation** — `scripts/curate_sec_edgar.py` (deterministic filters only)
   - Removed non-operating vehicles by SIC (6770 blank checks, 6798 REITs,
     funds/brokers/trusts SICs) and name markers (`trust`, `spac`, `fund`,
     `reit`, `units`, ...). ADRs of operating foreign companies are KEPT.
   - Added `country = United States` for all records with a valid US
     state/territory code; foreign/unmapped jurisdictions stay unknown.
   - Removed 25 duplicate CIKs → `sec_edgar_companies_curated.json` (881).
4. **Import** — `python -m predictron_engine.dataset.cli populate`
   - Clean store, `881 imported, 0 failed, 0 duplicates, 0 validation failures`.

## Framework change (additive, dataset tooling only)

`predictron_engine/dataset/acquisition/pipeline.py::_process_batch` bypassed
`ImportPipeline.run()` provenance attachment. A `_attach_provenance` helper now
records per-field provenance (source, retrieval date) for every imported record,
with `engine_version` truthfully attributed to `{source}/pipeline`. All 3,204
tests pass after the change.

## Results

| Metric | Value |
|---|---|
| Imported companies | 881 |
| Failed / skipped / duplicates | 0 / 0 / 0 |
| Validation issues | 0 (`is_valid: true`) |
| Duplicate groups | 0 |
| Distinct sectors (SIC) | 224 |
| Import quality score | 94.74 / 100 |
| Import rate | 22.5 records/s |

### Sector highlights
- Core tech (SIC 737x, 3674, 357x, 3823/3825, 3841/3845, 4812/4813): 131
- Health/biotech (pharmaceuticals 65, biological products 18, surgical
  instruments 18): ~99
- Finance: 82 · Mining/resources: 67
- Top states: DE 350, (empty) 111, E9 56, NV 54, A1 41, X0 18, MD 16, D0 14

### Identifier coverage
- SEC CIK: 881/881 · State of incorporation: 770/881 · Country: 529/881 (US)

## Known limitations (never fabricated)
- **Website:** empty for all records — SEC EDGAR does not publish company
  websites. Quality report flags `missing_required_field: 881` as expected.
- **Funding stage:** `unknown` — SEC EDGAR reports no funding stage.
- **Country:** only US-derived; foreign records left unknown.

## Reports
- `reports/population_report_v5.json` — import metrics
- `reports/validation_report_v5.json` — schema/integrity validation
- `reports/quality_report_v5.json` — quality checks + findings
- `reports/statistics_v5.json` — sector/country/stage/missing-field stats
- `reports/dataset_report_v5.json` — CLI export report
- `reports/pilot_review_v5.json` — consolidated pilot review

## Reproducibility
```
python scripts/acquire_sec_edgar.py                  # fetch SEC data (1,045)
python scripts/curate_sec_edgar.py ...               # filter → 881
python -m predictron_engine.dataset.cli populate --dataset data/population/store \
    --config data/population/population_config.json \
    --report data/population/reports/population_report_v5.json
python scripts/generate_pilot_reports.py data/population/store data/population/reports
```