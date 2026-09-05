# Manual Download Processes (For Sources Requiring External Download)

Exact steps for sources that could not be auto-fetched for the V5 pilot.
Each entry lists URLs, file names, expected format, and target fields.

## 1. UK Companies House (bulk CSV)

- Download page: `https://download.companieshouse.gov.uk/`
- Files published daily; naming pattern:
  `BasicCompanyDataAsOneFile-<DATE>.zip` (full snapshot, e.g.
  `BasicCompanyDataAsOneFile-2026-09-04.zip`).
- Also available partitioned: `BasicCompanyData-<N>-<DATE>.zip` where N is the
  500k-record partition number.
- Expected format: single-row-per-company CSV with
  `CompanyNumber, CompanyName, CompanyStatus, CompanyType, CountryOfOrigin,
  IncorporationDate, ... , SICCode_SicText_1` (+ 2..4). SIC codes are combined
  as `1234 - Description`.
- Target fields for Predictron: `CompanyName` (startup_name), `CompanyNumber`
  (identifier), `CountryOfOrigin`, `IncorporationDate`, `SICCode_SicText_1`
  (sector). **No website field.**
- Steps:
  1. Download the latest single file zip (100MB+).
  2. Extract the CSV into `data/population/sources/uk/`.
  3. Curate (active status, tech-derivable SIC codes, de-duplicate) then import
     using the existing `companies_house` connector.
- Terms: open data license (free to reuse with attribution).

## 2. Y Combinator Open Source Startups (CSV on data.world)

- Dataset page: `https://data.world/ycombinator/open-source-startups`
- Requires: free data.world account (login) and accepting the dataset license.
- Steps:
  1. Create/verify a data.world account and sign in.
  2. Accept the license on the dataset page top-right ("Accept & Download").
  3. Export the dataset as CSV (Files tab → download), file typically shown as
     a single CSV.
  4. Save to `data/population/sources/yc/`.
- Expected format: CSV with columns `Company, Website, Description, Region,
  Industry, CB_URL, ...` (~1,000 rows).
- Target fields for Predictron: `Company`, `Website` (**solves the SEC website
  gap**), `Description`, `Industry`.
- Steps:
  1. Import as its own source via the `yc_oss` connector, OR
  2. Use as enrichment: match on normalized company name → attach `website`
     to existing SEC records with provenanced `yc_oss/website`.

## 3. SEC EDGAR (reference — no manual download needed)

- No download required; fetched live by `scripts/acquire_sec_edgar.py`
  from `https://www.sec.gov/files/company_tickers.json` and
  `https://data.sec.gov/submissions/CIK{CIK10}.json`.
- Set descriptive `User-Agent`, max ~10 req/s (already implemented).

## 4. Generic CSV exports (Crunchbase / PitchBook / Kaggle)

- Download any export as CSV into `data/population/sources/csv/`.
- Use the `csv_export` connector with explicit column mapping.
- Validate licensing/terms before use; mark provenance as the export source.