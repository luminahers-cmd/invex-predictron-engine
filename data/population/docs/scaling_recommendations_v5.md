# Scaling to 10,000+ Companies — Recommendations

Status: pilot of **881 companies** imported from SEC EDGAR (V5).
This document is the pre-scaling review required before expanding to 10k+.

## 1. Recommended source mix (in priority order)

1. **SEC EDGAR (done, 881)** — fully automated, no key. Scale options:
   - Acquire the remaining tickers (10,412 total) and apply the same
     deterministic curation → several thousand more operating companies.
   - Add an **annual/semi-annual basis script** that re-runs daily with a
     delta (new CIKs only) so the corpus stays current with no full refetch.
2. **YC Open Source Startups** — best immediate fix for the **missing website**
   field. Requires a free data.world account (licensing acceptance, manual
   download). ~1,000 YC companies with websites + descriptions. Pairs well with
   SEC EDGAR: same company matched by name → website + industry enrichment.
3. **UK Companies House** — bulk CSV (free, open data) for UK coverage
   (millions of companies, `SICCode_1`, incorporation date, country of origin).
   Import a curated slice (e.g. active tech SICs) via the existing connector.
4. **Crunchbase / Kaggle exports** — generic `csv_export` connector accepts any
   manually downloaded CSV with flexible column mapping. Use for enrichment
   only (websites, funding stage); treat as third-party and validate.

## 2. Website enrichment strategy

The only material quality gap is `website` (0% coverage → quality
`missing_required_field`). Options in order of trust:

- **YC OSS / Crunchbase exports** — authoritative, batch, no scraping.
- **SEC → filing → identity extraction** — possible via the XBRL/8-K company
  summaries, but slow and must respect SEC fair-access (rate-limit 10 req/s).
  Do NOT scrape sites that prohibit it.
- **Search-based resolution** — only with documented extraction policy and
  provenance; must keep `unknown` where unresolved.

Recommended target: website coverage ≥ 60% before considering the corpus
"released", since the engine currently has no website to collect evidence from.

## 3. Data quality gates before 10k

- **Dedup across sources** — same CIK (SEC), company number (UK), name+country:
  use the existing dedup with explicit group keys; record merge policy.
- **Deterministic curation only** — keep vehicle/operating filters as explicit
  allowlists; never guess funding stage or country.
- **Provenance per field** (already enforced) — extend to enrichment sources
  (e.g. `yc_oss/website`, `companies_house/sic_code_1`).
- **Validation** — keep `verify` zero-issue requirement before each release.

## 4. Operations & rate limits

- SEC EDGAR: identify `User-Agent` as `predictron-dataset (contact@...)`,
  max ~10 req/s; the acquisition script already throttles. For a full 10,412
  refetch this is ~17 min at 10 req/s; prefer delta updates.
- Companies House: no key needed for bulk download; SIC/registry matching is
  case-normalized.

## 5. Deliverables pending for scale-out

1. `companies_house` bulk CSV staged at `data/population/sources/uk/` with a
   curation script mirroring `scripts/curate_sec_edgar.py`.
2. `yc_oss` CSV staged at `data/population/sources/yc/` with name-normalization
   matching to return websites for existing SEC records.
3. A delta-update runner for SEC EDGAR.
4. Merge policy + provenance rules for cross-source enrichment.
5. Re-run `verify` + `dedup` + quality report after each new source.

## 6. Known constraint

881 is within the pilot target (500–1,000). Reaching ~1,000 from SEC EDGAR
alone is possible with a second non-overlapping batch; the corpus's *quality*
value improves faster by adding websites (YC OSS) than by adding more
website-less SEC records.