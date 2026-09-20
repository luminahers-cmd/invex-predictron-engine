# Company Profile Enrichment

## Overview

The Company Profile Enrichment feature (Project E1) promotes raw, unstructured
source metadata into canonical, queryable company profile fields on every
stored `DatasetRecord`. It is the data-quality backbone for building a
production-grade company intelligence platform: structured fields enable
domain-based deduplication, multi-country statistics, field-level validation,
and downstream analytics without parsing free-text metadata.

**This module does NOT change any engine scoring, reasoning, confidence,
recommendation, or threshold behavior.** It only refines how dataset records
are stored and described.

## Architecture

```
predictron_engine/dataset/
├── models.py             # CompanyProfile model + DatasetRecord.profile
├── enrichment.py         # Normalization, merging, EnrichmentService (Project E1)
├── imports.py            # RawImportRecord.profile + metadata promotion
├── dedup.py              # Domain-first identity keys
├── quality.py            # duplicate_domain finding
├── statistics.py         # Industries tally, profile-preferred countries
├── validation_utils.py   # Profile field validation checks
├── sources/              # Adapters may attach profiles directly
│   ├── sec_edgar.py
│   ├── yc_oss.py
│   └── gov_registries.py
└── cli.py                # predictron-dataset enrich command
```

## CompanyProfile

A first-class, queryable model attached to every `DatasetRecord`. All fields
default to empty/`None`:

| Field | Type | Description |
|-------|------|-------------|
| `domain` | `str \| None` | Canonical company domain (e.g. `acme.com`) |
| `industries` | `list[str]` | Normalized (lowercase) industry tags |
| `headquarters` | `str \| None` | Human-readable HQ string |
| `country_code` | `str \| None` | ISO 3166-1 alpha-2 country code |
| `city` | `str \| None` | HQ city |
| `region` | `str \| None` | HQ state / region / province |
| `founded_year` | `int \| None` | Year the company was founded |
| `founded_date` | `str \| None` | ISO `YYYY-MM-DD` founding date |
| `employee_count` | `int \| None` | Headcount at analysis time |
| `employee_range` | `str \| None` | Banded range (`1-10`, `11-50`, ...) |
| `description` | `str \| None` | Short company description |
| `legal_name` | `str \| None` | Registered legal name (explicit, not derived) |
| `status` | `str \| None` | `active` / `dissolved` / `closed` |

Helpers:

- `profile.populated_fields() -> list[str]` — subset of named fields that are
  populated.
- `profile.is_empty() -> bool` — True when no fields are populated.

## Enrichment Pipeline

`predictron_engine/dataset/enrichment.py` implements the pipeline:

1. **`extract_domain(url)`** — canonical host with `www.` (and scheme/path)
   stripped.
2. **`normalize_country_code(value)`** — maps alpha-2 codes (passthrough),
   `UK -> GB`, three-letter ISO codes (from a known set), and common English
   country names/aliases to ISO 3166-1 alpha-2.
3. **`normalize_industry(value)`** — lowercases and collapses whitespace.
4. **`normalize_status(value)`** — lowercases and maps `UK -> GB`-style
   synonyms onto `active` / `dissolved` / `closed` when recognizable.
5. **`employee_range(count)`** — deterministic banding of headcount.
6. **`profile_from_raw(raw)` / `profile_from_metadata(metadata, ...)`** —
   build a `CompanyProfile` from either a `RawImportRecord` (for adapters that
   attach one directly) or arbitrary `analysis_metadata`.
7. **`merge_profiles(base, incoming)`** — fill empty fields on `base` from
   `incoming`; existing values are never overwritten (non-destructive).
8. **`enrich_record(record)`** — derives a profile from a record's own
   `analysis_metadata` + `website` + `startup_name`, merges it into the
   existing profile, and reports which fields were added.
9. **`EnrichmentService.enrich(records)` / `enrich_store()`** — runs the
   pipeline over in-memory records or the whole store, persisting updates via
   `DatasetStore.update_record()`.

### Rules

- **Never fabricated.** Country codes, industries, and statuses are only
  inferred from explicit source values; `unknown` is never guessed.
- **Non-destructive.** `merge_profiles` only fills empty fields.
- **Idempotent.** Re-running enrichment on an already-enriched store updates
  zero records.
- **Deterministic.** The same record always produces the same profile.

## Import Integration

- `RawImportRecord` gained a `profile: CompanyProfile | None` field. Adapters
  can attach structured profiles directly (SEC EDGAR, Y Combinator OSS,
  Companies House, US state registries).
- `_normalize_dataset` merges `profile_from_raw(raw)` (adapter-attached
  profile) with `profile_from_metadata(raw.metadata)` (promoted free-text
  metadata), so canonical fields win over string heuristics.

## Deduplication

`dedup.py` now emits a canonical-domain identity key from both
`profile.domain` and the record website:

- Priority: `domain` -> `identifier` -> `website` -> `name`.
- Two records resolving to the same canonical `example.com` match as
  duplicates even when their submitted URLs differ (`https://example.com` vs
  `https://www.example.com`).

## Quality

`quality.py` adds a `duplicate_domain` finding: records sharing a canonical
domain are flagged (case-insensitive) alongside existing duplicate checks.

## Statistics

`statistics.py` now:

- Tallies `industries` (normalized to lowercase) from `profile.industries`.
- Prefers `profile.country_code` when counting records by country, falling
  back to legacy metadata keys and finally `unknown`.

## Validation

`validation_utils.py` extends `validate_record_fields` with profile checks:

- `domain_mismatch` — `profile.domain` differs from the website host.
- `empty_industry` — an industry tag is blank.
- `invalid_founded_year` — outside `[1800, 2200]`.
- `invalid_employee_count` — negative.

`validate_record_completeness` additionally reports missing
`profile.domain` / `profile.industries` / `profile.country_code` /
`profile.headquarters` / `profile.founded_year` / `profile.employee_count` /
`profile.description` fields.

## CLI

```bash
predictron-dataset enrich --dataset ./dataset_store
```

Runs store-wide enrichment and prints an `EnrichmentReport`:

```json
{
  "report_type": "enrichment_report",
  "records_processed": 120,
  "records_updated": 87,
  "fields_populated": {"domain": 65, "country_code": 78, "industries": 54},
  "enrichment_rate": 0.72
}
```

## Usage

```python
from predictron_engine.dataset import EnrichmentService

store = DatasetStore("./my_dataset").initialize()
service = EnrichmentService(store)
report = service.enrich_store()
print(report.enrichment_rate)
```

## Constraints

This module does NOT:

- Modify `PredictronEngine`, scoring, reasoning, confidence, recommendations,
  or thresholds.
- Fabricate company attributes that are not present in source data.
- Overwrite already-populated profile fields.
- Introduce external services (profile resolution APIs are a future connector).

All engine tests pass. Ruff and Mypy are clean.