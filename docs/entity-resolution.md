# Entity Resolution & Fuzzy Deduplication

## Overview

The Entity Resolution module (Project E2) transforms a collection of
`DatasetRecord` instances into canonical `CompanyIdentity` objects using a
multi-stage, explainable, machine-learning-free matching pipeline. It replaces
exact-key deduplication with canonical company identity resolution across
heterogeneous sources.

**This module does NOT modify any engine scoring, reasoning, confidence,
recommendation, or threshold behavior.** It only resolves which records refer
to the same logical company.

## Architecture

```
predictron_engine/dataset/
├── company_name.py          # Suffix stripping, normalization, canonical keys
├── fuzzy.py                 # Levenshtein, Jaro-Winkler, token overlap, composite
├── identity.py              # CompanyIdentity model, merge rules, builder
├── entity_resolution.py     # Multi-stage matching, blocking, clustering, resolver
├── duplicate_review.py      # Classification: auto-merge / manual review / unrelated
├── dedup.py                 # Exact-key deduplication (legacy, still available)
├── models.py                # DatasetRecord, CompanyProfile
└── cli.py                   # entity-resolve, duplicate-review, dedup commands
```

## Pipeline Stages

### 1. Blocking (`ResolutionIndex.build`)

Records are bucketed by strong identifiers to avoid O(n²) comparisons:

| Bucket Type | Key | Purpose |
|-------------|-----|---------|
| Identifier | `{kind}:{value}` | SEC CIK, Companies House number, registration number, etc. |
| Domain | Canonical domain | Matches `acme.com` across records |
| Name | Canonical name key | Stripped/suffix-removed exact name match |
| Token | Individual name tokens | Shared significant words |

Only records sharing at least one bucket are compared as candidate pairs.

### 2. Multi-Stage Matching (`score_pair`)

Each candidate pair is scored through four progressive stages:

**Stage 1 — Exact Strong Identifiers**
- SEC CIK match → confidence 0.99
- Companies House number match → confidence 0.99
- Registration number match → confidence 0.99
- Other registry identifiers → confidence 0.98

**Stage 1b — Exact Canonical Domain**
- Same normalized domain → confidence 0.98

**Stage 2 — Normalized Core-Name Equality**
- After: Unicode NFC normalization, case folding, corporate suffix stripping,
  punctuation removal, whitespace collapse
- Same core name → confidence 0.92–0.97 (boosted by country/industry match)
- Country conflict → confidence 0.68 (below auto-merge threshold)

**Stage 3 — Fuzzy Name Similarity**
- Composite score: 0.5 × Jaro-Winkler + 0.3 × token overlap + 0.2 × normalized Levenshtein
- Identity threshold: ≥ 0.86
- Review threshold: ≥ 0.72

**Stage 4 — Composite Profile Evidence**
- Country match, industry overlap, city/state match, domain overlap
- Each supporting field adds a small boost to the final confidence

### 3. Confidence Scoring

Every match result returns:

| Field | Type | Description |
|-------|------|-------------|
| `confidence` | `float` | 0.0–1.0 match confidence |
| `match_reason` | `str` | Human-readable reason (e.g. `exact_domain`, `fuzzy_name_country`) |
| `matched_fields` | `list[str]` | Which fields matched (e.g. `["domain", "country"]`) |
| `evidence` | `MatchEvidence` | Structured supporting evidence |

### 4. Clustering

Accepted pair edges (confidence ≥ identity threshold) are unioned into
connected clusters via union-find. Each cluster becomes one `CompanyIdentity`.

### 5. Classification

Each cluster is labeled:
- **`automatic_merge`** — confidence ≥ identity threshold (0.86)
- **`manual_review`** — confidence between review threshold (0.72) and identity threshold
- **`unrelated`** — below review threshold (never merged)

## CompanyIdentity Model

The canonical identity for a resolved company:

| Field | Type | Description |
|-------|------|-------------|
| `identity_id` | `str` | UUIDv4 |
| `record_ids` | `list[str]` | All member record IDs |
| `canonical_name` | `str` | Most frequent core-name variant |
| `canonical_domain` | `str \| None` | Most frequent domain |
| `record_names` | `dict[str, str]` | `record_id → original name` (provenance) |
| `record_domains` | `dict[str, str]` | `record_id → original domain` (provenance) |
| `aliases` | `list[str]` | Alternate names, former names, spellings |
| `alternate_domains` | `list[str]` | All domains across members |
| `industries` | `list[str]` | Sorted union of industry tags |
| `country_code` | `str \| None` | Majority-vote country code |
| `country_codes` | `list[str]` | All distinct country codes |
| `identifiers` | `dict[str, list[str]]` | Kind → sorted union of values |
| `identifier_sources` | `dict[str, list[str]]` | `"kind:value" → record IDs` |
| `sources` | `list[str]` | Distinct record sources |
| `merge_decision` | `str` | `singleton` or `automatic_merge` |

## Merge Rules (Deterministic, Lossless)

1. **Preserve provenance** — every member `record_id` and its original name/domain are retained.
2. **Never discard conflicting values** — conflicting names/domains become aliases/alternate domains.
3. **Maintain source attribution** — `identifier_sources` tracks which record supplied each identifier.
4. **Keep historical identifiers** — all identifier values across members are retained.
5. **No fabricated data** — country, founding year, status are only taken from member records.

## Alias System

Supports:
- Alternate legal names (from metadata keys: `aliases`, `former_name`, `previous_name`, `also_known_as`, `legal_name`, `registered_name`)
- Alternate spellings (preserved as aliases during merge)
- Corporate suffix variations (handled by `company_name.py` suffix stripping)
- All aliases are sorted and deduplicated in `finalize()`

## Name Normalization (`company_name.py`)

Four deterministic steps:
1. Unicode normalization (NFC) + case folding
2. Corporate suffix stripping (Inc, LLC, Ltd, PLC, GmbH, BV, SA, SAS, Oy, AB, AG, KK, SARL, SRL, SpA, PTY, PTE, PVT, etc.)
3. Punctuation removal + whitespace collapse
4. Tokenization with stopword removal

Two canonical forms:
- `core_name()` — suffix-stripped, punctuation-free (used by fuzzy matching)
- `canonical_name_key()` — maximal-compression exact-match key (used by blocking)

## Fuzzy Similarity (`fuzzy.py`)

Pure-Python, dependency-free implementations:
- **Levenshtein** edit distance (O(m×n) time, O(min(m,n)) memory)
- **Normalized Levenshtein** (0–1 scale)
- **Jaro** similarity
- **Jaro-Winkler** similarity (prefix boost up to 4 chars)
- **Token overlap** coefficient (uses min-set denominator)
- **Jaccard** similarity
- **Composite `name_similarity`**: 0.5 × Jaro-Winkler + 0.3 × token overlap + 0.2 × normalized Levenshtein

## Duplicate Review Queue (`duplicate_review.py`)

Classifies resolution results into three buckets:
- **Automatic merge** — multi-member clusters with high confidence
- **Manual review** — pairs between review and identity thresholds
- **Unrelated** — explicitly split pairs

`build_review_report()` produces a `DuplicateReviewReport` with counts and details.

## CLI Commands

### `predictron-dataset entity-resolve`

```bash
predictron-dataset entity-resolve --dataset ./dataset_store
```

Resolves all stored records into canonical identities. Output:

```json
{
  "identities": [...],
  "report": {
    "report_type": "entity_resolution_report",
    "record_count": 120,
    "identity_count": 95,
    "merged_record_count": 25,
    "duplicate_clusters": [...],
    "confidence_distribution": {"0.9-1.0": 30, "0.8-0.9": 5},
    "merge_statistics": {...}
  }
}
```

### `predictron-dataset duplicate-review`

```bash
predictron-dataset duplicate-review --dataset ./dataset_store
```

Classifies clusters and lists manual-review candidates:

```json
{
  "report_type": "duplicate_review_report",
  "automatic": [...],
  "manual_review": [...],
  "unrelated": [...],
  "counts": {"automatic": 10, "manual_review": 3, "unrelated": 0}
}
```

### `predictron-dataset dedup`

```bash
predictron-dataset dedup --dataset ./dataset_store
```

Exact-key deduplication (legacy, still available).

## Usage (Python API)

```python
from predictron_engine.dataset import EntityResolver, build_review_report

resolver = EntityResolver(
    min_confidence=0.72,
    identity_threshold=0.86,
)
result = resolver.resolve(records)

# Access identities
for identity in result.identities:
    print(identity.canonical_name, len(identity.record_ids))

# Build review report
review = build_review_report(result.report)
for item in review.manual_review:
    print(f"Review: {item.record_ids} (confidence={item.confidence:.2f})")
```

## Performance

Designed for 100k–1M company records:

- **Blocking** eliminates O(n²) comparisons by bucketing records
- **Domain buckets** group companies by canonical domain
- **Name buckets** group by canonical name key (suffix-stripped, compressed)
- **Token buckets** group by individual significant name tokens
- **Fuzzy matching** runs only within buckets, not across the full dataset
- **Union-find** clustering is nearly O(1) per edge with path compression

## Reports

### Entity Resolution Report

| Section | Description |
|---------|-------------|
| `duplicate_clusters` | List of resolved clusters with record IDs, confidence, merge decision |
| `confidence_distribution` | Histogram of pair confidences in 0.1-width buckets |
| `merge_statistics` | Cluster count, merged record count, largest/average cluster size |

### Duplicate Review Report

| Section | Description |
|---------|-------------|
| `automatic` | Clusters auto-merged |
| `manual_review` | Pairs requiring human review |
| `unrelated` | Explicitly split pairs |
| `counts` | Summary counts for each category |

## Constraints

- **Deterministic** — same inputs always produce the same output
- **Explainable** — every match has a human-readable reason and evidence
- **No ML** — all matching is rule-based with explicit thresholds
- **Additive only** — no existing module behavior is modified
- **Backwards compatible** — existing CLI commands and API still work
- **No fabricated data** — only values from member records are used
