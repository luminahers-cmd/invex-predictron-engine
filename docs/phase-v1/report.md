# Phase V1 — Real-World Prediction Validation Framework (Design Report)

**Status:** Design only — no implementation, no data population.
**Scope:** Enable scientifically valid evaluation of whether the Predictron
Engine makes good investment decisions on real startups.
**Constraint compliance:** No engine/scoring/reasoning/recommendation/
threshold/confidence changes. No benchmark expectation changes. No snapshot
changes. No fabricated outcome labels, no downloaded datasets.

This report covers the five workstreams:

- **V1A** — Ground-Truth Dataset Schema
- **V1B** — Evaluation Metrics
- **V1C** — Validation Pipeline
- **V1D** — Recommended Data Sources
- **V1E** — Architecture Review & Smallest Additions

A single inert code artifact accompanies this report:
`benchmarks/ground_truth/schema.py` (+ `tests/test_ground_truth_schema.py`).
It is a **pure data-contract definition** with no engine coupling, no loader,
no metrics, and no side effects. It exists so the schema is auditable and
versioned in code, and is the concrete instance of the "smallest architectural
addition" called for in **V1E**.

---

## Why this framework exists

The current benchmark suite (`benchmarks/`, snapshots in
`benchmarks/expected_outputs/`, corpus in `benchmarks/offline_evidence/`)
measures **deterministic consistency** of pipeline outputs across synthetic
startup cases. It answers: *"Does the engine produce stable, explainable,
well-formed output?"* It does **not** answer: *"Are the engine's investment
decisions correct against reality?"*

Phase V1 introduces a **second, orthogonal evaluation axis**: prediction
validity against ground truth. It deliberately reuses the benchmark
infrastructure for *running* the engine, but adds new artifacts only for
*scoring predictions against observed outcomes*.

---

## V1A — Ground-Truth Dataset Schema

**Design principle:** a *prediction* and its *ground truth* must be
unambiguous about (a) the analysis-time context, (b) the engine prediction,
(c) the observed outcome, and (d) the evaluation horizon. Every record is
attributable to named sources — nothing is fabricated.

### Canonical record

A dataset is a sequence of `GroundTruthRecord` JSON objects (one per analysed
startup). Full definition is in `benchmarks/ground_truth/schema.py`. The
schema is versioned (`GROUND_TRUTH_SCHEMA_VERSION = 1`) and stored as
JSON-lines for append-only, commit-friendly, diffeable records.

| Group | Field | Type | Notes |
|-------|-------|------|-------|
| **Identity** | `startup_id` | str | stable unique identifier |
| | `startup_name` | str | display name |
| | `startup_website` | str? | join key used by evidence collection / YC / Crunchbase |
| | `industry_category` | str? | coarse industry |
| | `geography` | str? | HQ country / region |
| **Analysis context** | `analysis_timestamp` | datetime(UTC) | when the engine analysed the startup |
| | `funding_stage_at_analysis` | enum | idea…growth (mirrors engine stages) |
| **Engine prediction** | `engine_snapshot` | `EngineSnapshot` | decision category, conviction, composite score, overall confidence, decision confidence, engine version, optional full-report path |
| **Ground truth** | `actual_status` | enum | operating / acquired / shutdown / unknown |
| | `outcome_observation_date` | date | when status was last verified |
| | `evaluation_horizon_days` | int | analysis→observation delta (horizon closed) |
| | `outcome_events[]` | list | verified funding / follow-on / acquisition / shutdown / ARR milestone / employee growth / investor participation events |
| | `exit_value_usd` | float? | reported exit value |
| **Provenance** | `recorded_at` | datetime | dataset write time |
| | `sources[]` | list[str] | source identifiers from the V1D catalogue |
| | `schema_version` | int | schema version |

### Outcome events (longitudinal)

`outcome_events` is the temporal spine for follow-up-round, ARR-milestone,
employee-growth and investor-participation evaluation. Each event has a
`kind`, an `occurred_at` date, an optional monetary value, an optional scalar
(e.g. headcount), a free-text description, and `sources[]`.

### Binary success label (explicit, not fabricated)

`derive_binary_outcome(record) -> 0 | 1 | None` is provided as a **pure
function** and returns `None` (un-scoreable) whenever the outcome is
ambiguous rather than inventing a label:

- `acquired` → `1` iff `exit_value_usd > 0`, else `0`.
- `shutdown` → `0`.
- `operating` → `1` iff a positive ARR milestone exists in the horizon; else
  `None` (mere survival is not success).
- `unknown` → `None`.

**Why `None` matters:** the V1B metrics MUST treat `None` as "insufficient
ground truth — exclude", never as a fabricated 0/1. This is the anti-pattern
guard against inventing outcome labels.

### Evaluation horizon & longitudinal design

The schema is longitudinal because a startup is scored at `analysis_timestamp`
with `funding_stage_at_analysis` as an explicit input, and its outcome is
read later (months/years). The engine must be run on **information available
at the analysis date** — this is the key temporal-integrity requirement
(see V1C). Future records can append new outcome events without rewriting
history, so accuracy is measured at fixed horizons (`evaluation_horizon_days`)
rather than against a moving present.

---

## V1B — Evaluation Metrics

**Cardinal rule:** every metric states the exact ground-truth columns it
requires and is **undefined (not zero, not heuristically filled) when that
ground truth is absent.** No metric below is ever computed from synthetic
outcomes.

Metrics consume two sources:
1. **Engine prediction** — from `engine_snapshot` (decision category,
   composite score, confidence) or the full serialized `Report`.
2. **Ground truth** — `derive_binary_outcome` labels and `outcome_events`.

### Decision/threshold metrics (binary ground truth required)

| Metric | Ground truth required | Definition / remarks |
|--------|----------------------|----------------------|
| **Precision** | binary label + decision | TP / (TP+FP) among invest/strong_invest predictions |
| **Recall** | binary label + decision | TP / (TP+FN) over true successes |
| **F1 / Fβ** | binary label + decision | harmonic mean; F0.5 weights precision (VC-flavoured, cf. VCBench) |
| **Decision accuracy** | binary label + decision | (TP+TN)/N; report base rate separately (class imbalance is expected) |
| **False positive cost** | binary label + decision | counts/cost of investing in a failed startup; requires an agreed cost model or simple FP count |
| **False negative cost** | binary label + decision | cost of passing on a success; FN count or forgone-return model |

### Ranking/score metrics (ordered composite score required)

| Metric | Ground truth required | Definition / remarks |
|--------|----------------------|----------------------|
| **ROC / AUC** | binary label + composite score rank | area under ROC over score thresholds |
| **Precision-recall curve / AUPRC** | binary label + score rank | preferred for rare-success domains |

### Calibration metrics (confidence + outcome required)

| Metric | Ground truth required | Definition / remarks |
|--------|----------------------|----------------------|
| **Brier score** | binary label + probability | mean squared error of predicted vs actual probability |
| **Expected Calibration Error (ECE)** | binary label + confidence | reuse `compute_calibration_error` from `calibration_sprint8.py` (already deterministic); bins by predicted confidence |
| **Reliability diagram** | binary label + confidence | per-bin mean-confidence vs actual-accuracy |
| **Confidence calibration** | binary label + confidence | e.g. calibration slope / correlation between predicted prob and empirical frequency (replaces the current completeness-proxy metric for *prediction* purposes) |

### Simulation metrics

| Metric | Ground truth required | Definition / remarks |
|--------|----------------------|----------------------|
| **Investment ROI simulation** | raw outcome events + exit_value_usd (+ a transparent allocation rule) | back-test a portfolio: invest where decision says invest, apply exit/ARR outcomes, compute IRR/MOIC. Requires an explicit, documented allocation & valuation rule; not run until the dataset has enough matured records |

### Explicit metric → ground-truth contract

Each metric is defined by `(required_inputs, exclusion_rule)`. For example:

- `precision(y_pred_binary, y_true_binary)` → both required; skip if no
  invest predictions.
- `roc_auc(score, y_true)` → sortable score + binary labels with `None`
  excluded.
- `brier_score(prob, y_true)` → probability in [0,1] + binary labels.

No metric is computed if `derive_binary_outcome` returns `None` for any of its
samples; insufficient-data metrics are reported as **"insufficient
ground-truth"**, not as 0.

---

## V1C — Validation Pipeline

The pipeline reuses the **existing benchmark runner** for engine execution and
adds a thin evaluation stage that consumes ground truth.

### Workflow

1. **Load historical cases** — a curated list of real startups with a known
   later outcome (`GroundTruthRecord` metadata), each with startup name +
   website + a snapshot of analysis-time inputs.
2. **Run the engine** at the historical `analysis_timestamp` context using the
   existing `run_benchmark()` / `build_replay_engine()` path
   (`benchmarks/benchmark_runner.py`). Evidence is the *historical* corpus, so
   results are reproducible and free of look-ahead bias (reuse
   `offline_evidence/` replay or a time-scoped corpus).
3. **Align prediction ↔ ground truth** by `startup_id`, producing per-record
   `(engine_snapshot_vs_observed_outcome)` pairs.
4. **Compare predictions against known outcomes** via the V1B metrics.
5. **Generate evaluation reports** reusing the `benchmark_report.py`
   formatting pattern (text + JSON), with a new "prediction-validity" section
   distinct from the existing consistency sections.
6. **Track regressions over time** by persisting metric results per dataset
   revision and diffing (mirroring the snapshot-diff approach), so an engine
   change that improves consistency but worsens prediction accuracy is
   surfaced.

### Integration with the benchmark architecture

- **Reuse:** `run_benchmark`, `CaseResult`, snapshot save/load, report
  generators, `offline_evidence` replay, CI wiring.
- **Add (small):** a ground-truth loader (reads `GroundTruthRecord` JSON-lines),
  a metric evaluator, and a report renderer — all *outside* the engine, under
  `benchmarks/` (see V1E).
- **No change:** engine, scoring, reasoning, threshold, confidence, snapshot,
  or existing benchmark expectations.

### Temporal-integrity (look-ahead bias) guard

The single most important correctness requirement: an engine run for a record
must only "see" evidence dated **at or before** `analysis_timestamp`. The
schema's `analysis_timestamp` + source-provenance design makes this
enforceable. Populating the dataset therefore requires time-scoped corpora,
which the existing replay mechanism is well suited to support.

---

## V1D — Dataset Sources (Researched, not downloaded)

| Source | Availability | Licensing | Reliability | Update frequency | Suitability |
|--------|--------------|-----------|-------------|------------------|-------------|
| **SEC EDGAR** (10-K/10-Q/8-K/S-1, XBRL; `data.sec.gov`, `efts.sec.gov`) | Free, no key; full-text since 2001; rate-limit ~10 req/s and a descriptive `User-Agent` required | Public domain | Very high (authoritative regulator filings, incl. delisted) | Daily, near-real-time | **High** for later-stage (filers); revenue/ARR via XBRL; validation/corroboration of exits |
| **YC companies** (`yc-oss/api`, YC directory) | Free, no key, community-maintained API | Factual public data; derived/measured columns ~CC BY 4.0 | Medium-high; **survivorship caveat** and status lag | Daily (directory; snapshots per refresh) | **High for early-stage cohort** — batch + status (Active/Acquired/Public/Inactive) + team size; strongest free cohort source |
| **VCBench** (`vcbench.com`, Vela Partners) | Public train split; private test held out | Commercial; contest-gated private labels | High (published benchmark, arXiv 2509.14448) | Static benchmark revision | Medium for methodology/metrics reference (F0.5, precision-first) and curated positives |
| **PHBench** (`phbench.com`, HuggingFace `ihlamury/phbench`) | Gated access on HuggingFace | CC BY 4.0 | High (peer-use benchmark) | Static | Medium — launch-signal prediction of Series A, useful as a complementary cohort |
| **IdeaProof open datasets** (failure DB, failure base rates) | Free CSV/JSON | CC BY 4.0 | Medium; curated failures with per-fact sources | Periodic | **High** for failures/shutdowns + failure base rates (needed for FP/FN cost priors) |
| **Kaggle startup datasets** (e.g. Crunchbase-exported status sets) | Free | Mostly permissive (varies); verify per file | **Low–medium** — stale (often 1990–2013), survivorship/curation caveats | Static/activity-driven | Low for present-day decisions; useful for methodological prototyping only |
| **TechCrunch / funding announcements** (web, via the engine's existing search providers) | Publicly accessible prose | Text is factual reporting; brief quotes acceptable for sourcing (not bulk redistribution) | Medium-high for major rounds | Continuous | **High** as a corroborating source for funding/ARR/exit events; already reachable via the engine's evidence stack |
| **Crunchbase** (API is a *license/paywall case study*) | **Paid** — free API tier removed in 2026; ~$588–$1,188+/yr entry, sales-led above; Venture Program free for eligible VCs | Proprietary license agreement | High on covered firms; coverage/update bias toward funded companies | Near-real-time | **High value but paid** — strongest commercial ground-truth for follow-on rounds/investor participation; candidate later |
| **Government business registries** (e.g. Delaware/CA SOS, UK Companies House API) | Companies House API free; US SOS typically paid-document | Derived factual data; Companies House open | High for incorporation/dissolution dates | High (dissolution events timely) | **Medium** — excellent for terminal events (shutdowns), poor for revenue/ARR |

### Recommended primary stack for the first dataset

1. **YC cohort** (early-stage, high survival/base-rate value, free) as the
   backbone, matched to
2. **IdeaProof + government registries** for shutdown/terminal events,
3. **SEC EDGAR + TechCrunch** for revenue/ARR and exit corroboration,
4. **Crunchbase API later** (paid) for investor-participation / follow-on
   depth once the framework proves out.

### Verification of access

I verified via public sources (2026) that: SEC EDGAR is free with no key
(rate-limited, `User-Agent` required); `yc-oss/api` is free and key-less;
VCBench/PHBench are benchmark-gated; Crunchbase's free API tier **was
removed** and is now paid/sales-led — so the report does **not** assume free
Crunchbase access. Nothing was downloaded into the repository.

---

## V1E — Architecture Review: Smallest Additions

**No new infra (no Redis/K8s/external store), no rewrite.** The engine,
scoring, reasoning, confidence, recommendations, thresholds, snapshots, and
existing benchmarks are untouched. The evaluation axis is additive.

### What already exists that we reuse

- `benchmarks/benchmark_runner.py` — `run_benchmark`, `CaseResult`,
  `build_replay_engine`, offline replay → reuse for running predictions.
- `benchmarks/offline_evidence/` + `evidence/replay/dataset.py` — deterministic
  time-scoped evidence replay → the temporal-integrity mechanism.
- `benchmarks/benchmark_metrics.py` — metric container/pattern (`MetricResult`)
  to mirror.
- `benchmarks/benchmark_report.py` — report + diff formatting pattern.
- `predictron_engine/decision/calibration_sprint8.py` —
  `compute_calibration_error` (deterministic ECE) → reuse directly.
- `.github/workflows/ci.yml` — snapshot-diff gate → can add a metric-diff gate.

### Proposed new module (design-only now)

```
benchmarks/ground_truth/
  schema.py      # GroundTruthRecord + enums + derive_binary_outcome  (DONE, inert)
  __init__.py    # re-exports
tests/test_ground_truth_schema.py   # 7 contract tests (DONE)
```

Planned, **not yet implemented** (green-lighted in a later phase after the
dataset exists):

```
benchmarks/ground_truth/
  loader.py        # read GroundTruthRecord JSON-lines (validate via schema)
  evaluator.py     # compute V1B metrics; every metric guards ground-truth
  report.py        # render prediction-validity report (mirrors benchmark_report)
  evaluate.py      # CLI: --dataset <path> --engine-replay <corpus> ...
  cli-integration  # optional: a `--eval-ground-truth` flag on benchmark_report
tests/  (loader, evaluator, report unit tests; no engine changes)
```

### Why this is the smallest change

- The ground truth lives **outside** the engine as immutable JSON — the engine
  remains a pure function used by both the existing consistency benchmarks and
  the new prediction benchmark.
- Metrics are **pure functions** of engine output + ground truth; they call
  existing deterministic `compute_calibration_error` where possible.
- No changes to `app/`, `predictron_engine/`, existing `benchmarks/`, or any
  snapshot. CI can get an additive optional gate.

### Anti-goals (explicitly out of scope)

- Do NOT couple ground truth into the engine or the live API.
- Do NOT compute metrics with synthetic/heuristic outcomes (the current
  `benchmark_metrics._calibration_error` uses a completeness-proxy — that is a
  *consistency* metric and stays; prediction metrics require real labels).
- Do NOT add infra, queues, or a database; JSON-lines + the existing file-based
  snapshot pattern are sufficient at this scale.

---

## Implementation Roadmap

| Step | Phase | Deliverable | Effort |
|------|-------|-------------|--------|
| 0 | V1 (this) | Schema + contract tests + design report | ~0.5–1 dev-day (done) |
| 1 | V1.1 | `loader.py` (validate/dedupe records) | 1 dev-day |
| 2 | V1.2 | `evaluator.py` (all V1B metrics with ground-truth guards) | 2–3 dev-days |
| 3 | V1.3 | `report.py` + CLI + CI gate | 1–2 dev-days |
| 4 | V1.4 | Retro-populate *first* ground-truth cohort (YC-derived) for a **dated** window (≥3–5 yrs old) | 3–5 dev-days incl. source verification |
| 5 | V1.5 | Run full evaluation; calibration-vs-consistency split report | 1 dev-day |
| 6 | V1.6 | Crawl: add Crunchbase (paid) for funding depth; expand to SEC/EdTech/climate cohorts | ongoing |

Effort is estimates at typical senior-engineer rates; V1.4 is the largest
single item because of manual source verification and temporal-integrity
checks.

---

## Risks

1. **Survivorship & selection bias** — public datasets (esp. Crunchbase-derived
   and YC) over-weight funded/live companies. Mitigation: include IdeaProof
   failures + government dissolution records; report base rates alongside
   every metric; report the exclusion of `unknown`/unverifiable records.
2. **Look-ahead bias** — using post-analysis evidence in the engine run.
   Mitigation: time-scoped replay corpus, `analysis_timestamp`-gated sources
   (enforced by loader/evaluator).
3. **Label ambiguity** — "operating without ARR proof" / low-value exits.
   Mitigation: `derive_binary_outcome` returns `None`; no fabricated labels.
4. **Temporal staleness of ground truth** — status columns lag reality.
   Mitigation: the schema's `outcome_observation_date` records what was
   verifiable *then*; re-observation appends rather than mutating.
5. **Small-N statistics** — a first cohort may be only a few hundred mature
   records, making AUC/ECE noisy. Mitigation: bootstrapped confidence
   intervals around metrics; report sample counts; prefer early-stage binary
   outcome base rates over precision claims.
6. **Licensing/paywall** — Crunchbase free API removed (2026). Mitigation: the
   stack above proceeds with free sources; Crunchbase is deferred.
7. **Regime drift** — 2010-era data poorly predicts 2026. Mitigation: fix
   **evaluation windows**, not the whole history; stratify by vintage.
8. **Metric won't gate on None** — the biggest correctness risk in code.
   Mitigation: a contract test asserting insufficient-data metrics report
   "insufficient ground-truth" (added in V1.2).

## Estimated engineering effort (total)

- **V1 report + inert schema:** ~0.5–1 dev-day (this sprint, delivered).
- **Full framework (V1.1–V1.6):** ~8–13 dev-days over subsequent phases,
  dominated by ground-truth retro-population and verification.

---

## Acceptance criteria status

- **No production prediction behavior changes:** ✅ — no engine/app changes.
- **No benchmark expectation changes:** ✅ — existing `expected_outputs/`
  snapshots and metrics untouched.
- **No snapshot changes:** ✅ — none regenerated; `git status` shows only new
  files + this report.
- **No scoring changes:** ✅.
- **Focus on scientific validity:** ✅ — schema separates prediction from
  outcome; metrics demand explicit ground truth and never fabricate labels;
  pipeline guards temporal integrity; sources are real and verified.

## Verification performed

- New module imports and passes `ruff` and `mypy` (strict).
- `tests/test_ground_truth_schema.py`: 7 tests pass (contract-only).
- Existing test suite re-run to confirm no regressions (see final check in the
  delivery summary).
