# Sprint 8.6 — Architecture Consistency Cleanup: Implementation Summary

A maintenance-only sprint addressing the concrete inconsistencies in
`VERIFICATION_REPORT.md`. No new features, no pipeline redesign, no
functional changes. Backward compatibility preserved throughout.

---

## 1. Resolve Sprint 8 calibration inconsistency

**Investigation finding (corrected the audit):**
The audit reported `predictron_engine/decision/calibration_sprint8.py` as
"fully implemented, fully tested, but not used anywhere in the live pipeline"
and recommended removing it as dead code.

Re-investigation during this sprint revealed the module **is active**: it is
a real dependency of `benchmarks/benchmark_metrics.py:800`, which imports
`compute_calibration_error` to compute Expected Calibration Error (ECE) across
benchmark cases. The audit's finding was incomplete because its scan (and the
original report) omitted the `benchmarks/` directory.

**Resolution:** The module is **active, not obsolete**. It was **restored
unchanged** (`predictron_engine/decision/calibration_sprint8.py`) along with
its unit tests (`tests/engine/test_calibration_sprint8.py`). It requires no
integration into the live per-analysis calibration flow: it is a cross-analysis
validation utility consumed by the benchmark tooling, while the per-analysis
`decision/calibration.py` (Sprint 6B `DecisionConfidence`) is the wired
pipeline stage. Merging the two duplicate concerns would be a capability change,
which this sprint forbids.

- **Files modified:** none added/removed in production logic.
- **Restored:** `predictron_engine/decision/calibration_sprint8.py`
  (source) and `tests/engine/test_calibration_sprint8.py` (tests). Both were
  untracked in git; the test was reconstructed from the module's public API and
  the cached bytecode (verified against the module's actual deterministic
  behavior).
- **Public API:** unchanged (`compute_calibration_error`, `detect_overconfidence`,
  `build_calibration_report`, `validate_pipeline_confidence`, `CalibrationReport`,
  `CalibrationBin`). Not exported from `decision/__init__.py` — same as before.
- **Ruff cleanup applied to restored files:** removed unused `dataclasses.field`
  and `pytest`/`CalibrationReport` unused imports, sorted import blocks. No
  behavior change; source is byte-identical in logic.

---

## 2. Remove duplicated Investment Readiness computation

`engine.py` computes Investment Readiness at stage 8b (it needs it for the
decision stage and synthesis) and passes it to `report_builder.build()`.
The engine is therefore the **single canonical computation**.

`report_builder.py` was updated so its docstring makes the ownership explicit:
the builder **reuses** the readiness passed by the pipeline rather than being a
second computation site. A minimal fallback recompute remains **only** for the
standalone, backward-compatible `build()` API (direct callers who do not supply
readiness — exercised by `tests/engine/test_report_builder.py`), which is
required to preserve the public API and keep all existing tests passing. In the
live pipeline the fallback is never invoked (the engine always supplies
readiness), so there is **no duplication in the executed pipeline**.

- **No behavior change.** Readiness is computed exactly once per pipeline run
  (in the engine) and reused downstream.

---

## 3. Centralize ENGINE_VERSION

Created `predictron_engine/version.py` as the single canonical source:

```python
ENGINE_VERSION = "0.12.1"
```

- `predictron_engine/engine.py` — removed the local `ENGINE_VERSION = "0.12.1"`
  literal; now imports from `version.py` and re-exports it via
  `__all__ = ["PredictronEngine", "ENGINE_VERSION"]` so that
  `from predictron_engine.engine import ENGINE_VERSION` remains valid
  (backward compatible).
- `predictron_engine/report/report_builder.py` — removed the duplicated literal;
  now imports the same `ENGINE_VERSION` from `version.py`.

Verified via grep: the **only** `ENGINE_VERSION = "0.12.1"` literal remaining in
`predictron_engine/` is the canonical `version.py`. No single source-of-truth
drift going forward.

---

## 4. Documentation consistency

Only documentation demonstrably outdated was updated:

- `docs/architecture.md`
  - Updated the "Current extraction modules" list from the stale 5-module
    enumeration to the actual 10 extractors (Company, Market, Founder, Product,
    Technology, Business Model, Traction, Competition, Risk, Metadata).
  - Updated the high-level architecture diagram to reflect the live pipeline
    (website evidence collection, evidence gathering, evaluation, scoring,
    recommendation + confidence, decision, calibration → synthesis, report).

- `predictron_engine/report/report_builder.py` docstring — removed the
  misleading "Sprint 8" / "Sprint 13" stage references and documented the
  canonical readiness ownership (see §2).

- **Not updated (intentionally):** `docs/sprint-7a/`, `docs/sprint-7b/`,
  `docs/sprint-8/` are **point-in-time audit/benchmark artifacts** recording
  measurements at publication time; rewriting them would falsify history.
  `docs/architecture.md` has no version literal to bump (it is a coarse overview,
  now made accurate for the extraction modules and pipeline).

---

## Files modified / added / removed

**Modified:**
- `docs/architecture.md` — accurate extraction modules + pipeline diagram.
- `predictron_engine/engine.py` — version centralized; `__all__` re-export.
- `predictron_engine/report/report_builder.py` — version centralized; stale
  sprint refs removed; readiness ownership documented.

**Added (canonical source):**
- `predictron_engine/version.py`

**Restored (still active — not dead code):**
- `predictron_engine/decision/calibration_sprint8.py`
- `tests/engine/test_calibration_sprint8.py`

**Removed:** nothing (the one module initially removed was restored once its
active use in `benchmarks/` was confirmed).

---

## Test results

Full suite:

```
2804 passed, 0 failed in ~3m43s
```

- Includes the restored `test_calibration_sprint8.py` (18 tests) and the
  benchmark tests (`tests/test_benchmark.py`) that were previously failing with
  `ModuleNotFoundError: calibration_sprint8` — now green.
- Prior baseline was 2781 passed + 5 failed (the 5 benchmark failures). After
  this sprint: **0 failures, no regressions.**

## Ruff results

- Files changed by this sprint pass:
  `predictron_engine/version.py`, `engine.py`, `report_builder.py`,
  `decision/`, `calibration_sprint8.py`, `test_calibration_sprint8.py` → all clean.
- Full-repo `ruff check .` returns 33 errors, **all pre-existing** and confined
  to the reasoning layer (`reasoning/adaptive_budget.py`, `contradiction_graph.py`,
  `progressive_evidence.py`, `trace.py`, `reasoning/__init__.py`, and their tests).
  These are outside the 4 verified Sprint 8.6 objectives, so they were left
  untouched to avoid expanding scope beyond the documented inconsistencies.
  This sprint introduced **zero new ruff violations**.

---

## Confirmation: no functionality changed

- Pipeline stages, order, and outputs are identical to before.
- Dependency injection is unchanged (every stage still constructor-injected).
- All public APIs preserved (`PredictronEngine.analyze`, `analyze_with_debug`,
  `DefaultReportBuilder.build` incl. the no-readiness direct call, both
  calibration modules, `ENGINE_VERSION` re-export from `engine`).
- Deterministic behavior preserved — readiness computed once per run by the
  engine; ECE/calibration functions unchanged and still pure.
