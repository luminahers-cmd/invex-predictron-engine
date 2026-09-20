# Decision Intelligence Layer

> Deterministic, fully explainable investment reasoning for the Investron
> Engine — project **E7**. No ML, no LLMs, no embeddings, no probabilistic
> black boxes. Every decision can be traced, explained, calibrated, and
> reproduced bit-for-bit.

## 1. What it is

The Decision Intelligence layer turns a Feature Store snapshot for a company
into a transparent investment verdict together with:

- **Decision Trace** — a deterministic DAG that shows exactly how every raw
  feature flows into the final verdict, with evidence provenance on every
  node.
- **Contributions** — how each feature pushes the decision positive /
  negative / neutral / confidence.
- **Explanations** — human-readable prose that references real evidence,
  plus top strengths and weaknesses.
- **Calibration** — a deterministic adjustment to the reported confidence
  derived from Benchmark Platform ground truth, so confidence is neither
  overstated nor understated.
- **Reports** — deterministic, serializable report objects and a Markdown
  renderer.

It only consumes Feature Store snapshots. It never recomputes raw features
and never mutates them. It consumes (without duplicating) the calibration
outputs produced by the Benchmark Platform.

## 2. Design constraints

| Constraint | How satisfied |
| --- | --- |
| No ML / LLM / embeddings | Pure arithmetic + fixed thresholds |
| Fully deterministic | Same snapshot → identical report/trace output |
| Explainable | Every node carries evidence + a rule string |
| No fabricated reasoning | Explanations reference actual evidence references |
| Backwards compatible | Old `predictron_engine.decision` API kept intact |
| No external APIs | No network calls; operates on local feature data |

Determinism is a first-class property: the same `CompanyFeatureSet` always
yields the same contributions, trace, explanation, calibrated confidence,
and (modulo the report identifier and timestamps) the same report dict.

## 3. Modules

All under `predictron_engine/decision/`.

| Module | Responsibility |
| --- | --- |
| `intelligence_models.py` | Value objects: contributions, traces, explanations, calibration, reports. |
| `feature_engine.py` | `DecisionFeatureEngine` — normalizes Feature Store values, computes weights, feature provenance. |
| `contribution.py` | `ContributionEngine` — per-feature contributions, classification, dimension scores, strengths/weaknesses. |
| `trace.py` | `DecisionTraceEngine` — deterministic reasoning graph + verdict thresholds. |
| `explainability.py` | `ExplainabilityEngine` — prose explanations with real evidence. |
| `calibration_layer.py` | `CalibrationLayer` — deterministic confidence calibration consuming benchmark ground truth. |
| `report.py` | `DecisionReportBuilder` — deterministic serializable reports + Markdown. |
| `service.py` | `DecisionIntelligenceService` — top-level orchestration. |
| `cli.py` | `predictron-decision` five-command CLI. |

## 4. Decision flow

```
Feature Store snapshot
      │
      ▼
DecisionFeatureEngine   ── normalize / weight / provenance
      │
      ▼
ContributionEngine      ── contributions + classification + strengths
      │
      ▼
DecisionTraceEngine     ── reasoning graph + verdict
      │
      ▼
ExplainabilityEngine    ── human wording + recommendations
      │
      ▼
CalibrationLayer        ── adjust confidence from benchmark ground truth
      │
      ▼
DecisionReportBuilder   ── report dict / JSON / Markdown (+ CLI)
```

## 5. Determinism contract

- All contributions, dimension scores, overall scores, verdicts and
  explanations are derived from feature values only.
- Traces use deterministic node ids (`raw:<feature_id>`,
  `intermediate:<feature_id>`, `dimension:<category>`, `overall`,
  `decision`) — no random ids inside the trace.
- The only non-deterministic fields in a report are the report id, report /
  computation timestamps and the calibration timestamp; everything else is
  identical across runs.

## 6. CLI

```
predictron-decision <command> [options]

decision-report    --company-id X [--verdict V | --confidence C]
                   [-o report.json]
decision-trace     --company-id X [--compact]
decision-explain   --company-id X [--verdict V]
decision-compare   [--company-ids A,B] [-o compare.json]
decision-calibrate --expected-conf E --actual-conf A
                   [--history history.json]
```

Example:

```bash
predictron-decision decision-report --company-id rec-123
predictron-decision decision-explain --company-id rec-123 --verdict invest
predictron-decision decision-calibrate --expected-conf 0.80 --actual-conf 0.60
```

## 7. Python API

```python
from predictron_engine.decision import DecisionIntelligenceService

service = DecisionIntelligenceService()
report = service.produce_decision(feature_set)   # CompanyFeatureSet
verdict = report.decision_summary["verdict"]
conf = report.decision_summary["confidence"]

# Trace +
trace = service.produce_trace(feature_set)
# Explanation
explanation = service.produce_explanation(feature_set)
# Calibration
adjustment = service.calibrate(
    expected_confidence=0.8, actual_confidence=0.6,
)
```

## 8. Tests

`tests/decision_intel/` covers models, feature engine, contributions,
trace, explainability, calibration, reports, service, CLI, backwards
compatibility, and determinism — **350+ assertions**. Run:

```bash
python -m pytest tests/decision_intel
python -m ruff check predictron_engine/decision tests/decision_intel
python -m mypy predictron_engine/decision
```
