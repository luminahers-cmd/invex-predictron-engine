# Sprint 7A — Consistency Audit Report

**Date:** 2026-08-25
**Method:** Automated structural checks + manual review of known patterns

---

## Contradiction Checks

### High Score but Negative Recommendation
**Status: NOT FOUND** — Scores map to decision thresholds monotonically.
Recommendation generation in `strategies/` checks score ≥ thresholds before emitting.
No path exists to produce a high-scoring negative recommendation.

### High Trust with Low Confidence
**Status: NOT FOUND** — Trust and confidence are independent axes by design.
`Observation.trust_score` measures evidence source quality.
`Observation.confidence` measures how strongly evidence supports the conclusion.
A high-truth low-confidence observation is valid (reliable source, ambiguous evidence).

### Strong Evidence with Weak Reasoning
**Status: NOT FOUND** — `evidence_backed.py` computes `evidence_agreement_ratio` and
`evidence_conflict_count` on observations. Strong evidence produces high agreement,
which feeds into `confidence.py:compute_reasoning_confidence`. No path breaks this chain.

### Weak Evidence with Strong Conviction
**Status: NOT FOUND** — Conviction level in `decision_engine.py` is bounded by
`_compute_confidence_factor` which scales down based on observation count and average
confidence. Zero evidence → low conviction.

### Conflicting Trade-Offs
**Status: CHECK** — `synthesis/tradeoffs.py` generates trade-offs from opposing signals.
By design, each trade-off contains one strength and one concern. The `net_assessment`
field labels which side dominates. Multiple trade-offs can have conflicting
`net_assessment` labels — this is intentional and represents genuine tensions
in the data.

### Inconsistent Priorities
**Status: CHECK** — `synthesis/priorities.py` uses a deterministic scoring function
`_priority_weight` that combines category weight, strategy-level priority, confidence,
and rank. The sort is stable and ties fall back to strategy emission order.
No inconsistency risk as long as strategy list order doesn't change.

### Missing Citations
**Status: NOT FOUND** — Every `Observation`, `Recommendation`, `DimensionAssessment`,
`RiskItem`, and `OpportunityItem` carries a `citations: list[EvidenceCitation]` field.
All producers populate this field. Citations default to `[]` (empty list), not missing.

### Broken Provenance Chains
**Status: NOT FOUND** — Traceability chain:
```
Observation.provenance_document_ids → EvidenceItem.citations → EvidenceCitation.source_document_ids
→ EvidenceDocument.id → Provider → Original URL
```
Every link is populated by its producer. The chain is complete.

### Duplicate Risks
**Status: INTENTIONAL OVERLAP** — Low-confidence assessments appear as both
`RiskItem` (`risks.py:338-352`) and as concern side of `TradeOff` (`tradeoffs.py:142-146`).
The `RiskItem` and `TradeOff` are distinct synthesis artifacts with different structures.
`RiskItem.dedup_key` prevents the same risk from appearing twice within the risk register.

### Duplicate Opportunities
**Status: INTENTIONAL OVERLAP** — Similar to risks: high-scoring dimensions appear as
both `OpportunityItem` and as strength side of `TradeOff`. Same dedup applies within
the opportunity register.

---

## Structural Consistency Findings

### 1. Confidence Weight Sum (Bug)
`confidence_engine.py:53-70` — weights sum to 1.05, not 1.00:
```python
WEIGHTS = {
    "data_completeness": 0.25,
    "assessment_confidence": 0.25,
    "evidence_quality": 0.20,
    "feature_coverage": 0.10,
    "observation_quality": 0.10,
    "reasoning_confidence": 0.15,
}
# Sum: 0.25+0.25+0.20+0.10+0.10+0.15 = 1.05
```
The clamp at `confidence_engine.py:95` masks this, but intermediate values exceed [0,1].
**Fix:** Normalize weights to sum to 1.0.

### 2. Risk Threshold Drift
Three hand-maintained copies of risk thresholds:
- `risks.py:34-38`: `runway<6, NRR<90, churn>10, LTV/CAC<1`
- `decision_engine.py:937-957`: Same numbers inline
- `quantitative_signals.py` + `quantitative_cross_signal.py`: Same numbers in reasoning rules

**Status:** All copies currently match. No drift detected. But fragile.

### 3. Recommendation Dedup Gap
`CompositeRecommendationEngine` (`composite.py:60-77`) doesn't deduplicate.
`DefaultRecommendationEngine` (`recommendation_engine.py:112-133`) has dedup via fingerprint.
Production uses `CompositeRecommendationEngine`, so duplicate actions from
different strategies can survive.

**Status:** Current strategies produce distinct actions, so no actual duplicates
observed in practice. But the gap is real.

### 4. Evidence Relevance Convention
Knowledge base facts use `relevance_score=1.0` vs detected signals use `0.8`
(`competition_provider.py:184-186 vs :230`). This is a documented convention,
not an inconsistency, but not explicitly documented in one place.

### 5. Validation Finding Severity Inconsistency
`pipeline_validator.py:163-174` uses `severity="info"` for observations with
confidence < 0.2, but `pipeline_validator.py:241-254` uses `severity="warning"`
for confidence assessments with confidence < 0.2. Same threshold, different severity.

**Status:** Minor, affects debug output only.

---

## Summary

The engine has no major logical contradictions. The confidence weight sum bug
(1.05 vs 1.00) is the only numerical inconsistency. All other findings are
design-level tensions (overlapping synthesis artifacts, parallel threshold
maintenance) that are either intentional or currently aligned.
