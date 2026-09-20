"""Ground Truth Evaluation Platform (Project E5).

A deterministic, immutable, backwards-compatible evaluation platform that
continuously measures how well the Predictron engine performs against
real-world startup outcomes.

The platform is a *consumer* of ground truth: it never fabricates labels,
never hallucinates outcomes, and never modifies historical records.  It:

* loads curated ``GoldenDataset`` versions (company id, historical
  prediction, historical evidence, verified outcome, verification date,
  provenance),
* replays the current engine over the dataset using deterministic offline
  evidence corpora (``BenchmarkRunner``),
* computes deterministic evaluation metrics (``GroundTruthMetrics``),
* detects explainable drift across runs/versions (``DriftDetector``),
* persists every execution in an append-only history (``BenchmarkHistory``),
* generates deterministic reports,
* exposes a ``predictron-benchmark`` CLI.

Modules:
    models:   Golden dataset data contracts.
    dataset:  Loading, validation, and hashing of golden datasets.
    runner:   Deterministic benchmark execution over a golden dataset.
    metrics:  Ground-truth evaluation metrics (deterministic).
    drift:    Explainable drift detection between runs/versions.
    history:  Append-only benchmark execution history.
    reports:  Deterministic report generation.
    export:   JSON/markdown export helpers.
    cli:      ``predictron-benchmark`` command-line interface.
"""

from __future__ import annotations
