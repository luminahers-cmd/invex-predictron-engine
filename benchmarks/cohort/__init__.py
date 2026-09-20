"""Cohort builder for the Ground Truth Evaluation Platform (Milestone V1.4).

Deterministic, provenance-preserving pipeline that turns a *cohort manifest*
(real inputs + time anchors + best-known outcomes) into a golden dataset for
:mod:`benchmarks.ground_truth_eval`.  Every manifest record contributes:

* the analysis-time ``request`` (name, website, description, ...),
* a committed offline evidence corpus reference,
* a time-scoped, engine-run **historical prediction** pinned to
  ``analysis_timestamp`` (the look-ahead guard blocks corpus evidence that
  postdates the analysis instant),
* a best-known ``verified_outcome`` with temporal anchors and provenance.

Nothing fabricates labels: a record without a documented outcome is emitted
as *pending ground truth* and excluded from the dataset; an outcome not yet
independently verified is recorded verbatim but flagged ``is_example`` in its
provenance (the platform's existing non-derogation of ground truth).
"""

from __future__ import annotations

from benchmarks.cohort.manifest import (
    MANIFEST_SCHEMA_VERSION,
    CohortManifest,
    ManifestOutcome,
    ManifestSourceRecord,
)

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "CohortManifest",
    "ManifestOutcome",
    "ManifestSourceRecord",
]
