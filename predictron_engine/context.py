"""Immutable execution context for a single analysis run.

The AnalysisContext carries cross-cutting state through the pipeline —
the correlation id, the normalized startup, and any website evidence
collected before feature extraction. Stage dependencies that want the
context accept it as a read-only value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from predictron_engine.evidence.models import EvidenceBundle
from predictron_engine.models.startup import Startup


@dataclass(frozen=True)
class AnalysisContext:
    """Read-only context shared across all pipeline stages.

    Parameters
    ----------
    request_id:
        Correlation id for the analysis run (logs and tracing).
    startup:
        The normalized startup being analyzed.
    evidence_bundle:
        Website evidence collected before extraction. Empty bundle when
        no website was provided or collection was skipped.
    metadata:
        Arbitrary per-run metadata set by stages that is not part of the
        public report. Never mutated after construction.
    """

    request_id: str
    startup: Startup
    evidence_bundle: EvidenceBundle = field(
        default_factory=lambda: EvidenceBundle.empty("")
    )
    metadata: dict[str, Any] = field(default_factory=dict)
