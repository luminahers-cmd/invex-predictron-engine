"""Tests for the temporal-validity rules in golden dataset validation (V1.4).

The look-ahead guard only makes sense when provenance timestamps are
coherent: a timezone-naive analysis anchor, or an outcome observed before
the analysis, must be rejected; an observation window below the declared
evaluation horizon is warned about — but never repaired.
"""

from __future__ import annotations

from datetime import UTC, datetime

from benchmarks.ground_truth_eval.dataset import validate_golden_dataset
from benchmarks.ground_truth_eval.models import GoldenDataset

ANALYSIS = datetime(2026, 1, 1, tzinfo=UTC)


class TestTemporalValidation:
    def _report(self, entry_builder, **overrides: object):
        entry = entry_builder(company_id="temporal_co")
        for key, value in overrides.items():
            entry = entry.model_copy(update={key: value})
        dataset = GoldenDataset(dataset_name="t", benchmark_version="1", entries=[entry])
        return validate_golden_dataset(dataset)

    def _find(self, report, severity: str) -> list[str]:
        return [i.message for i in report.issues if i.severity == severity]

    def test_naive_analysis_timestamp_is_error(self, entry_builder) -> None:
        report = self._report(
            entry_builder, analysis_timestamp=datetime(2026, 1, 1), evaluation_horizon_days=None
        )
        assert any("timezone-aware" in m for m in self._find(report, "error"))

    def test_outcome_before_analysis_is_error(self, entry_builder) -> None:
        report = self._report(
            entry_builder,
            analysis_timestamp=datetime(2026, 7, 1, tzinfo=UTC),
            evaluation_horizon_days=None,
        )
        assert any("before analysis_timestamp" in m for m in self._find(report, "error"))

    def test_below_horizon_is_warning(self, entry_builder) -> None:
        report = self._report(
            entry_builder,
            analysis_timestamp=ANALYSIS,
            evaluation_horizon_days=365,
        )
        assert any("below evaluation_horizon_days" in m for m in self._find(report, "warning"))

    def test_coherent_pinned_entry_passes(self, entry_builder) -> None:
        report = self._report(
            entry_builder,
            analysis_timestamp=ANALYSIS,
            evaluation_horizon_days=None,
        )
        assert self._find(report, "error") == []
