"""Integration tests for the committed example golden dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.ground_truth_eval.dataset import dataset_hash, validate_golden_dataset
from benchmarks.ground_truth_eval.metrics import compute_metrics
from benchmarks.ground_truth_eval.runner import BenchmarkRunner

# Fingerprint of the committed example_v1.json. Updating the dataset (adding
# cases or correcting outcomes) changes this hash and must be done deliberately.
#
# V1.4 (time-scoped cohort milestone) added the optional GoldenEntry temporal
# anchors (analysis_timestamp / evaluation_horizon_days), which serialize as
# explicit nulls in canonical form, so the canonical hash of every committed
# dataset advanced once. Recomputed as
# dataset_hash(load_golden_dataset("benchmarks/golden_datasets/example_v1.json")).
EXPECTED_DATASET_HASH = "dd7c342c5e8ebd6d43babb688aa151c0eda4e33c294b929699b90d68a6e68a3d"


class TestExampleDataset:
    def test_loads(self, example_dataset) -> None:
        assert example_dataset.dataset_name == "predictron_example"
        assert example_dataset.entry_count == 13

    def test_scoreable_counts(self, example_dataset) -> None:
        assert example_dataset.scoreable_entry_count() == 11
        unscoreable = [e for e in example_dataset.entries if not e.verified_outcome.is_scoreable]
        assert sorted(e.company_id for e in unscoreable) == [
            "ai_infrastructure",
            "devtools",
        ]

    def test_all_marked_example(self, example_dataset) -> None:
        assert all(e.provenance.is_example for e in example_dataset.entries)
        assert all(e.provenance.notes for e in example_dataset.entries)

    def test_every_entry_has_committed_corpus(self, example_dataset) -> None:
        for e in example_dataset.entries:
            corpus = e.historical_evidence.reference
            assert corpus is not None
            path = (
                Path(__file__).resolve().parents[2]
                / "benchmarks"
                / "offline_evidence"
                / f"{corpus}.json"
            )
            assert path.exists(), f"missing corpus for {e.company_id}: {corpus}"

    def test_requests_are_reproducible(self, example_dataset) -> None:
        for e in example_dataset.entries:
            assert e.request["startup_name"] == e.company_name
            assert len(e.request["description"]) >= 100

    def test_decisions_derived_from_historical(self, example_dataset) -> None:
        by_id = {e.company_id: e for e in example_dataset.entries}
        assert by_id["b2b_saas"].historical_prediction.predicted_positive() is True
        assert by_id["deep_tech"].historical_prediction.predicted_positive() is None
        assert by_id["deep_tech"].historical_prediction.decision == "watch"

    def test_dataset_hash_stable(self, example_dataset) -> None:
        assert dataset_hash(example_dataset) == EXPECTED_DATASET_HASH

    def test_validation_passes_with_warnings(self, example_dataset) -> None:
        report = validate_golden_dataset(example_dataset)
        assert report.is_valid is True
        assert report.error_count == 0
        assert report.warning_count >= 2


class TestExampleReplay:
    def test_all_succeeded(self, example_run) -> None:
        assert len(example_run.entries) == 13
        assert len(example_run.successful_entries) == 13

    def test_fresh_metrics_match_fingerprint(
        self, example_dataset, example_run, example_metrics
    ) -> None:
        denied = compute_metrics(example_dataset, example_run)
        assert denied.to_dict() == example_metrics.to_dict()

    def test_confusion_fingerprint(self, example_metrics) -> None:
        cm = example_metrics.confusion
        assert (cm.true_positives, cm.false_positives) == (7, 3)
        assert (cm.true_negatives, cm.false_negatives) == (0, 0)
        assert cm.accuracy == pytest.approx(0.7, abs=1e-6)
        assert cm.recall == pytest.approx(1.0, abs=1e-6)

    def test_engine_is_deterministic(self, example_dataset) -> None:
        r1 = BenchmarkRunner().run(example_dataset, run_id="det_1")
        r2 = BenchmarkRunner().run(example_dataset, run_id="det_2")
        assert r1.result_hash == r2.result_hash

    def test_no_more_than_expected_scores(self, example_run) -> None:
        scores = [e.overall_score for e in example_run.entries if e.overall_score is not None]
        assert len(scores) == 13
        assert all(0 <= s <= 100 for s in scores)
