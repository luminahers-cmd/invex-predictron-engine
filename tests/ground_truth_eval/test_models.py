"""Tests for the golden dataset data contracts (Project E5 models)."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from benchmarks.ground_truth.schema import FundingStage, StartupStatus
from benchmarks.ground_truth_eval import models as m


class TestDecisionSets:
    @pytest.mark.parametrize("decision", ["strong_invest", "invest"])
    def test_positive_decisions(self, decision: str) -> None:
        assert decision in m.POSITIVE_DECISIONS
        assert m.predicted_positive(decision) is True

    @pytest.mark.parametrize("decision", ["pass"])
    def test_negative_decisions(self, decision: str) -> None:
        assert decision in m.NEGATIVE_DECISIONS
        assert m.predicted_positive(decision) is False

    @pytest.mark.parametrize("decision", ["watch", "investigate_further"])
    def test_neutral_decisions(self, decision: str) -> None:
        assert decision in m.NEUTRAL_DECISIONS
        assert m.predicted_positive(decision) is None

    @pytest.mark.parametrize("decision", ["unknown", "", "invest-other"])
    def test_unrecognized_decision_is_neutral(self, decision: str) -> None:
        assert m.predicted_positive(decision) is None

    def test_none_decision_is_neutral(self) -> None:
        assert m.predicted_positive(None) is None  # type: ignore[arg-type]


class TestHistoricalPrediction:
    def test_predicted_positive_positive(self) -> None:
        p = m.HistoricalPrediction(
            overall_score=78.0, overall_confidence=0.7, decision="strong_invest"
        )
        assert p.predicted_positive() is True

    def test_predicted_positive_negative(self) -> None:
        p = m.HistoricalPrediction(overall_score=20.0, overall_confidence=0.3, decision="pass")
        assert p.predicted_positive() is False

    def test_predicted_positive_neutral(self) -> None:
        p = m.HistoricalPrediction(overall_score=50.0, overall_confidence=0.5, decision="watch")
        assert p.predicted_positive() is None

    @pytest.mark.parametrize("bad_score", [-1.0, 101.0, 700])
    def test_score_bounds_enforced(self, bad_score: float) -> None:
        with pytest.raises(ValidationError):
            m.HistoricalPrediction(
                overall_score=bad_score, overall_confidence=0.5, decision="invest"
            )

    @pytest.mark.parametrize("bad_conf", [-0.1, 1.5])
    def test_confidence_bounds_enforced(self, bad_conf: float) -> None:
        with pytest.raises(ValidationError):
            m.HistoricalPrediction(
                overall_score=50.0, overall_confidence=bad_conf, decision="invest"
            )


class TestHistoricalEvidence:
    def test_reference_prefers_corpus_name(self) -> None:
        e = m.HistoricalEvidence(corpus_name="b2b_saas", corpus_path="some/path.json")
        assert e.reference == "b2b_saas"

    def test_reference_falls_back_to_path(self) -> None:
        e = m.HistoricalEvidence(corpus_path="some/path.json")
        assert e.reference == "some/path.json"

    def test_reference_none_when_both_missing(self) -> None:
        assert m.HistoricalEvidence().reference is None

    def test_sources_defaults_empty(self) -> None:
        assert m.HistoricalEvidence().sources == []


class TestVerifiedOutcome:
    @pytest.mark.parametrize(
        ("status", "exit_value", "expected"),
        [
            ("acquired", 120_000_000, 1),
            ("acquired", 0, 0),
            ("acquired", None, 0),
            ("shutdown", None, 0),
            ("operating", None, None),
        ],
    )
    def test_binary_outcome_by_status(self, status, exit_value, expected) -> None:
        o = m.VerifiedOutcome(
            status=StartupStatus(status),
            verification_date=datetime(2026, 1, 1).date(),
            exit_value_usd=exit_value,
        )
        assert o.binary_outcome() == expected

    def test_operating_with_arr_milestone_is_success(self, entry_builder) -> None:
        entry = entry_builder(status="operating", arr_usd=5_000_000.0)
        assert entry.verified_outcome.binary_outcome() == 1

    def test_operating_without_milestone_is_unscoreable(self, entry_builder) -> None:
        entry = entry_builder(status="operating", arr_usd=None)
        assert entry.verified_outcome.binary_outcome() is None

    def test_is_scoreable_property(self, entry_builder) -> None:
        acquired = entry_builder(status="acquired", exit_value_usd=10_000_000)
        assert acquired.verified_outcome.is_scoreable is True
        alive = entry_builder(status="operating", arr_usd=None)
        assert alive.verified_outcome.is_scoreable is False

    def test_verification_date_linked_to_entry(self, entry_builder) -> None:
        entry = entry_builder()
        assert entry.verification_date == entry.verified_outcome.verification_date


class TestGoldenEntry:
    def test_required_fields(self, entry_builder) -> None:
        entry = entry_builder(company_id="acme_1")
        assert entry.company_id == "acme_1"
        assert entry.request["startup_name"] == "Sample Company"

    def test_metadata_defaults(self, entry_builder) -> None:
        entry = entry_builder()
        assert entry.metadata == {}
        assert entry.provenance.is_example is True

    def test_missing_request_rejected_by_validation(self) -> None:
        from benchmarks.ground_truth_eval.dataset import validate_golden_dataset

        entry = m.GoldenEntry(
            company_id="x",
            company_name="X",
            request={},
            historical_prediction=m.HistoricalPrediction(
                overall_score=50.0, overall_confidence=0.5, decision="invest"
            ),
            historical_evidence=m.HistoricalEvidence(corpus_name="b2b_saas"),
            verified_outcome=m.VerifiedOutcome(
                status=StartupStatus.OPERATING,
                verification_date=datetime(2026, 1, 1).date(),
            ),
        )
        dataset = m.GoldenDataset(
            dataset_name="t",
            benchmark_version="1.0",
            entries=[entry],
        )
        report = validate_golden_dataset(dataset)
        assert not report.is_valid
        assert any("startup_name" in i.message for i in report.issues)


class TestGoldenDataset:
    def test_entry_count(self, entry_builder) -> None:
        d = m.GoldenDataset(dataset_name="t", benchmark_version="1.0", entries=[entry_builder()])
        assert d.entry_count == 1

    def test_entry_by_id(self, entry_builder) -> None:
        e = entry_builder(company_id="alpha")
        d = m.GoldenDataset(dataset_name="t", benchmark_version="1.0", entries=[e])
        assert d.entry_by_id("alpha") is e
        assert d.entry_by_id("missing") is None

    def test_scoreable_entry_count(self, entry_builder) -> None:
        stats = [
            ("c1", "acquired", None, 10_000_000),
            ("c2", "shutdown", None, None),
            ("c3", "operating", None, None),
            ("c4", "operating", 2_000_000, None),
        ]
        entries = [
            entry_builder(
                company_id=cid,
                status=status_c,
                arr_usd=arr_c,
                exit_value_usd=exit_c,
            )
            for cid, status_c, arr_c, exit_c in stats
        ]
        d = m.GoldenDataset(dataset_name="t", benchmark_version="1.0", entries=entries)
        assert d.scoreable_entry_count() == 3

    def test_created_at_defaults_to_utc(self) -> None:
        d = m.GoldenDataset(dataset_name="t", benchmark_version="1.0")
        assert d.created_at.tzinfo is not None

    def test_schema_version_constant(self) -> None:
        assert m.GOLDEN_DATASET_SCHEMA_VERSION == 1
        assert m.GROUND_TRUTH_SCHEMA_VERSION == 1


class TestImmutabilityContract:
    def test_golden_entry_is_not_touched_by_runner(self, example_dataset) -> None:
        """The runner must never mutate historical records."""
        from benchmarks.ground_truth_eval.runner import BenchmarkRunner

        before = example_dataset.model_dump()
        runner = BenchmarkRunner()
        runner.run(example_dataset, run_id="pytest_no_mutate")
        after = example_dataset.model_dump()
        assert before == after

    def test_entry_model_dump_roundtrip(self, entry_builder) -> None:
        e = entry_builder()
        clone = m.GoldenEntry.model_validate(e.model_dump())
        assert clone == e

    def test_funding_stage_used_in_outcome_derivation(self) -> None:
        assert FundingStage.SEED.value == "seed"
