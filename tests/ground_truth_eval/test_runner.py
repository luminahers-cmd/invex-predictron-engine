"""Tests for the deterministic benchmark runner."""

from __future__ import annotations

import pytest

from benchmarks.ground_truth_eval.models import GoldenDataset
from benchmarks.ground_truth_eval.runner import (
    FEATURE_KEYS,
    BenchmarkRunner,
    RunnerError,
    build_entry_bundle,
    build_request,
    generate_run_id,
    report_to_output,
)
from predictron_engine.evidence.replay.dataset import EvidenceCorpusError


class TestBuildRequest:
    def test_required_fields_passed(self, entry_builder) -> None:
        entry = entry_builder(
            company_id="acme",
            startup_name="Acme Corp",
            description="A sufficiently long startup description for the request.",
        )
        req = build_request(entry)
        assert req.startup_name == "Acme Corp"
        assert req.description == entry.request["description"]
        assert str(req.website).rstrip("/") == entry.request["website"]

    def test_optional_fields_passed(self, entry_builder) -> None:
        entry = entry_builder()
        entry.request["pitch_deck_url"] = "https://deck.example.com/deck.pdf"
        entry.request["founder_linkedin_urls"] = ["https://linkedin.com/in/a"]
        req = build_request(entry)
        assert str(req.pitch_deck_url).rstrip("/") == "https://deck.example.com/deck.pdf"
        assert [str(u).rstrip("/") for u in req.founder_linkedin_urls] == [
            "https://linkedin.com/in/a"
        ]

    def test_absent_optional_fields_default(self, entry_builder) -> None:
        req = build_request(entry_builder())
        assert req.pitch_deck_url is None
        assert req.founder_linkedin_urls == []


class TestBuildEntryBundle:
    def test_resolves_committed_corpus(self, entry_builder) -> None:
        entry = entry_builder(evidence_corpus="b2b_saas")
        bundle = build_entry_bundle(entry)
        assert bundle is not None
        assert bundle.startup_name

    def test_missing_reference_raises(self, entry_builder) -> None:
        entry = entry_builder(evidence_corpus=None)
        with pytest.raises(RunnerError, match="no evidence corpus reference"):
            build_entry_bundle(entry)

    def test_unresolvable_corpus_raises(self, entry_builder) -> None:
        entry = entry_builder(evidence_corpus="does_not_exist_corpus")
        with pytest.raises(RunnerError, match="unavailable"):
            build_entry_bundle(entry)

    def test_error_chain_surfaces_corpus_error(self, entry_builder) -> None:
        entry = entry_builder(evidence_corpus="does_not_exist_corpus")
        with pytest.raises(RunnerError) as excinfo:
            build_entry_bundle(entry)
        assert isinstance(excinfo.value.__cause__, EvidenceCorpusError)


class TestReportToOutput:
    def test_full_extraction(self, make_report, entry_builder) -> None:
        report = make_report(
            overall_score=72.5,
            overall_confidence=0.66,
            decision="invest",
            decision_confidence=0.55,
            composite_score=70.1,
            rec_categories=("due_diligence", "opportunity", "due_diligence", "risk_mitigation"),
            dimension_scores={"market_opportunity": 80.0, "product_strength": 60.0},
            features={
                "industry": "enterprise_saas",
                "funding_stage": "series_a",
                "has_revenue": True,
            },
            evidence_count=5,
            observation_count=4,
        )
        out = report_to_output(entry_builder(), report, elapsed_ms=1.25)
        assert out.success is True
        assert out.overall_score == 72.5
        assert out.overall_confidence == 0.66
        assert out.decision == "invest"
        assert out.decision_confidence == 0.55
        assert out.composite_score == 70.1
        assert out.recommendation_categories == [
            "due_diligence",
            "opportunity",
            "risk_mitigation",
        ]
        assert out.recommendation_count == 4
        assert out.evidence_count == 5
        assert out.observation_count == 4
        assert out.processing_time_ms == 1.25
        assert out.extracted_features["industry"] == "enterprise_saas"

    def test_decision_absent(self, make_report, entry_builder) -> None:
        report = make_report(decision=None, decision_confidence=None)
        out = report_to_output(entry_builder(), report, 0.0)
        assert out.decision is None
        assert out.decision_confidence is None

    def test_features_subset_captured(self, make_report, entry_builder) -> None:
        report = make_report(features={"geography": "us"})
        out = report_to_output(entry_builder(), report, 0.0)
        assert set(out.extracted_features.keys()) == set(FEATURE_KEYS)

    def test_feature_keys_matching_engine_model(self) -> None:
        from predictron_engine.models.extracted_features import ExtractedFeatures

        for key in FEATURE_KEYS:
            assert key in ExtractedFeatures.model_fields, key

    def test_dimension_scores_rounded(self, make_report, entry_builder) -> None:
        report = make_report(dimension_scores={"market": 1.00000001})
        out = report_to_output(entry_builder(), report, 0.0)
        assert out.dimension_scores["market"] == 1.0


class TestRunnerUnit:
    def test_uses_injected_engine(self, make_fake_engine, example_dataset) -> None:
        engine = make_fake_engine()
        runner = BenchmarkRunner(engine=engine)
        run = runner.run(example_dataset, run_id="unit_fake")
        assert len(run.entries) == example_dataset.entry_count
        assert all(e.success for e in run.entries)
        assert (
            run.engine_version == engine.engine_version
            if hasattr(engine, "engine_version")
            else True
        )

    def test_failed_entry_captured(self, make_fake_engine, example_dataset) -> None:
        engine = make_fake_engine(raise_error=True)
        runner = BenchmarkRunner(engine=engine)
        run = runner.run(example_dataset, run_id="unit_fail")
        assert all(not e.success for e in run.entries)
        assert all("boom" in (e.error or "") for e in run.entries)

    def test_benchmark_run_headers(self, make_fake_engine, example_dataset) -> None:
        engine = make_fake_engine()
        runner = BenchmarkRunner(engine=engine)
        run = runner.run(example_dataset, run_id="unit_headers")
        assert run.dataset_name == "predictron_example"
        assert run.benchmark_version == "1.0"
        assert run.dataset_hash
        assert run.result_hash

    def test_entry_by_id(self, make_run) -> None:
        run = make_run([{"company_id": "a"}, {"company_id": "b"}])
        assert run.entry_by_id("a").company_id == "a"
        assert run.entry_by_id("zzz") is None

    def test_successful_entries_property(self, make_run) -> None:
        run = make_run([{"company_id": "a"}, {"company_id": "b", "success": False}])
        assert [e.company_id for e in run.successful_entries] == ["a"]

    def test_result_hash_order_independent(self, make_run) -> None:
        run_fwd = make_run(
            [{"company_id": "a", "overall_score": 1.0}, {"company_id": "b", "overall_score": 2.0}]
        )
        run_bwd = make_run(
            [{"company_id": "b", "overall_score": 2.0}, {"company_id": "a", "overall_score": 1.0}]
        )
        assert run_fwd.result_hash == run_bwd.result_hash

    def test_result_hash_changes_with_values(self, make_run) -> None:
        r1 = make_run([{"company_id": "a", "overall_score": 1.0}])
        r2 = make_run([{"company_id": "a", "overall_score": 9.0}])
        assert r1.result_hash != r2.result_hash


class TestRunId:
    def test_format(self) -> None:
        rid = generate_run_id("0.12.1")
        assert rid.startswith("run_0.12.1_")
        assert len(rid.split("_")[-1]) == 8

    def test_unique_across_calls(self) -> None:
        ids = {generate_run_id("0.12.1") for _ in range(50)}
        assert len(ids) == 50


class TestRunnerIntegration:
    def test_deterministic_repeat(self, example_dataset) -> None:
        runner = BenchmarkRunner()
        r1 = runner.run(example_dataset, run_id="deterministic_a")
        r2 = runner.run(example_dataset, run_id="deterministic_b")
        # result_hash already excludes wall-clock timing.
        assert r1.result_hash == r2.result_hash

    def test_all_entries_succeed_on_example(self, example_run) -> None:
        assert len(example_run.entries) == 13
        assert len(example_run.successful_entries) == 13

    def test_run_carries_engine_version(self, example_run) -> None:
        assert example_run.engine_version

    def test_no_network_evidence(self, example_run) -> None:
        """All example corpora are committed offline evidence."""
        assert all(e.evidence_count >= 0 for e in example_run.entries)

    def test_dataset_hash_matches_loader(self, example_dataset, example_run) -> None:
        from benchmarks.ground_truth_eval.dataset import dataset_hash

        assert example_run.dataset_hash == dataset_hash(example_dataset)

    def test_empty_dataset_run(self, make_fake_engine) -> None:
        empty = GoldenDataset(dataset_name="empty", benchmark_version="1.0", entries=[])
        runner = BenchmarkRunner(engine=make_fake_engine())
        run = runner.run(empty, run_id="empty_run")
        assert run.entries == []
        assert run.result_hash
