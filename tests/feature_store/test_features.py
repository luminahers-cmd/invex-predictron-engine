"""Tests for feature computation functions across all categories."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from predictron_engine.dataset.signals.model import SignalType
from predictron_engine.feature_store.features import ALL_FEATURES
from tests.feature_store.conftest import (
    FakeGraph,
    FakeGraphQueries,
    FakeProfile,
    FakeRecord,
    make_signal,
    make_timeline,
)

AS_OF = datetime(2023, 1, 1, tzinfo=UTC)


def build(engine, record=None, **kwargs):
    record = record or FakeRecord(
        profile=FakeProfile(
            founded_year=2019,
            industries=["SaaS"],
            country_code="US",
        ),
        funding_stage_at_analysis="seed",
    )
    return engine.build_company_features(
        record, as_of=AS_OF, **kwargs,
    )


class TestCompanyFeatures:
    def test_company_age_zero_years(self, engine):
        fs = build(engine)
        assert fs.features["company_age"].value == pytest.approx(4.0, abs=0.02)

    def test_company_age_older(self, engine):
        record = FakeRecord(profile=FakeProfile(founded_year=2013))
        fs = build(engine, record)
        assert fs.features["company_age"].value == pytest.approx(10.0, abs=0.02)

    def test_company_age_missing(self, engine):
        record = FakeRecord()
        fs = build(engine, record)
        assert fs.features["company_age"].value is None

    def test_funding_stage_enum(self, engine):
        record = FakeRecord(
            funding_stage_at_analysis="series_a",
        )
        fs = build(engine, record)
        assert fs.features["funding_stage"].value == "series_a"

    def test_funding_stage_missing(self, engine):
        record = FakeRecord()
        fs = build(engine, record)
        assert fs.features["funding_stage"].value is None

    def test_employee_band_explicit(self, engine):
        record = FakeRecord(profile=FakeProfile(employee_range="201-1000"))
        fs = build(engine, record)
        assert fs.features["employee_band"].value == "201-1000"

    def test_employee_band_count(self, engine):
        record = FakeRecord(profile=FakeProfile(employee_count=5))
        fs = build(engine, record)
        assert fs.features["employee_band"].value == "1-10"

    def test_employee_band_large(self, engine):
        record = FakeRecord(profile=FakeProfile(employee_count=5000))
        fs = build(engine, record)
        assert fs.features["employee_band"].value == "1000+"

    def test_operating_country_missing(self, engine):
        record = FakeRecord()
        fs = build(engine, record)
        assert fs.features["operating_country"].value is None

    def test_industry_multi(self, engine):
        record = FakeRecord(profile=FakeProfile(industries=["AI", "SaaS"]))
        fs = build(engine, record)
        assert fs.features["industry"].value == ["AI", "SaaS"]


class TestFundingFeatures:
    def test_total_funding_single(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, amount=1_000_000),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["total_funding"].value == 1_000_000

    def test_total_funding_multiple(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0, amount=1_000_000),
            make_signal(SignalType.FUNDING_ROUND, days_offset=100, amount=2_000_000),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["total_funding"].value == 3_000_000

    def test_total_funding_zero(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, amount=0),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["total_funding"].value == 0.0

    def test_total_funding_none_amounts(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["total_funding"].value == 0.0

    def test_average_round_size_single(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, amount=5_000_000),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["average_round_size"].value == 5_000_000

    def test_average_round_size_none(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["average_round_size"].value is None

    def test_round_count_ignores_non_funding(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, amount=100),
            make_signal(SignalType.ARR_MILESTONE, amount=200),
            make_signal(SignalType.LAYOFFS),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["funding_round_count"].value == 1

    def test_investor_count_dedupes(self, engine):
        tl = make_timeline(signals=[
            make_signal(
                SignalType.FUNDING_ROUND, investors=["Sequoia", "Sequoia", "YC"],
            ),
            make_signal(
                SignalType.FUNDING_ROUND, days_offset=100, investors=["YC"],
            ),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["investor_count"].value == 2

    def test_investor_count_empty(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, amount=100),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["investor_count"].value == 0

    def test_funding_recency_same_day(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["funding_recency"].value == 0.0

    def test_funding_recency_older(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=-365),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["funding_recency"].value == pytest.approx(365.0, abs=1.0)

    def test_funding_recency_missing(self, engine):
        tl = make_timeline(signals=[make_signal(SignalType.LAYOFFS)])
        fs = build(engine, timeline=tl)
        assert fs.features["funding_recency"].value is None


class TestGrowthFeatures:
    def test_funding_velocity_one_round(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["funding_velocity"].value > 0

    def test_funding_velocity_two_rounds(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=-365),
            make_signal(SignalType.FUNDING_ROUND, days_offset=0),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["funding_velocity"].value == pytest.approx(2.0, abs=0.1)

    def test_funding_velocity_none_no_timeline(self, engine):
        fs = build(engine)
        assert fs.features["funding_velocity"].value is None

    def test_hiring_velocity_no_milestones(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["hiring_velocity"].value == 0.0

    def test_milestone_frequency_two_years(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.ARR_MILESTONE, days_offset=-365),
            make_signal(SignalType.ARR_MILESTONE, days_offset=0),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["milestone_frequency"].value == pytest.approx(2.0, abs=0.05)

    def test_growth_consistency_regular(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.ARR_MILESTONE, days_offset=-300),
            make_signal(SignalType.ARR_MILESTONE, days_offset=-200),
            make_signal(SignalType.ARR_MILESTONE, days_offset=-100),
            make_signal(SignalType.ARR_MILESTONE, days_offset=0),
        ])
        fs = build(engine, timeline=tl)
        cv = fs.features["growth_consistency"].value
        assert cv == pytest.approx(0.0, abs=0.01)

    def test_growth_consistency_too_few(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.ARR_MILESTONE, days_offset=-100),
            make_signal(SignalType.ARR_MILESTONE, days_offset=0),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["growth_consistency"].value is None

    def test_momentum_score_positive(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0, confidence=1.0),
            make_signal(SignalType.ARR_MILESTONE, days_offset=-30, confidence=0.9),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["momentum_score"].value > 1.0

    def test_momentum_single_signal(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0, confidence=0.5),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["momentum_score"].value == pytest.approx(0.5, abs=0.001)

    def test_momentum_older_signal_dampening(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=-365, confidence=1.0),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["momentum_score"].value < 1.0


class TestSignalsFeatures:
    def test_signal_frequency_single(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["signal_frequency"].value == 1.0

    def test_signal_frequency_empty(self, engine):
        tl = make_timeline(signals=[])
        fs = build(engine, timeline=tl)
        assert fs.features["signal_frequency"].value == 0.0

    def test_signal_diversity_all_same(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0),
            make_signal(SignalType.FUNDING_ROUND, days_offset=100),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["signal_diversity"].value == 1

    def test_signal_diversity_many_types(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0),
            make_signal(SignalType.ARR_MILESTONE, days_offset=10),
            make_signal(SignalType.LAYOFFS, days_offset=20),
            make_signal(SignalType.ACQUISITION, days_offset=30),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["signal_diversity"].value == 4

    def test_signal_recency_zero(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["signal_recency"].value == 0.0

    def test_signal_recency_missing(self, engine):
        tl = make_timeline(signals=[])
        fs = build(engine, timeline=tl)
        assert fs.features["signal_recency"].value is None

    def test_activity_score_missing_empty(self, engine):
        tl = make_timeline(signals=[])
        fs = build(engine, timeline=tl)
        assert fs.features["activity_score"].value is None

    def test_activity_score_positive(self, engine):
        tl = make_timeline(signals=[
            make_signal(SignalType.FUNDING_ROUND, days_offset=0),
            make_signal(SignalType.ARR_MILESTONE, days_offset=-50),
        ])
        fs = build(engine, timeline=tl)
        assert fs.features["activity_score"].value > 0


class TestKnowledgeGraphFeatures:
    def test_degree_zero(self, engine):
        graph = FakeGraph(_nodes={"rec-1": 0}, node_count=1, edge_count=0)
        fs = build(engine, knowledge_graph=graph)
        assert fs.features["graph_degree"].value == 0

    def test_degree_missing_node(self, engine):
        graph = FakeGraph(_nodes={}, node_count=1, edge_count=0)
        fs = build(engine, knowledge_graph=graph)
        assert fs.features["graph_degree"].value == 0

    def test_density_single_node(self, engine):
        graph = FakeGraph(node_count=1, edge_count=0)
        fs = build(engine, knowledge_graph=graph)
        assert fs.features["graph_density"].value == 0.0

    def test_density_complete(self, engine):
        graph = FakeGraph(node_count=3, edge_count=6)
        fs = build(engine, knowledge_graph=graph)
        assert fs.features["graph_density"].value == 1.0

    def test_component_size_isolated(self, engine):
        queries = FakeGraphQueries(_components=[{"isolated"}])
        fs = build(engine, graph_queries=queries)
        assert fs.features["connected_component_size"].value == 1

    def test_ecosystem_connections_including_investors(self, engine):
        queries = FakeGraphQueries(
            _neighbors={"rec-1": ["inv-1", "inv-2"]},
        )
        fs = build(engine, graph_queries=queries)
        assert fs.features["ecosystem_connections"].value == 2


class TestBenchmarkFeatures:
    def test_similarity_zero(self, engine):
        fs = build(
            engine, benchmark_context={"similarity_score": 0.0},
        )
        assert fs.features["benchmark_similarity"].value == 0.0

    def test_similarity_one(self, engine):
        fs = build(
            engine, benchmark_context={"similarity_score": 1.0},
        )
        assert fs.features["benchmark_similarity"].value == 1.0

    def test_success_rate_value(self, engine):
        fs = build(
            engine, benchmark_context={"historical_success_rate": 0.33},
        )
        assert fs.features["historical_success_rate"].value == 0.33

    def test_sector_accuracy_value(self, engine):
        fs = build(
            engine, benchmark_context={"sector_accuracy": 0.75},
        )
        assert fs.features["sector_accuracy_reference"].value == 0.75

    def test_empty_context_all_none(self, engine):
        fs = build(engine, benchmark_context={})
        assert fs.features["benchmark_similarity"].value is None
        assert fs.features["historical_success_rate"].value is None
        assert fs.features["sector_accuracy_reference"].value is None


class TestFeatureComputationContract:
    def test_all_computors_return_tuple(self):
        from predictron_engine.feature_store.registry import FeatureRegistry
        reg = FeatureRegistry()
        reg.register_all(ALL_FEATURES)
        record = FakeRecord()
        ctx: dict[str, Any] = {}
        for fid in reg.list_ids():
            computor = reg.get_computor(fid)
            defn = reg.get(fid)
            assert computor is not None and defn is not None
            result = computor(record, {}, ctx)
            assert isinstance(result, tuple), fid
            assert len(result) == 2, fid
            value, evidence = result
            assert isinstance(evidence, list), fid

    def test_evidence_structures_valid(self, engine, full_timeline):
        record = FakeRecord(
            profile=FakeProfile(founded_year=2019, industries=["SaaS"]),
            funding_stage_at_analysis="seed",
        )
        fs = engine.build_company_features(
            record, timeline=full_timeline, as_of=AS_OF,
        )
        for snap in fs.features.values():
            for ev in snap.evidence_references:
                assert ev.source_type in {
                    "record", "timeline", "outcome",
                    "knowledge_graph", "benchmark",
                }
                assert ev.source_id != ""

    def test_no_evidence_for_none_values(self, engine):
        record = FakeRecord()
        fs = engine.build_company_features(record, as_of=AS_OF)
        for snap in fs.features.values():
            if snap.value is None:
                assert snap.evidence_references == [], snap.feature_id

    def test_all_features_have_definition(self, engine, fake_record):
        fs = engine.build_company_features(fake_record, as_of=AS_OF)
        for fid in fs.features:
            defn = engine.registry.get(fid)
            assert defn is not None, fid
            assert defn.feature_id == fid
