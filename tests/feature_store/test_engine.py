"""Tests for the feature engine."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from predictron_engine.feature_store.engine import FeatureEngine
from predictron_engine.feature_store.models import (
    FeatureCategory,
    FeatureDefinition,
    FeatureStatus,
    ValueType,
)
from predictron_engine.feature_store.registry import FeatureRegistry
from tests.feature_store.conftest import (
    FakeRecord,
    make_signal,
    make_timeline,
)


def constant_computor(value: Any):
    def _computor(record: Any, deps: dict[str, Any], ctx: dict[str, Any]):
        return value, []
    return _computor


class TestEngineBasics:
    def test_build_company_features(self, engine, fake_record):
        fs = engine.build_company_features(
            fake_record, as_of=datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert fs.company_id == fake_record.record_id
        assert fs.feature_count() > 0
        assert fs.features["company_age"].value == pytest.approx(4.0, abs=0.01)

    def test_company_features_deterministic(self, engine, fake_record):
        as_of = datetime(2024, 1, 1, tzinfo=UTC)
        first = engine.build_company_features(fake_record, as_of=as_of)
        second = engine.build_company_features(fake_record, as_of=as_of)
        assert first.features["company_age"].value == \
            second.features["company_age"].value
        assert first.features["funding_stage"].value == \
            second.features["funding_stage"].value

    def test_company_age_value(self, engine, fake_record):
        fs = engine.build_company_features(
            fake_record, as_of=datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert fs.features["company_age"].value == pytest.approx(4.0, abs=0.02)

    def test_company_age_none_for_missing(self, engine, empty_record):
        fs = engine.build_company_features(empty_record)
        assert fs.features["company_age"].value is None

    def test_funding_stage_value(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        assert fs.features["funding_stage"].value == "seed"

    def test_employee_band(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        assert fs.features["employee_band"].value == "11-50"

    def test_operating_country(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        assert fs.features["operating_country"].value == "US"

    def test_industry(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        assert fs.features["industry"].value == "SaaS"

    def test_record_never_modified(self, engine, fake_record):
        original_profile = fake_record.profile
        original_raw = fake_record.raw_data
        engine.build_company_features(fake_record)
        assert fake_record.profile == original_profile
        assert fake_record.raw_data == original_raw

    def test_snapshot_ids_unique_per_build(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        ids = [s.snapshot_id for s in fs.features.values()]
        assert len(ids) == len(set(ids))

    def test_record_id_preserved(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        assert fs.record_id == fake_record.record_id

    def test_custom_record_id(self, engine, fake_record):
        fs = engine.build_company_features(fake_record, record_id="custom-id")
        assert fs.company_id == fake_record.record_id
        assert fs.record_id == fake_record.record_id


class TestEngineTimelineFeatures:
    def test_funding_round_count(self, engine, fake_record, full_timeline):
        fs = engine.build_company_features(
            fake_record, timeline=full_timeline,
        )
        assert fs.features["funding_round_count"].value == 2

    def test_total_funding(self, engine, fake_record, full_timeline):
        fs = engine.build_company_features(
            fake_record, timeline=full_timeline,
        )
        assert fs.features["total_funding"].value == 6_000_000

    def test_average_round_size(self, engine, fake_record, full_timeline):
        fs = engine.build_company_features(
            fake_record, timeline=full_timeline,
        )
        assert fs.features["average_round_size"].value == 3_000_000

    def test_momentum_score_computed(self, engine, fake_record, full_timeline):
        fs = engine.build_company_features(
            fake_record, timeline=full_timeline,
            as_of=datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert fs.features["momentum_score"].value is not None
        assert fs.features["momentum_score"].value > 0

    def test_momentum_zero_empty_timeline(self, engine, fake_record):
        fs = engine.build_company_features(
            fake_record, timeline=make_timeline(signals=[]),
            as_of=datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert fs.features["momentum_score"].value == 0.0

    def test_signal_diversity(self, engine, fake_record, full_timeline):
        fs = engine.build_company_features(
            fake_record, timeline=full_timeline,
        )
        assert fs.features["signal_diversity"].value >= 3

    def test_hiring_velocity(self, engine, fake_record, full_timeline):
        fs = engine.build_company_features(
            fake_record, timeline=full_timeline,
        )
        assert fs.features["hiring_velocity"].value >= 0

    def test_milestone_frequency(self, engine, fake_record, full_timeline):
        fs = engine.build_company_features(
            fake_record, timeline=full_timeline,
        )
        assert fs.features["milestone_frequency"].value >= 0

    def test_funding_recency(self, engine, fake_record, full_timeline):
        fs = engine.build_company_features(
            fake_record, timeline=full_timeline,
            as_of=datetime(2024, 1, 1, tzinfo=UTC),
        )
        assert fs.features["funding_recency"].value is not None

    def test_signal_recency(self, engine, fake_record, full_timeline):
        fs = engine.build_company_features(
            fake_record, timeline=full_timeline,
            as_of=datetime(2025, 1, 1, tzinfo=UTC),
        )
        assert fs.features["signal_recency"].value >= 0

    def test_signal_frequency(self, engine, fake_record, full_timeline):
        fs = engine.build_company_features(
            fake_record, timeline=full_timeline,
            as_of=datetime(2025, 1, 1, tzinfo=UTC),
        )
        assert fs.features["signal_frequency"].value >= 0

    def test_investor_count(self, engine, fake_record):
        from predictron_engine.dataset.signals.model import SignalType
        timeline = make_timeline(signals=[
            make_signal(
                SignalType.FUNDING_ROUND, days_offset=0, amount=100_000,
                investors=["Sequoia", "YC"],
            ),
            make_signal(
                SignalType.FUNDING_ROUND, days_offset=200, amount=1_000_000,
                investors=["a16z"],
            ),
        ])
        fs = engine.build_company_features(
            fake_record, timeline=timeline,
        )
        assert fs.features["investor_count"].value == 3

    def test_growth_consistency(self, engine, fake_record):
        from predictron_engine.dataset.signals.model import SignalType
        timeline = make_timeline(signals=[
            make_signal(SignalType.ARR_MILESTONE, days_offset=0, amount=1_000_000),
            make_signal(SignalType.ARR_MILESTONE, days_offset=100, amount=2_000_000),
            make_signal(SignalType.ARR_MILESTONE, days_offset=200, amount=4_000_000),
            make_signal(SignalType.ARR_MILESTONE, days_offset=300, amount=8_000_000),
        ])
        fs = engine.build_company_features(
            fake_record, timeline=timeline,
        )
        assert fs.features["growth_consistency"].value is not None

    def test_growth_consistency_none_few(self, engine, fake_record):
        from predictron_engine.dataset.signals.model import SignalType
        timeline = make_timeline(signals=[
            make_signal(SignalType.ARR_MILESTONE, days_offset=0),
            make_signal(SignalType.ARR_MILESTONE, days_offset=100),
        ])
        fs = engine.build_company_features(
            fake_record, timeline=timeline,
        )
        assert fs.features["growth_consistency"].value is None


class TestEngineDependencyFeatures:
    def test_benchmark_similarity_depends(self, engine, fake_record):
        fs = engine.build_company_features(
            fake_record,
            benchmark_context={"similarity_score": 0.85},
        )
        snap = fs.features["benchmark_similarity"]
        assert snap.status == FeatureStatus.COMPUTED
        assert snap.value == 0.85

    def test_benchmark_features_none_without_context(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        assert fs.features["benchmark_similarity"].value is None
        assert fs.features["historical_success_rate"].value is None

    def test_historical_success_rate(self, engine, fake_record):
        fs = engine.build_company_features(
            fake_record,
            benchmark_context={"historical_success_rate": 0.42},
        )
        assert fs.features["historical_success_rate"].value == 0.42

    def test_sector_accuracy(self, engine, fake_record):
        fs = engine.build_company_features(
            fake_record,
            benchmark_context={"sector_accuracy": 0.6},
        )
        assert fs.features["sector_accuracy_reference"].value == 0.6


class TestEngineGraphFeatures:
    def test_graph_degree(self, engine, fake_record):
        from tests.feature_store.conftest import FakeGraph, FakeGraphQueries
        graph = FakeGraph(_nodes={"rec-1": 3}, node_count=10, edge_count=5)
        queries = FakeGraphQueries(
            _components=[{"rec-1", "fnd-1"}],
            _neighbors={"rec-1": ["fnd-1"]},
        )
        fs = engine.build_company_features(
            fake_record, knowledge_graph=graph, graph_queries=queries,
        )
        snap = fs.features["graph_degree"]
        assert snap.status == FeatureStatus.COMPUTED
        assert snap.value == 3

    def test_graph_density(self, engine, fake_record):
        from tests.feature_store.conftest import FakeGraph
        graph = FakeGraph(node_count=10, edge_count=10)
        fs = engine.build_company_features(
            fake_record, knowledge_graph=graph,
        )
        density = fs.features["graph_density"].value
        assert density == pytest.approx(10 / 90, abs=0.01)

    def test_graph_density_empty(self, engine, fake_record):
        from tests.feature_store.conftest import FakeGraph
        graph = FakeGraph(node_count=0, edge_count=0)
        fs = engine.build_company_features(
            fake_record, knowledge_graph=graph,
        )
        assert fs.features["graph_density"].value == 0.0

    def test_connected_component_size(self, engine, fake_record):
        from tests.feature_store.conftest import FakeGraph, FakeGraphQueries
        graph = FakeGraph(node_count=5, edge_count=2)
        queries = FakeGraphQueries(
            _components=[{"rec-1", "x", "y", "z"}, {"other"}],
        )
        fs = engine.build_company_features(
            fake_record, knowledge_graph=graph, graph_queries=queries,
        )
        assert fs.features["connected_component_size"].value == 4

    def test_ecosystem_connections(self, engine, fake_record):
        from tests.feature_store.conftest import FakeGraph, FakeGraphQueries
        graph = FakeGraph(node_count=5, edge_count=4)
        queries = FakeGraphQueries(
            _neighbors={"rec-1": ["a", "b", "c"]},
        )
        fs = engine.build_company_features(
            fake_record, knowledge_graph=graph, graph_queries=queries,
        )
        assert fs.features["ecosystem_connections"].value == 3


class TestEngineFounderFeatures:
    def test_founder_count(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        assert fs.features["founder_count"].value == 2

    def test_founder_count_none(self, engine, empty_record):
        fs = engine.build_company_features(empty_record)
        assert fs.features["founder_count"].value is None

    def test_repeat_founder(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        assert fs.features["repeat_founder_indicator"].value == 0

    def test_repeat_founder_positive(self, engine):
        record = FakeRecord(
            raw_data={"founders": [{"prior_companies": ["X Corp"]}]},
        )
        fs = engine.build_company_features(record)
        assert fs.features["repeat_founder_indicator"].value == 1

    def test_founder_change_count(self, engine, fake_record):
        from tests.feature_store.conftest import FakeOutcome
        outcome = FakeOutcome(
            outcome_events=[
                {"event_type": "founder_change"},
                {"event_type": "other"},
                {"event_type": "founder_change"},
            ],
        )
        fs = engine.build_company_features(fake_record, outcome=outcome)
        assert fs.features["founder_change_count"].value == 2

    def test_founder_change_count_zero(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        assert fs.features["founder_change_count"].value == 0


class TestEngineErrorHandling:
    def test_computation_error_recorded(self):
        reg = FeatureRegistry()

        def broken(record, deps, ctx):
            raise RuntimeError("boom")

        reg.register(
            FeatureDefinition(
                feature_id="broken", feature_name="Broken",
                category=FeatureCategory.COMPANY,
                description="d", value_type=ValueType.FLOAT,
            ),
            broken,
        )
        reg.register(
            FeatureDefinition(
                feature_id="ok", feature_name="OK",
                category=FeatureCategory.COMPANY,
                description="d", value_type=ValueType.FLOAT,
            ),
            constant_computor(1.0),
        )
        eng = FeatureEngine(reg)
        fs = eng.build_company_features(FakeRecord())
        assert fs.features["broken"].status == \
            FeatureStatus.COMPUTATION_ERROR
        assert "boom" in (fs.features["broken"].error_message or "")
        assert fs.features["ok"].status == FeatureStatus.COMPUTED

    def test_missing_dependency_recorded(self):
        reg = FeatureRegistry()
        reg.register(
            FeatureDefinition(
                feature_id="parent", feature_name="Parent",
                category=FeatureCategory.COMPANY,
                description="d", value_type=ValueType.FLOAT,
                dependencies=["missing_dep"],
            ),
            constant_computor(1.0),
        )
        eng = FeatureEngine(reg)
        fs = eng.build_company_features(FakeRecord())
        assert fs.features["parent"].status == \
            FeatureStatus.MISSING_DEPENDENCY

    def test_registry_missing_feature_skipped(self, engine, fake_record):
        record = FakeRecord()
        record.profile.founded_year = 2015
        fs = engine.build_company_features(record)
        assert "does_not_exist" not in fs.features


class TestEngineBuildStore:
    def test_build_feature_store(self, engine):
        records = [
            FakeRecord(record_id="a"),
            FakeRecord(record_id="b"),
            FakeRecord(record_id="c"),
        ]
        snapshot = engine.build_feature_store(records)
        assert snapshot.company_count == 3
        assert snapshot.feature_count == 3 * len(snapshot.companies["a"].features)
        assert snapshot.list_companies() == ["a", "b", "c"]

    def test_empty_records(self, engine):
        snapshot = engine.build_feature_store([])
        assert snapshot.company_count == 0
        assert snapshot.feature_count == 0


class TestEngineRecompute:
    def test_recompute_feature(self, engine, fake_record, full_timeline):
        fs = engine.build_company_features(
            fake_record, timeline=full_timeline,
        )
        new_snap = engine.recompute_feature(
            "total_funding", fake_record.record_id,
            fs.features, fake_record, timeline=full_timeline,
        )
        assert new_snap is not None
        assert new_snap.value == fs.features["total_funding"].value

    def test_recompute_unknown_feature(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        result = engine.recompute_feature(
            "missing", fake_record.record_id, fs.features, fake_record,
        )
        assert result is None

    def test_recompute_dependency_feature(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        new_snap = engine.recompute_feature(
            "company_age", fake_record.record_id, fs.features, fake_record,
            as_of=datetime(2025, 1, 1, tzinfo=UTC),
        )
        assert new_snap is not None
        assert new_snap.value != fs.features["company_age"].value

    def test_recompute_missing_dependency(self, engine, fake_record):
        from predictron_engine.feature_store.models import FeatureDefinition
        reg = FeatureRegistry()
        reg.register(
            FeatureDefinition(
                feature_id="x", feature_name="X",
                category=FeatureCategory.COMPANY, description="d",
                value_type=ValueType.FLOAT, dependencies=["nope"],
            ),
            lambda record, deps, ctx: (1.0, []),
        )
        eng = FeatureEngine(reg)
        from tests.feature_store.conftest import FakeRecord
        result = eng.recompute_feature(
            "x", "rec-1", {}, FakeRecord(),
        )
        assert result is not None
        assert result.status == FeatureStatus.MISSING_DEPENDENCY


class TestEngineDeterminism:
    def test_same_input_same_output_bytes(self, engine, fake_record,
                                          full_timeline):
        as_of = datetime(2024, 1, 1, tzinfo=UTC)
        kwargs = {
            "timeline": full_timeline,
            "as_of": as_of,
            "benchmark_context": {"similarity_score": 0.9},
        }
        first = engine.build_company_features(fake_record, **kwargs)
        second = engine.build_company_features(fake_record, **kwargs)

        def canonical(fs):
            data = fs.model_dump()
            for snap in data["features"].values():
                snap.pop("snapshot_id", None)
            return data

        assert canonical(first) == canonical(second)

    def test_provenance_stable(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        snap = fs.features["company_age"]
        assert snap.provenance["engine_version"] == "0.12.1"
        assert "feature_version" in snap.provenance

    def test_evidence_references_present(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        snap = fs.features["company_age"]
        assert len(snap.evidence_references) == 1
        assert snap.evidence_references[0].source_type == "record"
        assert snap.evidence_references[0].source_id == fake_record.record_id

    def test_evidence_for_timeline_features(self, engine, fake_record,
                                            full_timeline):
        fs = engine.build_company_features(
            fake_record, timeline=full_timeline,
        )
        snap = fs.features["funding_round_count"]
        assert len(snap.evidence_references) == 1
        assert snap.evidence_references[0].source_type == "timeline"


class TestEngineValueTypes:
    def test_int_value_type(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        assert fs.features["founder_count"].value_type == ValueType.INT

    def test_float_value_type(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        assert fs.features["company_age"].value_type == ValueType.FLOAT

    def test_string_value_type(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        assert fs.features["funding_stage"].value_type == ValueType.STRING

    def test_none_value_type(self, engine, empty_record):
        fs = engine.build_company_features(empty_record)
        assert fs.features["company_age"].value_type == ValueType.NONE

    def test_status_default_computed(self, engine, fake_record):
        fs = engine.build_company_features(fake_record)
        assert fs.features["industry"].status == FeatureStatus.COMPUTED
